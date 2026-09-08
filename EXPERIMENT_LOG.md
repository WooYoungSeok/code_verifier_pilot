# Experiment Log

Running record of experimental settings and every change to them. **Any agent or
person picking this project up should read this file first**, then `README.md`.

Rule for this project: **no setting changes silently.** If you change a sampling
parameter, a split, the pair-construction protocol, a prompt, or a model id, add
a dated entry to the Changelog below and say which already-collected results it
invalidates. Results collected under different settings are not comparable and
must not be pooled into one table.

---

## Current settings (authoritative)

### Task

`f(problem, student_code, target_error_type) -> aligned | not_aligned`

Binary, target-conditioned. Not N-way classification.

### Models

| key | model id | provider | group |
|-----|----------|----------|-------|
| `gpt-5.1` | `gpt-5.1` | openai | closed |
| `gemini-2.5-flash` | `gemini-2.5-flash` | gemini | **excluded** -- free-tier quota 20/day, see 2026-09-07 entry |
| `claude-sonnet-4.6` | `claude-sonnet-4-6` | anthropic | closed |
| `qwen2.5-coder-7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` | local | open |
| `deepseek-coder-v2-lite` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | local | **excluded from groups** -- see 2026-09-07 entry |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | local | open (specialisation control) |

Defined in `src/verifier_pilot/clients/registry.py`.

### Sampling parameters

**temperature = 0, seed = 42 everywhere it exists, and one shared output cap.**

`MAX_ANSWER_TOKENS = 64` in `clients/base.py` is used by all four clients, so the
answer budget is not a per-model variable. Measured output maxima: 20 tokens
(gpt-5.1), 35 (claude, tool_use block), 6 (gemini) -- the cap never binds. If a
run ever reports `finish_reason` `length` / `max_tokens`, raise it there for all
models at once and add a changelog entry.

**Answer channel** (must match across models or it confounds open-vs-closed):
gpt-5.1 `response_format` json_schema strict; gemini `response_schema` enum;
claude forced tool use strict; open-weight greedy generation + parse.

| model | temperature | seed | notes |
|-------|-------------|------|-------|
| gpt-5.1 | **0.0** | **42** | `reasoning_effort` deliberately **unset** -- see 2026-09-07 entry |
| gemini-2.5-flash | **0.0** | **42** | `thinking_budget=0`, `safety_threshold=OFF`, `max_output_tokens=256` |
| claude-sonnet-4.6 | **0.0** | **not available** | Messages API has no seed; `thinking` omitted |
| open-weight | n/a (greedy) | 42 | `do_sample=False`, `max_new_tokens=32`, generate + parse |

**Known limitation to state in the write-up:** Claude Sonnet 4.6 cannot be
seed-pinned -- `anthropic.messages.create` exposes no seed parameter, verified
against SDK 0.71. Claude runs may vary slightly between repetitions, while
GPT-5.1 and Gemini are seed-pinned. This is recorded automatically as an
`unsupported_by_provider` parameter event in every Claude run manifest.

### Parameter-change policy

`strict_params=True` is the default. A sampling parameter the API rejects
**aborts the run** with the raw provider error, instead of being dropped
silently. `--allow-param-fallback` permits the drop; the drop is recorded in the
run manifest either way.

Every run writes `outputs/raw/<model>__<dataset>__<condition>.manifest.json`
containing requested parameters, effective parameters, and the full history of
parameter events with raw API errors. The console prints the same
requested-vs-effective table before each run.

### Datasets and splits

| | PyMETA | COJ2022 |
|-|--------|---------|
| source | HF `CircleCat/pymeta` | GitHub `DaSESmartEdu/ErrorCLR` |
| language | Python | C / C++ |
| gold column | `all_errortype2` | `Type` + `SubType` |
| label space | 13 classes (14-class taxonomy minus `No error`) | 22 `Type::SubType` |
| split mode | `problem_disjoint` (**default**) | `problem_disjoint` |
| ratios | 0.7 / 0.1 / 0.2, hash-assigned by problem id, seed 42 | same |

`--split-mode official` exists for PyMETA but **must not be used for the SFT
comparison** -- see the dataset findings in `README.md`.

### Pair construction

Per source submission: 1 positive + 1 random negative + 1 hard negative.
Negatives avoid the submission's entire gold label set. Evaluation uses two
balanced 50:50 sets sharing their positives (`random`, `hard`).
`--cap 30` source submissions per error type for the pilot. Seed 42.

Current test-split sizes at `--cap 30`: PyMETA 387 submissions / 1,161 pairs;
COJ2022 339 submissions / 1,017 pairs. **2,178 calls per model.**

### Conditions

`P1` problem + code, `P2` problem + reference + code (PyMETA).
`C1` code only, `C2` code + repaired reference (COJ2022).

### Metrics

Primary macro F1 over {aligned, not_aligned}. Aligned precision and
`false_aligned_rate` reported as the reward-hacking metrics. Paired cluster
bootstrap, resampling unit = **submission**, 1,000 resamples. API failures
excluded from scoring and reported separately.

---

## Changelog

### 2026-09-07 -- Gemini 2.5 Flash dropped from the pilot: free-tier quota is 20/day

**Decision.** `gemini-2.5-flash` is excluded from the model groups. The closed
arm is gpt-5.1 and claude-sonnet-4.6; the open arm is the Qwen pair.

**Why.** The quota is not a rate limit that backing off can absorb. The 429
carries:

    quotaId    GenerateRequestsPerDayPerProjectPerModel-FreeTier
    quotaValue 20
    model      gemini-2.5-flash

**20 requests per day.** The pilot needs 2,178 per model per condition, i.e.
about 109 days. Measured directly: 12 requests succeeded before the wall, and a
later probe at 8.8 RPM (6.5s spacing) failed 10/10 -- pacing does not help a
daily cap.

The Batch API, which would have its own quota, returns
`400 FAILED_PRECONDITION` on this key even for a minimal request, while
`batches.list()` succeeds -- batch job creation requires billing to be enabled.

**What this costs.** RQ1 loses one of its three frontier models; the closed-vs-open
comparison (RQ2) and the SFT comparison (RQ3) are unaffected, since both need
only a credible closed baseline and gpt-5.1 plus claude-sonnet-4.6 provide it.
RQ4 (PyMETA vs COJ2022) is unaffected.

**To add it back.** Enable billing on the Google AI Studio project, then
`--models gemini-2.5-flash --datasets all`. The per-pair cache means only Gemini
is charged; nothing already collected is re-run. Add a changelog entry when you
do, and note that its results were collected on a different date from the others.

**Also invalidated.** 365 partial gemini records (353 of them 429 failures) were
deleted. No Gemini numbers have been reported.

### 2026-09-07 -- INCIDENT: editing source during a live run killed two models

**What happened.** The closed-model run was launched, then `MAX_ANSWER_TOKENS`
was added to `clients/base.py` while it was still executing. `registry.build()`
imports each provider module lazily, so when the run reached gemini and claude
it read the *new* `gemini.py` / `anthropic_client.py` off disk -- which import
`MAX_ANSWER_TOKENS` -- against the *old* `base` module already resident in
memory, which does not define it:

    [SETUP FAILED] ImportError: cannot import name 'MAX_ANSWER_TOKENS'
                   from 'verifier_pilot.clients.base'

gpt-5.1 finished normally because its module was imported before the edit.
gemini and claude never started, so no results were lost or corrupted -- the
failure was loud and at setup time, not silent mid-run.

**Rule.** Do not edit `src/` while a run is in flight. Either wait for it to
finish, or stop it first. The per-pair cache makes stopping cheap: nothing
already answered is re-paid.

**Consequence for provenance.** Because gemini and claude never ran, they were
re-launched under the unified `MAX_ANSWER_TOKENS = 64`. gpt-5.1's completed
results were produced under the previous per-model cap of 512. See the
token-cap entry: the cap never bound in any run (verified: zero records with
`finish_reason` `length`/`max_tokens`, max 20 output tokens for gpt-5.1), so the
outputs are what a 64 cap would have produced, and the manifests record the
value each run actually used.

### 2026-09-07 -- Output-token cap unified at 64 for every model

**What changed.** `MAX_ANSWER_TOKENS = 64` in `clients/base.py`, used by all four
clients. Previously gpt-5.1 512, claude 1024, gemini 256, open-weight 32.

**Why.** The answer budget was a per-model variable for no reason.

**Does not invalidate anything.** The caps never bound: across the collected
runs every gpt-5.1 response finished with `stop` (max 20 output tokens), every
claude response with `tool_use` (max 35), gemini max 6, and no run reported
`length` or `max_tokens`. A cap that is never reached does not affect the
output, so results collected under the old per-model caps remain valid and
comparable. Verified by inspecting `finish_reason` and `usage.output_tokens` in
every cached record before changing the value.

### 2026-09-07 -- Open-weight answer channel: back to generation + parse

**What changed.** `HFLocalClient` defaults to `constrained=False`: the answer is
now produced by greedy generation (`do_sample=False`, `max_new_tokens=32`) and
parsed from the text, matching how the reward-model evaluations this pilot
follows read open-weight models, and matching what the API models do (they
generate a complete JSON object under a schema).

**Why.** The previous default compared the logits of the first token of each
answer. `not_aligned` tokenises to `['not', '_aligned']`, so the comparison was
between the rare token `aligned` and the very common generic token `not`. The
prior on `not` dominated and the verifier collapsed:

| dataset | true aligned rate | predicted aligned | median logit margin |
|---------|-------------------|-------------------|---------------------|
| PyMETA  | 33.3% | 18.3% | -2.25 |
| COJ2022 | 33.3% | **1.8%** | -4.38 |

At 1.8% the COJ verifier is effectively a constant `not_aligned` predictor. That
is a measurement artefact, not model capability, and it confounded RQ2 -- the
open-vs-closed comparison the arm exists to make, since the closed models were
answering through a completely different channel.

`constrained=True` is still available and now prints a warning naming the two
tokens being compared, so the artefact stays reproducible but cannot be reported
by accident.

**Invalidates.** All `qwen2.5-coder-7b` results (PyMETA P1 1,161 pairs, COJ2022
C1 1,017 pairs). Caches deleted; the PyMETA numbers quoted in conversation
(macro F1 0.5849, aligned recall 0.2972) are void.

### 2026-09-07 -- Excluded models are excluded in code, not just in this log

**What changed.** `registry.GROUPS` now filters through
`EXCLUDED_FROM_GROUPS`, and `resolve()` prints the exclusion reason.

**Why.** The previous entry recorded the decision to drop
`deepseek-coder-v2-lite`, but the `open` group was still derived from every
model with `group == "open"`. `--models open` therefore started a 31 GB
DeepSeek download mid-run, which was killed and deleted with the disk at 98%
full. A decision recorded only in prose is not a decision the code makes.

### 2026-09-07 -- Open-weight scope: Qwen pair only, DeepSeek deferred

**Decision.** The zero-shot open-weight arm runs `qwen2.5-coder-7b` and
`qwen2.5-7b` only. `deepseek-coder-v2-lite` is deferred.

**Why.** Hardware: single RTX 4090, 24 GB VRAM, and only 66 GB free disk.
DeepSeek-Coder-V2-Lite is a 16B MoE needing ~31 GB in bf16, so it does not fit
in 24 GB and would have to run 4-bit while the two Qwen models run bf16. That
precision mismatch confounds RQ2 -- a lower DeepSeek score could be
quantisation rather than model quality -- and the extra 31 GB download would
leave the disk at ~20 GB free.

The Qwen pair is the cleaner comparison anyway: same family, same size, same
precision, differing only in coding specialisation, which is exactly the RQ2
contrast (`Qwen2.5-Coder-7B-Instruct` vs `Qwen2.5-7B-Instruct`).

**To add DeepSeek later:** run it 4-bit via `--load-in-4bit`, and either
re-run both Qwen models 4-bit for a like-for-like comparison or report the
precision difference explicitly. Add a changelog entry when you do.

### 2026-09-07 -- GPT-5.1: reasoning_effort removed, temperature=0 and seed=42 restored

**What changed.** `OpenAIClient` defaults are now `temperature=0.0`,
`reasoning_effort=None`, `seed=42`. Previously `reasoning_effort="low"` with
`temperature=0.0`.

**Why.** The run logged `API rejected 'temperature'; retrying without it`, which
suggested GPT-5.1 could not run at temperature 0. Probing the live API showed
the real cause is a *combination*:

| request | result |
|---------|--------|
| `temperature=0` alone | OK |
| `reasoning_effort="low"` alone | OK |
| `temperature=0` + `reasoning_effort="low"` | **400** `unsupported_value` on `temperature`: does not support 0 with this model, only the default (1) is supported |
| `temperature=0` + `seed=42` | OK |

Setting `reasoning_effort` makes GPT-5.1 validate the call as a reasoning
request, which locks temperature to its default of 1. Unset, it stays on the
non-reasoning path where `temperature=0` is accepted. The earlier
`reasoning_effort="low"` was an unforced choice that cost temperature control.
The previously working script this pilot was based on used `temperature=0` with
no `reasoning_effort`, and was correct.

**Also fixed.** The rejected parameter is now identified from the 400's
`error.param` field rather than by substring-matching the message text, which is
what mis-attributed this conflict to `temperature` in the first place.

**Invalidates.** 5 cached `gpt-5.1 / pymeta / P1` judgements produced under
`reasoning_effort="low"` with temperature at the API default. Cache deleted; no
results had been reported from it.

**If you want reasoning on:** you must also pass `temperature=None`, and you
must add an entry here. GPT-5.1 numbers collected with reasoning are not
comparable with those collected without it.

### 2026-09-07 -- Parameter provenance is recorded, not just printed

**What changed.** Silent parameter fallback removed. `strict_params=True` is the
default and aborts on a rejected sampling parameter; `--allow-param-fallback`
opts back in. Every client records `requested_params`, `effective_params`, and
`param_events` with the raw provider error; the runner writes these to a per-run
`.manifest.json` and prints a requested-vs-effective table.

**Why.** The previous behaviour dropped a rejected parameter, printed one line to
stdout, and left no trace in the results. A verifier scored under a temperature
nobody recorded is a silently invalid experiment.

**Covered by this:** the Gemini safety-threshold fallback, Anthropic
structured-output degradation, Anthropic's missing seed, and any future OpenAI
parameter rejection.

### 2026-09-07 -- Initial build

Repository created. Task reframed from N-way classification to
target-conditioned binary verification. Four dataset findings established
against the released data (PyMETA split leakage, `all_errortype2` as gold,
COJ2022 `SubType` / `Undefined` / multi-error, COJ2022 0-indexed `Line_ID`),
documented in `README.md`.

---

## Open questions / not yet decided

* **Claude seed.** No API-level fix. Options: report as a limitation (current),
  or run Claude n times and report variance. Not yet done.
* **`--cap 30`.** Pilot budget. Rare PyMETA classes (`Other errors` 177,
  `TabError` 162 in the full pool) contribute fewer than 30 at the test split;
  macro-averaging is what makes this acceptable. Revisit if a class ends up with
  very low support in the final test set.
* **COJ2022 `Undefined` (43.6%).** Currently dropped at subtype granularity. The
  `--granularity type` run over all 5,885 programs has not been done and would
  answer whether coarse-level verification is easier.
* **Problem diversity in the PyMETA test split is low.** `problem_disjoint` at
  0.7/0.1/0.2 puts 387 submissions in test but only **25 distinct questionIds**
  (COJ2022 test has 83 problems from 339 programs). The bootstrap clusters by
  submission, so the CIs do not reflect problem-level variance, and results
  should be read as "on these 25 problems". Raising `--cap` does not help --
  it adds submissions, not problems. Options: widen the test ratio, or add a
  problem-level bootstrap. Not yet decided.
* **P2 / C2 ablation.** Not yet run.
* **SFT.** Not yet run. Train one adapter per dataset; do not pool the taxonomies.

---

## Results collected so far

| date | model | dataset | condition | status |
|------|-------|---------|-----------|--------|
| 2026-09-07 | all closed | -- | -- | smoke test passed, 2 fixtures each |
| -- | -- | -- | -- | **no full runs completed yet** |

Add a row when a run finishes, and point at its manifest so the settings stay
recoverable.
