# Can LLMs Serve as Reliable Error Verifiers for Student Code?

A pilot study on **PyMETA** (Python interpreter errors) and **COJ2022** (C/C++
semantic implementation errors).

> **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) is the authoritative record of current
> experimental settings and every change to them.** Read it before running
> anything, and add a dated entry whenever you change a sampling parameter, a
> split, the pair protocol, a prompt, or a model id.

The task is **target-conditioned binary alignment verification**, not error
classification:

```
f(problem, student_code, target_error_type) -> aligned | not_aligned
```

This is the function an RL reward model needs. A student simulator generates
erroneous code conditioned on a target error type; the verifier decides whether
the generated code actually exhibits that error. PyMETA's own experiments are
14-way classification -- reframing it as conditioned verification is the point
of difference, and it makes the two datasets comparable despite their taxonomies
sitting at different levels of abstraction.

---

## Research questions

| RQ  | Question |
|-----|----------|
| RQ1 | Are closed-source frontier LLMs adequate verifiers with prompting alone? |
| RQ2 | Do code-specialised open-weight models match them zero-shot? |
| RQ3 | Does verifier-specific SFT push open models past the closed ones? |
| RQ4 | How does performance differ between observable execution failures (PyMETA) and semantic implementation errors (COJ2022)? |

Cross-cutting: does a **hard negative** (a plausible sibling error type) break
verifiers that a random negative does not? And is a reference solution required?

---

## Findings from dataset validation (read before running)

These were established by checking the released data directly, and they change
how the pilot must be run.

### 1. PyMETA's released splits are not problem-disjoint

Verified against `CircleCat/pymeta@main`:

* All **145** test `questionId`s also occur in `train`.
* **217** `(questionId, studentAnswer)` pairs are byte-identical across train
  and test -- 8.8% of the test set's unique erroneous submissions.

Evaluating an SFT'd verifier under the official splits scores it on problems,
and sometimes on exact programs, that it trained on. That would inflate RQ3
specifically -- the comparison the pilot exists to make.

`--split-mode problem_disjoint` (the default) re-splits by `questionId` and
de-duplicates; `--split-mode official` reproduces the released splits.

### 2. Gold labels come from `all_errortype2`, not `error_category`

In the released CSVs `error_category` holds only `error` / `no_error`, despite
the dataset card describing it as the single-error label. `all_errortype2`
carries the real labels: **19** raw values, collapsing to a **14-class**
taxonomy once the rare tail (`AttributeError`, `RuntimeError`, `SyntaxWarning`,
`MemoryError`, `ZeroDivisionError`, `ModuleNotFoundError`) folds into
`Other errors`. Dropping `No error` leaves the **13** classes the verifier sees.

### 3. COJ2022 needs three corrections

* The column is **`SubType`**, not `Sub_Type`.
* **43.6%** of the 8,511 annotated errors are `<Type>::Undefined` -- annotated
  at the coarse level only. They are excluded from sub-type work by default
  (`--granularity subtype`), leaving 3,809 usable programs over 22
  `Type::SubType` labels; `--granularity type` keeps all 5,885 over 4 labels.
* **Programs carry multiple errors.** 5,912 programs hold 8,511 errors. Gold is
  a label *set*, so a negative must avoid every label in it, not just one.

### 4. `Line_ID` is 0-indexed, and an empty `Buggy_Line` means insertion

Needed to reconstruct the repaired reference program for condition C2. Verified:
99.05% of the 7,350 rows with a non-empty `Buggy_Line` match at
`lines[Line_ID]`; 1-indexing matches 0.1%. The 1,105 rows with an empty
`Buggy_Line` are insertions. Every replacement is checked against `Buggy_Line`
before being applied, and a mismatch degrades the reference rather than
corrupting it (3,778 / 3,809 reconstruct fully verified).

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in the keys you need
```

`.env` lives at the **repository root** and is git-ignored. Keys are resolved by
`verifier_pilot.env`, which walks up to the repo root -- the reference script
this pilot started from loaded `.env` from the script's own directory, so a
root-level `.env` was silently never read.

---

## Running the pilot

```bash
# 1. download PyMETA + COJ2022 (~60 MB, into git-ignored data/)
python scripts/prepare_data.py

# 2. build the evaluation pairs (30 source submissions per error type)
python scripts/build_pairs.py --split test --cap 30

# 3. verify API access cheaply BEFORE spending quota
python scripts/smoke_test.py --models closed

# 4. Experiment 1 -- zero-shot verifiers
python scripts/run_eval.py --models closed --datasets all
python scripts/run_eval.py --models open   --datasets all     # needs a GPU

# 5. Experiment 3 -- context ablation
python scripts/run_eval.py --models closed --datasets pymeta  --conditions P1 P2
python scripts/run_eval.py --models closed --datasets coj2022 --conditions C1 C2

# 6. results
python scripts/report.py --bootstrap 1000 --compare
```

### Experiment 2 -- open-weight verifier SFT

```bash
python scripts/build_pairs.py --split train --cap none
python scripts/build_pairs.py --split dev   --cap none

python scripts/train_sft.py --dataset pymeta  --model qwen2.5-coder-7b
python scripts/train_sft.py --dataset coj2022 --model qwen2.5-coder-7b

python scripts/run_eval.py --models qwen2.5-coder-7b \
    --adapter runs/sft/pymeta_qwen2.5-coder-7b/adapter \
    --name qwen2.5-coder-7b-sft --datasets pymeta
```

Train **one adapter per dataset**. The two taxonomies describe different kinds
of thing; a single verifier over their union would have to treat one label space
as two levels of abstraction at once.

---

## Design

### Pairs

Each source submission yields one positive and two negatives:

| Kind | Target | Truth |
|------|--------|-------|
| positive | the submission's own error type | `aligned` |
| random negative | uniform over all other labels | `not_aligned` |
| hard negative | a sibling of the true type | `not_aligned` |

Hard negatives are the discriminative test. For COJ2022 a sibling is another
sub-type under the same coarse `Type` (`Control::WrongOperator` vs
`Control::OffByOneError`); for PyMETA they come from curated confusion groups
(`IndexError` / `KeyError`, `NameError` / `UnboundLocalError`,
`SyntaxError` / `IndentationError` / `TabError`). Without them a verifier that
only notices "some runtime error exists" scores far better than it deserves.

Evaluation uses two balanced 50:50 sets that **share their positives**:
`random = positives + random negatives`, `hard = positives + hard negatives`.

### Conditions

| | PyMETA | COJ2022 |
|-|--------|---------|
| primary | **P1** problem + student code | **C1** student code only |
| upper bound | **P2** problem + reference + student code | **C2** student code + repaired reference |

COJ2022 ships no natural-language problem statement, only a `Problem_ID` --
which is why its primary condition is code-only. If C2 is much stronger than C1,
the limit is missing problem context rather than verifier capability.

### Leakage control

Neither dataset's answer columns ever reach the model. Enforced twice: loaders
copy only allow-listed fields, and every rendered prompt passes
`data.leakage.assert_no_leakage` before the request goes out.

Blocked for PyMETA: `testOutcome`, `status`, `R_errorcount`, `R_traceback`,
`R_errortype`, `C_errormessage`, `C_errortype`, `all_errortype`,
`all_errortype2`, `error_category`, `state`.
Blocked for COJ2022: `Type`, `SubType`, `Buggy_Line`, `Line_ID`,
`Repaired_Line`, `Result`.

`Buggy_Line` and `Line_ID` are withheld deliberately: handing over the defect's
location would make the task localisation-assisted classification, which is far
easier than what an RL verifier actually faces.

### Metrics

Primary is **macro F1** over {aligned, not_aligned}. The metric that decides
whether a verifier is usable as a reward is **aligned precision** -- a false
`aligned` tells the policy that a program lacking the requested error has it,
and a policy optimising against that verifier will find exactly those cases.
`false_aligned_rate` is reported for the same reason.

Significance uses a **paired cluster bootstrap** with the **submission** as the
resampling unit, because a submission's positive and negatives are built from
the same program and the random/hard sets share their positives. Pairing matters:
with a few hundred submissions, overlapping marginal CIs routinely hide a real
and consistent difference.

Per-error-type results are always reported. A verifier strong in aggregate but
blind to `LogicError` or `OffByOneError` is not usable as a reward for those
categories. Where a slice contains only one true class, accuracy is shown
instead of macro F1 and marked `a`.

API failures are counted and reported but **excluded** from scoring -- treating
a quota error as a wrong answer confounds model quality with run health.

### Sampling parameters are part of the record

All models run at **temperature 0** and **seed 42** wherever those exist. Two
provider limits are recorded rather than hidden: Claude Sonnet 4.6 has no `seed`
parameter, and GPT-5.1 rejects `temperature=0` if `reasoning_effort` is set (so
`reasoning_effort` is left unset -- see EXPERIMENT_LOG.md).

A parameter the API rejects is **never dropped silently**. By default the run
aborts with the raw provider error; `--allow-param-fallback` permits the drop.
Either way every run writes
`outputs/raw/<model>__<dataset>__<condition>.manifest.json` with the requested
parameters, the effective parameters, and every parameter event with its raw
API error, and prints the same requested-vs-effective table to the console.

---

## Why the Gemini call was failing

`clients/gemini.py` fixes five separate causes, any one of which produces the
"just ran it and got an error" symptom:

1. **Thinking consumes the whole output budget.** gemini-2.5-flash thinks by
   default with a dynamic budget. With a response schema it can spend every
   output token on thinking, return `finish_reason: MAX_TOKENS` and zero text
   parts. `response.text` is then `None`, and `response.text.strip()` raises
   `AttributeError` -- which is not recognisably transient, so a naive retry
   classifier gives up immediately and records `error` for the entire run.
   Fixed with `ThinkingConfig(thinking_budget=0)` plus an explicit
   `max_output_tokens`; a binary classifier needs no reasoning tokens.
2. **`BLOCK_NONE` is not always grantable.** Projects without the relevant
   allowlist get a 400. The 2.5 models take `OFF`. The client tries `OFF`, falls
   back to `BLOCK_NONE`, then to no safety settings, and caches what worked.
3. **`response.text` is `None` on a safety block too.** Student code gets
   flagged routinely. `prompt_feedback` / `finish_reason` are inspected and a
   typed reason is reported instead of an `AttributeError`.
4. **The key was never loaded.** `.env` was read from the script's directory
   rather than the repo root.
5. **Transient 429 / 503 / 500.** Retried with jittered exponential backoff.

Empty responses now surface as a typed `EmptyResponse` carrying the finish
reason and the settings that caused it, rather than a bare `error`.

The OpenAI and Anthropic clients take the same defensive approach to their own
quirks: GPT-5.1 rejects some parameters depending on snapshot, so
`temperature` / `reasoning_effort` / `response_format` are each dropped
individually on an "unsupported parameter" 400 and the drop is logged; Claude
uses forced tool use for structured output (the pinned SDK predates
`output_config`) and degrades to text parsing if that is refused.
**Run `scripts/smoke_test.py` first** -- it catches all of this for a few cents.

---

## Layout

```
src/verifier_pilot/
  taxonomy/      pymeta.py, coj2022.py     label spaces, descriptions, hard-negative groups
  data/          pymeta.py, coj2022.py     loaders and normalisation
                 pairs.py                  positive / random / hard construction
                 leakage.py                answer-column deny-lists and prompt tripwires
  clients/       gemini.py, openai_client.py, anthropic_client.py, hf_local.py, registry.py
  evaluation/    runner.py                 resumable cached execution
                 metrics.py, bootstrap.py  macro F1, aligned precision, paired CIs
  sft/           build_dataset.py, train_lora.py
scripts/         prepare_data, build_pairs, smoke_test, run_eval, train_sft, report
tests/           42 tests, no network or API keys required
```

Runs are **resumable**: every judgement is cached to
`outputs/raw/<model>__<dataset>__<condition>.jsonl`, and re-running skips
completed pairs, so an interrupted run costs nothing to continue.
`--refresh-failed` retries only the failures.

```bash
python -m pytest tests/ -q
```

## Data sources

* PyMETA -- <https://huggingface.co/datasets/CircleCat/pymeta>
* COJ2022 -- <https://github.com/DaSESmartEdu/ErrorCLR> (ErrorCLR, Han et al., SIGIR 2023)

Neither is redistributed here; `data/` is git-ignored and fetched by
`scripts/prepare_data.py`.
