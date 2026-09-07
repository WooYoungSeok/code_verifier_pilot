"""Evaluation runner: resumable, cached, leakage-checked.

Every judgement is appended to a JSONL cache keyed by
``(model, dataset, condition, pair_id)``. Re-running skips anything already
cached, so an interrupted run -- exhausted quota, a laptop closing, Ctrl-C --
resumes instead of re-paying for completed calls. ``--refresh-failed`` retries
only the rows that failed.

Before each request the rendered prompt goes through
``data.leakage.assert_no_leakage``, so a loader regression cannot quietly turn
the experiment into "read the traceback we handed you".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from ..clients.base import VerifierClient
from ..data.leakage import assert_no_leakage
from ..prompts import render

CACHE_VERSION = 1


def cache_path(output_dir: Path, model: str, dataset: str, condition: str) -> Path:
    safe = model.replace("/", "_")
    return output_dir / "raw" / f"{safe}__{dataset}__{condition}.jsonl"


def load_cache(path: Path) -> dict[str, dict]:
    """Read cached judgements, tolerating a truncated final line."""
    if not path.is_file():
        return {}
    cached: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue          # interrupted mid-write; the pair is simply re-run
            if "pair_id" in record:
                cached[record["pair_id"]] = record
    return cached


def run(
    client: VerifierClient,
    pairs: list[dict],
    dataset: str,
    condition: str,
    output_dir: Path,
    resume: bool = True,
    refresh_failed: bool = False,
    limit: int | None = None,
    sleep_s: float = 0.0,
    check_leakage: bool = True,
    progress: bool = True,
) -> list[dict]:
    """Evaluate ``client`` on ``pairs``. Returns one record per pair."""
    path = cache_path(output_dir, client.name, dataset, condition)
    path.parent.mkdir(parents=True, exist_ok=True)

    cached = load_cache(path) if resume else {}
    if refresh_failed:
        cached = {k: v for k, v in cached.items() if v.get("prediction") in ("aligned", "not_aligned")}

    todo = [p for p in pairs if p["pair_id"] not in cached]
    if limit is not None:
        todo = todo[:limit]

    if cached:
        print(f"  cache: {len(cached)} done, {len(todo)} to run -> {path.name}", flush=True)

    iterator = tqdm(todo, desc=f"{client.name} {dataset}/{condition}", disable=not progress)
    with path.open("a", encoding="utf-8") as handle:
        for pair in iterator:
            query = render(pair, condition)
            if check_leakage:
                assert_no_leakage(
                    query.user, dataset,
                    gold_labels=pair.get("gold_labels"),
                    target_label=pair.get("target_label"),
                )

            prediction = client.predict(query.system, query.user)
            record = _record(pair, prediction, client, dataset, condition)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            cached[pair["pair_id"]] = record

            if progress:
                mark = "OK" if record["correct"] else ("!!" if not prediction.ok else "XX")
                iterator.set_postfix_str(f"{mark} {record['prediction']}")
            if sleep_s:
                time.sleep(sleep_s)

    return [cached[p["pair_id"]] for p in pairs if p["pair_id"] in cached]


def _record(pair: dict, prediction, client: VerifierClient, dataset: str, condition: str) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "pair_id": pair["pair_id"],
        "model": client.name,
        "provider": client.provider,
        "dataset": dataset,
        "condition": condition,
        "submission_id": pair["submission_id"],
        "problem_id": pair["problem_id"],
        "split": pair.get("split"),
        "target_label": pair["target_label"],
        "gold_labels": pair["gold_labels"],
        "negative_type": pair["negative_type"],
        "label": pair["label"],
        "prediction": prediction.label,
        "correct": prediction.label == pair["label"],
        "ok": prediction.ok,
        "raw": prediction.raw[:400],
        "error_kind": prediction.error_kind,
        "error_detail": prediction.error_detail,
        "finish_reason": prediction.finish_reason,
        "attempts": prediction.attempts,
        "latency_s": prediction.latency_s,
        "usage": prediction.usage,
    }


def load_records(output_dir: Path, model: str, dataset: str, condition: str) -> list[dict]:
    """Read back a completed run for reporting."""
    return list(load_cache(cache_path(output_dir, model, dataset, condition)).values())
