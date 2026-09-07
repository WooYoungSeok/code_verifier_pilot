"""Build the verifier SFT set from pairs.

Mirrors the training-set construction of the study this pilot replicates: each
training submission contributes one positive and one negative, giving an exactly
balanced 50:50 set.

Two deliberate departures, both switchable:

``negatives="both"`` (default)
    Train on the random *and* the hard negative. Training only on random
    negatives teaches "is there any error of roughly this flavour", which is the
    behaviour the hard-negative test set is designed to catch. Use
    ``negatives="random"`` to reproduce the original protocol exactly.

``PyMETA verifiers and COJ verifiers are trained separately.``
    The two taxonomies sit at different levels of abstraction -- Python
    interpreter exceptions vs. semantic implementation errors -- so a single
    mixed verifier would have to learn a label space whose members mean
    different kinds of thing. Build one adapter per dataset.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..prompts import render


def build_examples(
    pairs: list[dict],
    condition: str,
    negatives: str = "both",
    balance: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Render pairs into chat-format SFT examples."""
    if negatives not in ("random", "hard", "both"):
        raise ValueError(f"negatives must be random|hard|both, got {negatives!r}")

    keep = {"positive"}
    keep |= {"random", "hard"} if negatives == "both" else {negatives}
    selected = [p for p in pairs if p["negative_type"] in keep]

    if balance:
        selected = _balance(selected, seed)

    examples: list[dict] = []
    for pair in selected:
        query = render(pair, condition)
        examples.append({
            "pair_id": pair["pair_id"],
            "submission_id": pair["submission_id"],
            "problem_id": pair["problem_id"],
            "dataset": pair["dataset"],
            "condition": condition,
            "negative_type": pair["negative_type"],
            "target_label": pair["target_label"],
            "messages": [
                {"role": "system", "content": query.system},
                {"role": "user", "content": query.user},
                {"role": "assistant", "content": pair["label"]},
            ],
            "label": pair["label"],
        })
    return examples


def _balance(pairs: list[dict], seed: int) -> list[dict]:
    """Downsample the majority class to exactly 50:50.

    With ``negatives="both"`` there are two negatives per positive, so without
    this the model can reach 67% by answering not_aligned unconditionally.
    Negatives are dropped alternately from the random and hard pools so both
    protocols stay represented.
    """
    import random

    positives = [p for p in pairs if p["label"] == "aligned"]
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        if pair["label"] != "aligned":
            by_kind[pair["negative_type"]].append(pair)

    rng = random.Random(seed)
    for bucket in by_kind.values():
        rng.shuffle(bucket)

    target = len(positives)
    kinds = sorted(by_kind)
    chosen: list[dict] = []
    index = 0
    while len(chosen) < target and any(by_kind[k] for k in kinds):
        kind = kinds[index % len(kinds)]
        if by_kind[kind]:
            chosen.append(by_kind[kind].pop())
        index += 1

    out = positives + chosen
    out.sort(key=lambda p: p["pair_id"])
    return out


def write_jsonl(examples: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def describe(examples: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    kinds: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example["label"]] += 1
        kinds[example["negative_type"]] += 1
    return {
        "n": len(examples),
        "by_label": dict(sorted(counts.items())),
        "by_negative_type": dict(sorted(kinds.items())),
        "submissions": len({e["submission_id"] for e in examples}),
        "problems": len({e["problem_id"] for e in examples}),
    }
