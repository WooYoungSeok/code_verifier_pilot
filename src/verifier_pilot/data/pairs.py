"""Verifier pair construction.

Each source submission yields one *positive* and up to two *negatives*:

    positive       (problem, code, e*)   -> aligned
    random negative(problem, code, e^-)  -> not_aligned,  e^- ~ U(labels \\ gold)
    hard negative  (problem, code, e^h)  -> not_aligned,  e^h ~ U(siblings \\ gold)

The random-negative protocol reproduces Learning to Be Wrong. The hard negative
is the discriminative test: for COJ2022 it is a different sub-type under the same
coarse ``Type``; for PyMETA it comes from a curated confusion group
(``IndexError`` vs ``KeyError``, ``NameError`` vs ``UnboundLocalError``, ...).
Without it a verifier that only notices "some runtime error exists" scores far
higher than it deserves.

Evaluation uses two balanced sets that share their positives:

    random set = positives + random negatives   (50:50)
    hard set   = positives + hard negatives     (50:50)

Because the two sets share rows, significance testing must cluster by
``submission_id`` -- see evaluation.bootstrap.

Negatives are drawn from the complement of the submission's **whole** gold set,
which matters for COJ2022 where 269 of 3,809 programs carry more than one
distinct label.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Callable, Iterable

POSITIVE = "positive"
RANDOM = "random"
HARD = "hard"
NEGATIVE_TYPES = (RANDOM, HARD)

ALIGNED = "aligned"
NOT_ALIGNED = "not_aligned"


def _rng(seed: int, *parts: str) -> random.Random:
    """Deterministic RNG keyed by content, not by iteration order.

    Sampling for a submission is therefore identical whether or not other
    submissions were filtered out upstream.
    """
    key = "|".join((str(seed),) + parts)
    digest = hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _pair_id(submission_id: str, negative_type: str, label: str) -> str:
    raw = f"{submission_id}|{negative_type}|{label}"
    return hashlib.md5(raw.encode("utf-8", "replace")).hexdigest()[:20]


def cap_per_label(
    records: list[dict],
    cap: int | None,
    seed: int = 42,
) -> list[dict]:
    """Keep at most ``cap`` submissions per primary gold label.

    The pilot budget is "up to N original error instances per error type"; rare
    classes simply contribute everything they have, which is why the headline
    metric is macro-averaged.
    """
    if cap is None:
        return list(records)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[record["gold_labels"][0]].append(record)
    kept: list[dict] = []
    for label, bucket in buckets.items():
        bucket = sorted(bucket, key=lambda r: r["submission_id"])
        if len(bucket) > cap:
            _rng(seed, "cap", label).shuffle(bucket)
            bucket = bucket[:cap]
        kept.extend(bucket)
    kept.sort(key=lambda r: r["submission_id"])
    return kept


def build_pairs(
    records: Iterable[dict],
    label_space: list[str],
    hard_pool_fn: Callable[[str], list[str]],
    describe_fn: Callable[[str], str],
    seed: int = 42,
    negative_types: tuple[str, ...] = NEGATIVE_TYPES,
) -> list[dict]:
    """Expand submissions into labelled verifier pairs."""
    unknown = set(negative_types) - set(NEGATIVE_TYPES)
    if unknown:
        raise ValueError(f"unknown negative types {sorted(unknown)}")

    pairs: list[dict] = []
    for record in records:
        gold = list(record["gold_labels"])
        if not gold:
            continue

        # --- positive -------------------------------------------------------
        target = gold[0] if len(gold) == 1 else _rng(seed, record["submission_id"], "pos").choice(gold)
        pairs.append(_make_pair(record, target, ALIGNED, POSITIVE, describe_fn))

        # --- negatives ------------------------------------------------------
        gold_set = set(gold)
        for negative_type in negative_types:
            if negative_type == RANDOM:
                pool = [x for x in label_space if x not in gold_set]
            else:
                sibling = [x for candidate in gold for x in hard_pool_fn(candidate)]
                pool = sorted({x for x in sibling if x not in gold_set})
            if not pool:
                continue
            choice = _rng(seed, record["submission_id"], negative_type).choice(sorted(pool))
            pairs.append(_make_pair(record, choice, NOT_ALIGNED, negative_type, describe_fn))

    pairs.sort(key=lambda p: (p["submission_id"], p["negative_type"]))
    return pairs


def _make_pair(
    record: dict,
    target: str,
    label: str,
    negative_type: str,
    describe_fn: Callable[[str], str],
) -> dict:
    return {
        "pair_id": _pair_id(record["submission_id"], negative_type, target),
        "dataset": record["dataset"],
        "submission_id": record["submission_id"],
        "problem_id": record["problem_id"],
        "split": record.get("split"),
        "problem": record.get("problem"),
        "reference_code": record.get("reference_code"),
        "reference_verified": record.get("reference_verified", True),
        "student_code": record["student_code"],
        "language": record["language"],
        "gold_labels": list(record["gold_labels"]),
        "target_label": target,
        "target_description": describe_fn(target),
        "label": label,
        "negative_type": negative_type,
    }


def evaluation_set(pairs: list[dict], negative_type: str) -> list[dict]:
    """The balanced set for one negative protocol: positives + those negatives."""
    if negative_type not in NEGATIVE_TYPES:
        raise ValueError(f"expected one of {NEGATIVE_TYPES}, got {negative_type!r}")
    keep = {POSITIVE, negative_type}
    return [p for p in pairs if p["negative_type"] in keep]


def summarise(pairs: list[dict]) -> dict:
    """Counts used by the prepare/report scripts."""
    by_type: dict[str, int] = defaultdict(int)
    by_label: dict[str, int] = defaultdict(int)
    for pair in pairs:
        by_type[pair["negative_type"]] += 1
        by_label[pair["label"]] += 1
    return {
        "total_pairs": len(pairs),
        "submissions": len({p["submission_id"] for p in pairs}),
        "problems": len({p["problem_id"] for p in pairs}),
        "by_negative_type": dict(sorted(by_type.items())),
        "by_label": dict(sorted(by_label.items())),
    }
