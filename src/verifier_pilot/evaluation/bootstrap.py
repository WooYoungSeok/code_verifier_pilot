"""Paired bootstrap confidence intervals, clustered by submission.

Two reasons the resampling unit is the **submission**, not the pair row:

1. A submission's positive and its negatives are built from the same program.
   They are not independent, so resampling rows would understate the variance
   and produce CIs that are too narrow.
2. The random-negative and hard-negative evaluation sets *share* their
   positives. Comparing them row-wise would double-count those rows.

Model-vs-model comparison is **paired**: every model answers the same pairs, so
each bootstrap replicate resamples a set of submissions once and scores both
models on that same resample. Pairing removes the between-submission variance
that both models share and is far more sensitive than comparing two independent
CIs -- with a few hundred submissions, overlapping marginal CIs routinely hide a
real and consistent difference.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .metrics import core_metrics

DEFAULT_RESAMPLES = 1000
DEFAULT_ALPHA = 0.05


def _cluster(records: list[dict]) -> dict[str, list[dict]]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        clusters[record["submission_id"]].append(record)
    return clusters


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_ci(
    records: list[dict],
    metric: str = "macro_f1",
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict:
    """Cluster bootstrap CI for one metric on one model's records."""
    clusters = _cluster(records)
    keys = sorted(clusters)
    if not keys:
        return {"metric": metric, "point": None, "ci_low": None, "ci_high": None, "n_clusters": 0}

    point = core_metrics(records).get(metric)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample: list[dict] = []
        for _ in range(len(keys)):
            sample.extend(clusters[keys[rng.randrange(len(keys))]])
        value = core_metrics(sample).get(metric)
        if value is not None:
            values.append(value)

    return {
        "metric": metric,
        "point": point,
        "ci_low": _percentile(values, alpha / 2),
        "ci_high": _percentile(values, 1 - alpha / 2),
        "n_clusters": len(keys),
        "n_resamples": len(values),
    }


def paired_comparison(
    records_a: list[dict],
    records_b: list[dict],
    metric: str = "macro_f1",
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict:
    """Paired cluster bootstrap for ``metric(A) - metric(B)``.

    Only submissions present in both runs are used, so the comparison is exactly
    like-for-like. Reports the difference CI and a two-sided bootstrap p-value
    (the proportion of replicates on the opposite side of zero, doubled).
    """
    clusters_a, clusters_b = _cluster(records_a), _cluster(records_b)
    keys = sorted(set(clusters_a) & set(clusters_b))
    if not keys:
        return {"metric": metric, "diff": None, "n_clusters": 0, "note": "no shared submissions"}

    shared_a = [r for k in keys for r in clusters_a[k]]
    shared_b = [r for k in keys for r in clusters_b[k]]
    point_a = core_metrics(shared_a).get(metric)
    point_b = core_metrics(shared_b).get(metric)
    point_diff = None if point_a is None or point_b is None else point_a - point_b

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(resamples):
        picked = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        sample_a = [r for k in picked for r in clusters_a[k]]
        sample_b = [r for k in picked for r in clusters_b[k]]
        value_a = core_metrics(sample_a).get(metric)
        value_b = core_metrics(sample_b).get(metric)
        if value_a is not None and value_b is not None:
            diffs.append(value_a - value_b)

    if not diffs:
        return {"metric": metric, "diff": point_diff, "n_clusters": len(keys)}

    n_le_zero = sum(1 for d in diffs if d <= 0)
    n_ge_zero = sum(1 for d in diffs if d >= 0)
    p_value = min(1.0, 2 * min(n_le_zero, n_ge_zero) / len(diffs))
    ci_low = _percentile(diffs, alpha / 2)
    ci_high = _percentile(diffs, 1 - alpha / 2)

    return {
        "metric": metric,
        "a": point_a,
        "b": point_b,
        "diff": point_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": bool(ci_low > 0 or ci_high < 0),
        "n_clusters": len(keys),
        "n_resamples": len(diffs),
    }


def compare_all(
    records_by_model: dict[str, list[dict]],
    metric: str = "macro_f1",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> list[dict]:
    """Every unordered model pair, ranked by effect size."""
    names = sorted(records_by_model)
    out: list[dict] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            result = paired_comparison(
                records_by_model[a], records_by_model[b],
                metric=metric, resamples=resamples, seed=seed,
            )
            result["model_a"], result["model_b"] = a, b
            out.append(result)
    out.sort(key=lambda r: -abs(r.get("diff") or 0))
    return out
