"""Verifier metrics.

Primary metric is **macro F1 over {aligned, not_aligned}**, matching the
verifier comparison this pilot replicates.

The metric that decides whether a verifier is usable as an RL reward is
**aligned precision**. A false "aligned" tells the policy that a program which
does *not* show the requested error does show it; a policy optimising against
that verifier will find and exploit exactly those cases. Aligned recall costs
sample efficiency; aligned precision costs correctness of the reward itself.

API failures are counted and reported but excluded from the scored set --
scoring them as wrong answers would confound model quality with quota problems.
"""

from __future__ import annotations

from collections import defaultdict

from sklearn.metrics import classification_report, confusion_matrix

ALIGNED = "aligned"
NOT_ALIGNED = "not_aligned"
LABELS = [ALIGNED, NOT_ALIGNED]


def _scored(records: list[dict]) -> list[dict]:
    """Rows with a usable prediction."""
    return [r for r in records if r.get("prediction") in LABELS]


def core_metrics(records: list[dict]) -> dict:
    """Accuracy, macro P/R/F1, and the per-class breakdown."""
    scored = _scored(records)
    if not scored:
        return {
            "n_scored": 0, "n_total": len(records), "n_failed": len(records),
            "accuracy": None, "macro_f1": None,
        }

    y_true = [r["label"] for r in scored]
    y_pred = [r["prediction"] for r in scored]
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    # rows = truth, cols = prediction, order [aligned, not_aligned]
    tp_aligned, fn_aligned = int(matrix[0][0]), int(matrix[0][1])
    fp_aligned, tn_aligned = int(matrix[1][0]), int(matrix[1][1])

    return {
        "n_total": len(records),
        "n_scored": len(scored),
        "n_failed": len(records) - len(scored),
        # macro F1 is only meaningful when both classes are represented; a slice
        # with one true class scores 0 for the absent class and reads as failure.
        "n_true_classes": len(set(y_true)),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "aligned_precision": report[ALIGNED]["precision"],
        "aligned_recall": report[ALIGNED]["recall"],
        "aligned_f1": report[ALIGNED]["f1-score"],
        "not_aligned_precision": report[NOT_ALIGNED]["precision"],
        "not_aligned_recall": report[NOT_ALIGNED]["recall"],
        "not_aligned_f1": report[NOT_ALIGNED]["f1-score"],
        "support_aligned": int(report[ALIGNED]["support"]),
        "support_not_aligned": int(report[NOT_ALIGNED]["support"]),
        "confusion": {
            "true_aligned_pred_aligned": tp_aligned,
            "true_aligned_pred_not_aligned": fn_aligned,
            "true_not_aligned_pred_aligned": fp_aligned,   # the reward-hacking cell
            "true_not_aligned_pred_not_aligned": tn_aligned,
        },
        "false_aligned_rate": (
            fp_aligned / (fp_aligned + tn_aligned) if (fp_aligned + tn_aligned) else None
        ),
    }


def failure_breakdown(records: list[dict]) -> dict:
    """Counts of API/parse failures by kind -- run health, not model quality."""
    kinds: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("prediction") not in LABELS:
            kinds[record.get("error_kind") or "Unknown"] += 1
    return dict(sorted(kinds.items(), key=lambda kv: -kv[1]))


def by_negative_type(records: list[dict]) -> dict:
    """Metrics for the random-negative and hard-negative evaluation sets.

    Each set is positives + that negative kind, so both are 50:50 balanced and
    directly comparable. The gap between them is the headline evidence for
    whether a verifier is genuinely discriminative.
    """
    positives = [r for r in records if r.get("negative_type") == "positive"]
    out: dict[str, dict] = {}
    for negative_type in ("random", "hard"):
        negatives = [r for r in records if r.get("negative_type") == negative_type]
        if not negatives:
            continue
        out[negative_type] = core_metrics(positives + negatives)
    if "random" in out and "hard" in out:
        random_f1, hard_f1 = out["random"]["macro_f1"], out["hard"]["macro_f1"]
        if random_f1 is not None and hard_f1 is not None:
            out["hard_minus_random_f1"] = round(hard_f1 - random_f1, 4)
    return out


def by_target_label(records: list[dict], min_support: int = 1) -> dict:
    """Per-error-type metrics.

    A verifier with strong overall F1 that fails on OffByOneError or LogicError
    is not usable as a reward signal for those categories, and the aggregate
    hides it.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["target_label"]].append(record)

    out: dict[str, dict] = {}
    for label, rows in sorted(groups.items()):
        if len(rows) < min_support:
            continue
        metrics = core_metrics(rows)
        metrics["n_positive"] = sum(1 for r in rows if r["label"] == ALIGNED)
        metrics["n_negative"] = sum(1 for r in rows if r["label"] == NOT_ALIGNED)
        out[label] = metrics
    return out


def by_gold_label(records: list[dict]) -> dict:
    """Per-error-type metrics keyed by the submission's *true* error.

    Complements :func:`by_target_label`: it answers "which real error types does
    the verifier recognise?" rather than "which prompted categories does it
    handle?".
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for gold in record.get("gold_labels") or []:
            groups[gold].append(record)
    return {label: core_metrics(rows) for label, rows in sorted(groups.items())}


def summarise(records: list[dict]) -> dict:
    """Everything the report needs for one (model, dataset, condition)."""
    return {
        "overall": core_metrics(records),
        "by_negative_type": by_negative_type(records),
        "by_target_label": by_target_label(records),
        "by_gold_label": by_gold_label(records),
        "failures": failure_breakdown(records),
    }


def text_report(records: list[dict]) -> str:
    """sklearn's classification_report for console output."""
    scored = _scored(records)
    if not scored:
        return "(no scored predictions)"
    return classification_report(
        [r["label"] for r in scored],
        [r["prediction"] for r in scored],
        labels=LABELS, digits=4, zero_division=0,
    )
