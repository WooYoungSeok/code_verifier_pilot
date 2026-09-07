"""Pair-construction invariants -- the correctness core of the experiment."""

import pytest

from verifier_pilot.data import pairs as P
from verifier_pilot.taxonomy import coj2022 as coj
from verifier_pilot.taxonomy import pymeta as pym


def _record(sid, labels, pid="p1"):
    return {
        "dataset": "coj2022", "submission_id": sid, "problem_id": pid,
        "student_code": "int main(){}", "language": "c", "problem": None,
        "reference_code": "int main(){return 0;}", "gold_labels": list(labels),
        "split": "test",
    }


def _build(records, space=None):
    return P.build_pairs(
        records, space or coj.LABELS, coj.hard_negative_pool, coj.describe
    )


def test_each_submission_yields_one_positive_and_two_negatives():
    built = _build([_record("s1", ["Control::WrongOperator"])])
    kinds = sorted(p["negative_type"] for p in built)
    assert kinds == ["hard", "positive", "random"]
    assert sum(p["label"] == "aligned" for p in built) == 1


def test_positive_target_is_in_gold():
    for pair in _build([_record("s1", ["Control::WrongOperator"])]):
        if pair["label"] == "aligned":
            assert pair["target_label"] in pair["gold_labels"]


def test_negatives_avoid_the_entire_gold_set():
    """The multi-label case: a negative must miss *every* gold label."""
    gold = ["Control::WrongOperator", "Control::OffByOneError", "Function::FunctionMissing"]
    for pair in _build([_record("s1", gold)]):
        if pair["label"] == "not_aligned":
            assert pair["target_label"] not in gold


def test_hard_negative_is_a_sibling_under_the_same_type():
    built = _build([_record("s1", ["Control::WrongOperator"])])
    hard = [p for p in built if p["negative_type"] == "hard"][0]
    assert hard["target_label"].startswith("Control::")
    assert hard["target_label"] != "Control::WrongOperator"


def test_random_negative_draws_from_the_full_space():
    """Across many submissions a random negative should leave the gold's own type."""
    records = [_record(f"s{i}", ["Control::WrongOperator"]) for i in range(60)]
    randoms = [p for p in _build(records) if p["negative_type"] == "random"]
    types = {p["target_label"].split("::")[0] for p in randoms}
    assert len(types) > 1


def test_evaluation_sets_are_balanced_and_share_positives():
    records = [_record(f"s{i}", ["Control::WrongOperator"]) for i in range(20)]
    built = _build(records)
    for negative_type in ("random", "hard"):
        subset = P.evaluation_set(built, negative_type)
        aligned = sum(p["label"] == "aligned" for p in subset)
        assert aligned * 2 == len(subset), f"{negative_type} set is not 50:50"
    shared = {p["pair_id"] for p in P.evaluation_set(built, "random")} & \
             {p["pair_id"] for p in P.evaluation_set(built, "hard")}
    assert len(shared) == 20   # the positives, which is why bootstrap clusters


def test_pair_ids_are_unique_and_deterministic():
    records = [_record(f"s{i}", ["Control::WrongOperator"]) for i in range(30)]
    first, second = _build(records), _build(list(reversed(records)))
    assert len({p["pair_id"] for p in first}) == len(first)
    assert [p["pair_id"] for p in first] == [p["pair_id"] for p in second]


def test_sampling_is_stable_under_upstream_filtering():
    """A submission's negative must not depend on which others were kept."""
    records = [_record(f"s{i}", ["Control::WrongOperator"]) for i in range(20)]
    full = {p["pair_id"]: p["target_label"] for p in _build(records)}
    subset = {p["pair_id"]: p["target_label"] for p in _build(records[5:10])}
    for pair_id, target in subset.items():
        assert full[pair_id] == target


def test_cap_per_label_limits_and_is_deterministic():
    records = [_record(f"s{i}", ["Control::WrongOperator"]) for i in range(50)]
    records += [_record(f"t{i}", ["Function::FunctionMissing"]) for i in range(5)]
    capped = P.cap_per_label(records, 10)
    assert sum(1 for r in capped if r["gold_labels"][0] == "Control::WrongOperator") == 10
    assert sum(1 for r in capped if r["gold_labels"][0] == "Function::FunctionMissing") == 5
    assert [r["submission_id"] for r in capped] == \
           [r["submission_id"] for r in P.cap_per_label(records, 10)]


def test_cap_none_keeps_everything():
    records = [_record(f"s{i}", ["Control::WrongOperator"]) for i in range(7)]
    assert len(P.cap_per_label(records, None)) == 7


def test_pymeta_pairs_use_the_13_class_space():
    record = _record("s1", ["IndexError"])
    record["dataset"] = "pymeta"
    built = P.build_pairs(
        [record], pym.ERROR_CLASSES, pym.hard_negative_pool, pym.describe
    )
    for pair in built:
        assert pair["target_label"] in pym.ERROR_CLASSES
        assert pair["target_description"]


def test_unknown_negative_type_rejected():
    with pytest.raises(ValueError):
        P.build_pairs([_record("s1", ["Control::WrongOperator"])], coj.LABELS,
                      coj.hard_negative_pool, coj.describe, negative_types=("bogus",))
