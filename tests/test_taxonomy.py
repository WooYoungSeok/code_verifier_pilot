"""Taxonomy invariants. These encode facts verified against the released data."""

import pytest

from verifier_pilot.taxonomy import coj2022 as coj
from verifier_pilot.taxonomy import pymeta as pym


def test_pymeta_raw_label_count():
    # 19 distinct all_errortype2 values across train+dev+test
    assert len(pym.RAW_TO_CANONICAL) == 19


def test_pymeta_collapses_to_14_classes():
    assert len(pym.CANONICAL_CLASSES) == 14
    assert len(set(pym.RAW_TO_CANONICAL.values())) == 14


def test_pymeta_verifier_space_excludes_no_error():
    assert pym.NO_ERROR not in pym.ERROR_CLASSES
    assert len(pym.ERROR_CLASSES) == 13


def test_pymeta_every_class_described_and_has_hard_negatives():
    for label in pym.ERROR_CLASSES:
        assert pym.describe(label)
        pool = pym.hard_negative_pool(label)
        assert pool, f"{label} has no hard negatives"
        assert label not in pool
        assert set(pool) <= set(pym.ERROR_CLASSES)


def test_pymeta_rare_tail_folds_into_other():
    for raw in ("AttributeError", "RuntimeError", "ZeroDivisionError", "MemoryError"):
        assert pym.canonicalize(raw) == pym.OTHER


def test_pymeta_unknown_label_raises():
    with pytest.raises(KeyError):
        pym.canonicalize("NoSuchError")


def test_coj_has_22_real_pairs():
    assert len(coj.LABELS) == 22
    assert len(set(coj.LABELS)) == 22


def test_coj_every_label_described():
    assert set(coj.LABELS) == set(coj.LABEL_DESCRIPTIONS)


def test_coj_hard_negatives_share_the_coarse_type():
    for label in coj.LABELS:
        type_, _ = coj.split_label(label)
        pool = coj.hard_negative_pool(label)
        assert pool, f"{label} has no siblings"
        assert label not in pool
        assert all(coj.split_label(x)[0] == type_ for x in pool)


def test_coj_subtypes_recur_across_types():
    # the reason labels are composite rather than bare sub-types
    assert "Control::WrongOperator" in coj.LABELS
    assert "Expression::WrongOperator" in coj.LABELS
    assert coj.describe("Control::WrongOperator") != coj.describe("Expression::WrongOperator")


def test_coj_undefined_is_recognised():
    assert coj.is_undefined("Control::Undefined")
    assert not coj.is_undefined("Control::WrongOperator")
    assert coj.describe("Control::Undefined") == coj.TYPE_DESCRIPTIONS["Control"]
