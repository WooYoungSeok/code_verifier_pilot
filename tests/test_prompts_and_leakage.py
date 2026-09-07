"""Prompt rendering and the leakage guard."""

import pytest

from verifier_pilot.data.leakage import (
    COJ_FORBIDDEN, PYMETA_FORBIDDEN, assert_no_leakage, check_record_fields,
)
from verifier_pilot.prompts import CONDITIONS, render

SAMPLE = {
    "pair_id": "x", "language": "python", "problem": "Sum two integers.",
    "reference_code": "print(int(input()) + int(input()))",
    "student_code": "a = int(input())\nprint(a + b)",
    "target_label": "NameError", "target_description": "A name is not bound.",
}


def test_p1_omits_reference_and_p2_includes_it():
    assert "Reference Solution" not in render(SAMPLE, "P1").user
    assert "Reference Solution" in render(SAMPLE, "P2").user


def test_c1_omits_problem_and_reference():
    user = render(SAMPLE, "C1").user
    assert "## Problem" not in user
    assert "Reference Solution" not in user
    assert "Student's Code" in user


def test_every_condition_renders_the_target_and_description():
    for condition in CONDITIONS:
        user = render(SAMPLE, condition).user
        assert SAMPLE["target_label"] in user
        assert SAMPLE["target_description"] in user


def test_missing_required_context_raises_rather_than_silently_degrading():
    without_problem = {**SAMPLE, "problem": None}
    with pytest.raises(ValueError):
        render(without_problem, "P1")
    without_reference = {**SAMPLE, "reference_code": ""}
    with pytest.raises(ValueError):
        render(without_reference, "P2")


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        render(SAMPLE, "Z9")


def test_gold_columns_are_on_the_deny_list():
    for column in ("all_errortype2", "R_traceback", "testOutcome", "error_category"):
        assert column in PYMETA_FORBIDDEN
    for column in ("Type", "SubType", "Buggy_Line", "Line_ID", "Repaired_Line"):
        assert column in COJ_FORBIDDEN


def test_traceback_in_a_prompt_is_caught():
    with pytest.raises(AssertionError):
        assert_no_leakage("Traceback (most recent call last):", "pymeta")


def test_target_label_may_appear_but_a_second_gold_may_not():
    prompt = "## Error Category\nControl::WrongOperator\n\nint main(){}"
    assert_no_leakage(prompt, "coj2022", ["Control::WrongOperator"], "Control::WrongOperator")
    leaky = prompt + "\n// TODO fix the OffByOneError\n"
    with pytest.raises(AssertionError):
        assert_no_leakage(leaky, "coj2022", ["Control::OffByOneError"], "Control::WrongOperator")


def test_record_carrying_a_gold_column_is_rejected():
    with pytest.raises(AssertionError):
        check_record_fields({"studentAnswer": "x", "all_errortype2": "NameError"}, "pymeta")
    check_record_fields({"studentAnswer": "x", "question": "q"}, "pymeta")
