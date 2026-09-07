"""Leakage control.

Both datasets ship columns that *are* the answer. Showing the verifier
``R_errortype=NameError`` and then asking "is the target NameError?" makes the
experiment meaningless, so the deny-lists below are enforced structurally: the
loaders build records containing only allow-listed fields, and
:func:`assert_no_leakage` re-checks every rendered prompt before it is sent.
"""

from __future__ import annotations

#: PyMETA columns that encode the gold label or the interpreter's verdict.
PYMETA_FORBIDDEN: frozenset[str] = frozenset({
    "testOutcome",     # raw CodeRunner output, contains the traceback verbatim
    "status",          # parsed execution status
    "R_errorcount",
    "R_traceback",
    "R_errortype",
    "C_errormessage",
    "C_errortype",
    "all_errortype",
    "all_errortype2",  # the gold column itself
    "error_category",
    "state",
})

#: PyMETA columns that may reach the model (exceptedAnswer only in condition P2).
PYMETA_ALLOWED: frozenset[str] = frozenset({
    "question", "exceptedAnswer", "studentAnswer",
})

#: COJ2022 columns that encode the gold label or localise the defect.
COJ_FORBIDDEN: frozenset[str] = frozenset({
    "Type", "SubType",       # gold label
    "Buggy_Line",            # localises the defect
    "Line_ID",               # localises the defect
    "Repaired_Line",         # the fix, line by line
    "Result",                # judge verdict
})

FORBIDDEN_BY_DATASET: dict[str, frozenset[str]] = {
    "pymeta": PYMETA_FORBIDDEN,
    "coj2022": COJ_FORBIDDEN,
}

#: Substrings that must never appear in a rendered prompt. Traceback markers are
#: the practical tell that interpreter output slipped into the student code field.
_PROMPT_TRIPWIRES: tuple[str, ...] = (
    "Traceback (most recent call last)",
    "R_errortype",
    "C_errortype",
    "all_errortype",
)


def assert_no_leakage(
    prompt: str,
    dataset: str,
    gold_labels: list[str] | None = None,
    target_label: str | None = None,
) -> None:
    """Raise if a rendered prompt carries gold-label evidence.

    ``gold_labels`` is checked only for datasets where a label name would not
    otherwise occur: COJ2022 sub-type names such as ``OffByOneError`` are not
    English prose a C program would contain, so finding one in the code or the
    reference means the annotation leaked. PyMETA labels are ordinary Python
    exception names that legitimately appear in student code (``except KeyError``),
    so they are not tripwires there.

    The *target* label is rendered on purpose under "## Error Category"; only
    occurrences beyond that one are treated as leaks.
    """
    for marker in _PROMPT_TRIPWIRES:
        if marker in prompt:
            raise AssertionError(
                f"leaked {marker!r} into a {dataset} prompt -- interpreter output "
                f"reached a field that is shown to the model"
            )
    if dataset != "coj2022" or not gold_labels:
        return

    target_subtype = (target_label or "").partition("::")[2]
    for gold in gold_labels:
        subtype = gold.partition("::")[2]
        if not subtype or subtype == "Undefined":
            continue
        allowance = 1 if subtype == target_subtype else 0
        if prompt.count(subtype) > allowance:
            raise AssertionError(
                f"leaked gold subtype {subtype!r} into a coj2022 prompt body "
                f"(seen {prompt.count(subtype)}x, allowed {allowance}x)"
            )


def check_record_fields(record: dict, dataset: str) -> None:
    """Raise if a record carries any forbidden column into the prompt layer."""
    forbidden = FORBIDDEN_BY_DATASET.get(dataset, frozenset())
    present = forbidden & set(record)
    if present:
        raise AssertionError(
            f"{dataset} record carries forbidden columns {sorted(present)}; these are "
            f"gold-generation only and must be dropped by the loader"
        )
