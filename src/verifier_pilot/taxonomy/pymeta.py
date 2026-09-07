"""PyMETA error taxonomy.

Gold labels come from ``all_errortype2`` -- NOT from ``error_category``, which is
a binary error/no_error column in the released HuggingFace CSVs and does not
match the "single-error label" wording in the dataset card.

Raw ``all_errortype2`` values observed across train+dev+test (19 total, verified
against CircleCat/pymeta @ main):

    No error 20886+2321, LogicError, SyntaxError, NameError, TypeError,
    IndentationError, UnboundLocalError, KeyError, IndexError, RecursionError,
    EOFError, ValueError, TabError, AttributeError, RuntimeError,
    SyntaxWarning, MemoryError, ZeroDivisionError, ModuleNotFoundError

Those 19 collapse to a 14-class taxonomy by folding the rare tail into
``Other errors``. The verifier task drops ``No error`` (we only ask whether an
*erroneous* program matches a target error type), leaving 13 usable classes.
"""

from __future__ import annotations

NO_ERROR = "No error"
OTHER = "Other errors"

#: raw ``all_errortype2`` value -> canonical 14-class label.
RAW_TO_CANONICAL: dict[str, str] = {
    "No error": NO_ERROR,
    "LogicError": "LogicError",
    "SyntaxError": "SyntaxError",
    "NameError": "NameError",
    "TypeError": "TypeError",
    "IndentationError": "IndentationError",
    "UnboundLocalError": "UnboundLocalError",
    "KeyError": "KeyError",
    "IndexError": "IndexError",
    "RecursionError": "RecursionError",
    "EOFError": "EOFError",
    "ValueError": "ValueError",
    "TabError": "TabError",
    # --- rare tail -> Other errors ---
    "AttributeError": OTHER,
    "RuntimeError": OTHER,
    "SyntaxWarning": OTHER,
    "MemoryError": OTHER,
    "ZeroDivisionError": OTHER,
    "ModuleNotFoundError": OTHER,
}

#: the 13 classes the verifier actually sees (14-class taxonomy minus "No error").
ERROR_CLASSES: list[str] = [
    "LogicError",
    "SyntaxError",
    "NameError",
    "TypeError",
    "IndentationError",
    "UnboundLocalError",
    "KeyError",
    "IndexError",
    "RecursionError",
    "EOFError",
    "ValueError",
    "TabError",
    OTHER,
]

CANONICAL_CLASSES: list[str] = [NO_ERROR] + ERROR_CLASSES  # the full 14

#: Shown to the verifier alongside the target label. Deliberately describes the
#: *category*, never the specific defect, so the description cannot leak the answer.
ERROR_DESCRIPTIONS: dict[str, str] = {
    "LogicError": (
        "The program runs to completion without raising a Python exception, but "
        "produces incorrect output for at least one test case. The defect is in the "
        "algorithm or reasoning, not in the syntax or the runtime behaviour."
    ),
    "SyntaxError": (
        "The source cannot be parsed by the Python interpreter: the code violates "
        "Python's grammar (e.g. an unclosed bracket, a missing colon, an invalid "
        "assignment target). The program never starts executing."
    ),
    "NameError": (
        "Execution reaches an identifier that is not bound in any accessible scope -- "
        "typically a misspelled or never-assigned global/builtin name."
    ),
    "TypeError": (
        "An operation or function is applied to an object of an inappropriate type, "
        "or is called with the wrong number/kind of arguments."
    ),
    "IndentationError": (
        "The source is not parseable because of indentation: an unexpected indent, an "
        "expected indented block, or an unindent that matches no outer level."
    ),
    "UnboundLocalError": (
        "A local variable is read before it has been assigned within its own function "
        "scope -- usually because the name is also assigned somewhere later in that "
        "function, which makes it local rather than global."
    ),
    "KeyError": (
        "A mapping is looked up with a key that is not present in it."
    ),
    "IndexError": (
        "A sequence is indexed with an integer outside the valid range for it."
    ),
    "RecursionError": (
        "The maximum recursion depth is exceeded -- typically a recursive function "
        "with a missing or unreachable base case."
    ),
    "EOFError": (
        "An input function such as input() hits end-of-file without reading any data, "
        "usually because the program tries to read more input than the test supplies."
    ),
    "ValueError": (
        "A function receives an argument of the right type but an inappropriate value, "
        "such as int() on a string that does not denote a number, or unpacking a "
        "sequence of the wrong length."
    ),
    "TabError": (
        "Indentation mixes tabs and spaces inconsistently, so the interpreter cannot "
        "determine the intended block structure."
    ),
    OTHER: (
        "A runtime error outside the common categories above -- for example "
        "AttributeError, ZeroDivisionError, RuntimeError, MemoryError, "
        "ModuleNotFoundError, or a SyntaxWarning raised as an error."
    ),
}

#: Confusion groups for hard-negative sampling. A hard negative is drawn from the
#: target's own group, so the verifier must separate genuinely similar failure
#: modes rather than merely noticing "some runtime error exists".
#: Every class appears in at least one group, and every group has >= 2 members.
HARD_NEGATIVE_GROUPS: list[list[str]] = [
    # unparseable source: which *kind* of parse failure?
    ["SyntaxError", "IndentationError", "TabError"],
    # name binding: global-miss vs local-before-assignment
    ["NameError", "UnboundLocalError"],
    # container lookup: sequence index vs mapping key
    ["IndexError", "KeyError"],
    # bad argument: wrong type vs right type / wrong value
    ["TypeError", "ValueError"],
    # control-flow / IO exhaustion
    ["RecursionError", "EOFError", OTHER],
    # silently-wrong output vs a raised exception on the same bad data
    ["LogicError", "TypeError", "ValueError", "IndexError", "KeyError"],
]


def canonicalize(raw_label: str) -> str:
    """Map a raw ``all_errortype2`` value onto the 14-class taxonomy."""
    key = (raw_label or "").strip()
    if key not in RAW_TO_CANONICAL:
        raise KeyError(
            f"unknown all_errortype2 value {raw_label!r}; add it to RAW_TO_CANONICAL "
            f"(known: {sorted(RAW_TO_CANONICAL)})"
        )
    return RAW_TO_CANONICAL[key]


def describe(label: str) -> str:
    return ERROR_DESCRIPTIONS[label]


def hard_negative_pool(label: str) -> list[str]:
    """Plausible-but-wrong labels for ``label``, drawn from its confusion groups."""
    pool: list[str] = []
    for group in HARD_NEGATIVE_GROUPS:
        if label in group:
            pool.extend(x for x in group if x != label)
    # stable order, de-duplicated
    return sorted(dict.fromkeys(pool))
