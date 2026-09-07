"""COJ2022 semantic-error taxonomy (ErrorCLR, Han et al., SIGIR 2023).

COJ2022 ships 8,511 annotated errors over 5,912 buggy **C/C++** programs from an
introductory course. Labels are hierarchical: a coarse ``Type`` (4 values) and a
``SubType``. The same sub-type deliberately recurs under several types (e.g.
``WrongOperator`` under both Control and Expression), so a bare sub-type is
ambiguous -- we therefore label with the composite ``Type::SubType``.

Verified against ErrorCLR@main ``COJ2022/error_info.csv``: 26 distinct
``Type::SubType`` pairs, of which 22 carry a real sub-type and 4 are
``<Type>::Undefined`` -- 3,707 of 8,511 rows (43.6%) are ``Undefined``, i.e.
annotated only at the coarse Type level.

``Undefined`` is excluded from sub-type-level pair construction by default
(``granularity="subtype"``); use ``granularity="type"`` to run the coarse
4-class task over the whole dataset instead.

Note the real column name is ``SubType`` (not ``Sub_Type``) and the code files
live in ``source_codes.zip`` as ``source_codes/<ID>.c`` or ``.cpp``.

Descriptions below are transcribed from the official ``error_types_info.md``.
"""

from __future__ import annotations

UNDEFINED = "Undefined"
SEP = "::"

TYPES: list[str] = ["Control", "Function", "Declaration", "Expression"]

TYPE_DESCRIPTIONS: dict[str, str] = {
    "Control": "An error pertaining to a control statement (loop or conditional).",
    "Function": (
        "An error pertaining to a function call or declaration. Input/output "
        "statements (printf / scanf / getchar / cin / ...) count as functions."
    ),
    "Declaration": "An error pertaining to a variable declaration.",
    "Expression": "An error inside an ordinary expression.",
}

#: The 22 real ``Type::SubType`` pairs, in the order given by error_types_info.md.
TYPE_SUBTYPES: dict[str, list[str]] = {
    "Control": [
        "ControlMisuse",
        "ConditionMissing",
        "WrongIndex",
        "WrongOperator",
        "LiteralMisuse",
        "VariableMisuse",
        "OffByOneError",
    ],
    "Function": [
        "FunctionMisuse",
        "FunctionMissing",
        "ReturnTypeMisuse",
        "ParameterTypeMisuse",
        "FormatStringMisuse",
        "WrongIndex",
    ],
    "Declaration": [
        "DataTypeMisuse",
        "WrongArraySize",
        "InitializationMissing",
        "LiteralMisuse",
    ],
    "Expression": [
        "WrongIndex",
        "WrongOperator",
        "LiteralMisuse",
        "VariableMisuse",
        "OffByOneError",
    ],
}

LABELS: list[str] = [f"{t}{SEP}{s}" for t, subs in TYPE_SUBTYPES.items() for s in subs]

#: Sub-type descriptions, transcribed from the official error_types_info.md.
#: Keyed by ``Type::SubType`` because the wording differs slightly per type.
LABEL_DESCRIPTIONS: dict[str, str] = {
    # ---- Control ----
    "Control::ControlMisuse": (
        "Misuse of a control action -- specifically confusing 'break' with 'continue', "
        "or a loop (while/for) with a conditional (if/switch)."
    ),
    "Control::ConditionMissing": (
        "A control or loop condition is missing, or an entire conditional branch is "
        "absent -- for example 'if(a > 0)' where 'if(a > 0 && b != 1)' was needed, or "
        "a missing 'else' branch."
    ),
    "Control::WrongIndex": (
        "A wrong array index inside a control statement -- whether the index is built "
        "from variables, literals or operators."
    ),
    "Control::WrongOperator": (
        "A wrong operator (arithmetic, relational or logical) inside a control "
        "statement -- for example 'if(i<n)' where 'if(i<=n)' was needed."
    ),
    "Control::LiteralMisuse": (
        "A wrong literal (number, string, char or constant) inside a control "
        "statement -- for example a loop starting at 0 where it should start at 1."
    ),
    "Control::VariableMisuse": (
        "The wrong variable is used inside a control statement -- for example "
        "'j < n-1' where 'j < i-1' was needed."
    ),
    "Control::OffByOneError": (
        "A control statement is off by exactly +1/-1 on a variable -- for example "
        "'while(sum != n)' where 'while(sum != n+1)' was needed."
    ),
    # ---- Function ----
    "Function::FunctionMisuse": (
        "The wrong function is called -- for example 'abs(x)' where 'floor(x)' was "
        "needed, or a call that should not be there at all."
    ),
    "Function::FunctionMissing": (
        "A required function call is absent -- for example a missing 'getchar();' or "
        "a missing 'round(...)' wrapper."
    ),
    "Function::ReturnTypeMisuse": (
        "The return type is wrong in a function declaration or call -- for example "
        "'char func(int a)' where 'char* func(int a)' was needed."
    ),
    "Function::ParameterTypeMisuse": (
        "A parameter type is wrong in a declaration or call -- for example passing 'a' "
        "to scanf where the address '&a' was needed."
    ),
    "Function::FormatStringMisuse": (
        "A format string in an input/output statement is wrong -- for example a "
        "printf conversion of '%f' where '%.2f' was needed, or a missing space in a "
        "cout literal."
    ),
    "Function::WrongIndex": (
        "A wrong array index inside a function call argument -- whether the index is "
        "built from variables, literals or operators."
    ),
    # ---- Declaration ----
    "Declaration::DataTypeMisuse": (
        "A variable is declared with the wrong data type -- for example 'int sum = 0;' "
        "where 'long sum = 0;' was needed."
    ),
    "Declaration::WrongArraySize": (
        "A declared array has the wrong size -- for example 'int a[n+1];' where "
        "'int a[n+10];' was needed."
    ),
    "Declaration::InitializationMissing": (
        "A declared variable is missing its initial value -- for example 'int i, sum;' "
        "where 'int i=0, sum=0;' was needed."
    ),
    "Declaration::LiteralMisuse": (
        "A wrong literal in a declaration -- for example 'int max_n = 8;' where "
        "'int max_n = 10;' was needed, or wrong values in an array initialiser."
    ),
    # ---- Expression ----
    "Expression::WrongIndex": (
        "A wrong array index inside an ordinary expression -- for example "
        "'a[i][j] * b[j]' where 'a[i][k] * b[k]' was needed."
    ),
    "Expression::WrongOperator": (
        "A wrong operator (arithmetic, relational or logical) inside an ordinary "
        "expression -- for example 'a += 1;' where 'a -= 1;' was needed."
    ),
    "Expression::LiteralMisuse": (
        "A wrong literal inside an ordinary expression, precision problems excluded "
        "-- for example 'ans * 10' where 'ans * 16' was needed."
    ),
    "Expression::VariableMisuse": (
        "The wrong variable is used inside an ordinary expression -- for example "
        "'pre = n;' where 'pre = i;' was needed."
    ),
    "Expression::OffByOneError": (
        "An ordinary expression is off by exactly +1/-1 -- for example 'm = n/2;' "
        "where 'm = n/2+1;' was needed."
    ),
}


def make_label(type_: str, subtype: str) -> str:
    return f"{type_}{SEP}{subtype}"


def split_label(label: str) -> tuple[str, str]:
    type_, _, subtype = label.partition(SEP)
    return type_, subtype


def is_undefined(label: str) -> bool:
    return split_label(label)[1] == UNDEFINED


def describe(label: str) -> str:
    """Description for a ``Type::SubType`` label, or a coarse ``Type`` label."""
    if label in LABEL_DESCRIPTIONS:
        return LABEL_DESCRIPTIONS[label]
    if label in TYPE_DESCRIPTIONS:
        return TYPE_DESCRIPTIONS[label]
    type_, subtype = split_label(label)
    if subtype == UNDEFINED and type_ in TYPE_DESCRIPTIONS:
        return TYPE_DESCRIPTIONS[type_]
    raise KeyError(f"no description for COJ2022 label {label!r}")


def hard_negative_pool(label: str) -> list[str]:
    """Sibling sub-types under the *same* coarse Type.

    This is the discriminative test: the verifier already knows the error is, say,
    a Control error, and must still decide *which* control error it is.
    """
    type_, subtype = split_label(label)
    return [
        make_label(type_, s)
        for s in TYPE_SUBTYPES.get(type_, [])
        if s != subtype
    ]
