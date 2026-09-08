"""Verifier prompt construction.

Mirrors the Learning-to-Be-Wrong verifier format -- (problem, solution, error
category, error description) -> ``aligned`` / ``not_aligned`` -- adapted to code.

Input conditions (Experiment 3, context ablation):

    PyMETA
      P1  problem statement + student code
      P2  problem statement + reference solution + student code
    COJ2022
      C1  student code only              (COJ ships no natural-language statement)
      C2  student code + repaired reference program
      C3  problem statement + student code

C3 makes COJ2022 structurally parallel to PyMETA's P1, which is what separates
"the model cannot judge semantic errors" from "the model had no idea what the
program was supposed to do". COJ2022 publishes no statements, so they were
reconstructed by hand from each problem's judge test cases -- see
data/coj2022_statements.json for authorship and the exclusion rule. Problems
whose rule the test cases did not determine carry no statement and are dropped
from C3 rather than given a hedged one.

Nothing derived from the gold label is ever rendered: no traceback, no
interpreter output, no buggy-line pointer, no diff. See data.leakage for the
column-level guard that enforces this upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a programming error evaluator. Given a programming problem, a "
    "student's code, and an error category, determine whether the error in the "
    "student's code aligns with the given error category.\n"
    "Answer 'aligned' only if the error actually present in the student's code "
    "belongs to the given error category. Answer 'not_aligned' if the student's "
    "code contains a different kind of error.\n"
    "Respond with exactly one of: aligned, not_aligned."
)

QUESTION = (
    "Does the error in the student's code align with the given error category?"
)

#: condition -> which optional context blocks are rendered
CONDITIONS: dict[str, dict[str, bool]] = {
    "P1": {"problem": True, "reference": False},
    "P2": {"problem": True, "reference": True},
    "C1": {"problem": False, "reference": False},
    "C2": {"problem": False, "reference": True},
    "C3": {"problem": True, "reference": False},
}

DATASET_CONDITIONS: dict[str, tuple[str, ...]] = {
    "pymeta": ("P1", "P2"),
    "coj2022": ("C1", "C2", "C3"),
}


@dataclass(frozen=True)
class VerifierQuery:
    """One rendered verifier question."""

    system: str
    user: str


def _fence(code: str, language: str) -> str:
    body = (code or "").rstrip("\n")
    return f"```{language}\n{body}\n```"


def render(sample: dict, condition: str) -> VerifierQuery:
    """Render one pair row into a system/user message pair.

    ``sample`` is a row produced by data.pairs.build_pairs and carries:
    ``student_code``, ``language``, ``target_label``, ``target_description``,
    and optionally ``problem`` / ``reference_code``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {sorted(CONDITIONS)}")
    flags = CONDITIONS[condition]
    language = sample.get("language") or "text"

    parts: list[str] = []

    if flags["problem"]:
        problem = (sample.get("problem") or "").strip()
        if not problem:
            raise ValueError(
                f"condition {condition!r} needs a problem statement but the sample "
                f"has none (pair_id={sample.get('pair_id')!r})"
            )
        parts.append(f"## Problem\n{problem}")

    if flags["reference"]:
        reference = (sample.get("reference_code") or "").strip()
        if not reference:
            raise ValueError(
                f"condition {condition!r} needs reference code but the sample has "
                f"none (pair_id={sample.get('pair_id')!r})"
            )
        parts.append(f"## Reference Solution\n{_fence(reference, language)}")

    parts.append(f"## Student's Code\n{_fence(sample['student_code'], language)}")
    parts.append(f"## Error Category\n{sample['target_label']}")
    parts.append(f"## Error Description\n{sample['target_description']}")
    parts.append(QUESTION)

    return VerifierQuery(system=SYSTEM_PROMPT, user="\n\n".join(parts))
