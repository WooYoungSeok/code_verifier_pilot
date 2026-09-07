"""PyMETA loader (HuggingFace ``CircleCat/pymeta``).

Produces normalised *submission* records. Gold labels come from
``all_errortype2`` mapped onto the 14-class taxonomy; ``No error`` rows are
dropped because the verifier question ("does this erroneous program show the
target error type?") is only defined on erroneous submissions.

Split modes
-----------
``official``
    Use the released ``train.csv`` / ``dev.csv`` / ``test.csv`` as-is.

``problem_disjoint`` (default)
    Re-split by ``questionId`` with a seeded assignment. **The released splits
    are not problem-disjoint**: all 145 test questionIds also occur in train,
    and 217 (questionId, studentAnswer) pairs -- 8.8% of the test set's unique
    erroneous submissions -- are byte-identical to a training submission. Under
    the official splits an SFT'd verifier is scored on problems, and sometimes
    on exact programs, it was trained on, which would inflate RQ3. Use
    ``official`` only to reproduce numbers comparable to the released splits.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from ..taxonomy import pymeta as tax
from .leakage import PYMETA_FORBIDDEN

csv.field_size_limit(10**9)

DATASET = "pymeta"
LANGUAGE = "python"
SPLITS = ("train", "dev", "test")
HF_BASE = "https://huggingface.co/datasets/CircleCat/pymeta/resolve/main"

#: only these source columns are ever read into a record
_METADATA_COLUMNS = ("questionId", "userId", "attemptId")


def raw_dir(root: Path) -> Path:
    return root / "data" / "raw" / "pymeta"


def _read_csv(path: Path) -> list[dict]:
    text = path.read_bytes().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _submission_id(row: dict, split: str, index: int) -> str:
    raw = f"{row.get('questionId','')}|{row.get('userId','')}|{row.get('attemptId','')}|{index}"
    return f"pymeta-{split}-{hashlib.md5(raw.encode('utf-8', 'replace')).hexdigest()[:16]}"


def _code_fingerprint(row: dict) -> str:
    """Identity of a (problem, program) pair, for cross-split dedup."""
    raw = f"{row.get('questionId','')}\x00{row.get('studentAnswer','') or ''}"
    return hashlib.md5(raw.encode("utf-8", "replace")).hexdigest()


def load_raw(root: Path) -> dict[str, list[dict]]:
    """Read the three released CSVs. Raises with a fix-it message if absent."""
    directory = raw_dir(root)
    missing = [s for s in SPLITS if not (directory / f"{s}.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing PyMETA splits {missing} under {directory}. "
            f"Run: python scripts/prepare_data.py --dataset pymeta"
        )
    return {s: _read_csv(directory / f"{s}.csv") for s in SPLITS}


def _to_record(row: dict, split: str, index: int) -> dict | None:
    """Normalise one CSV row, or None if it is unusable / not an error."""
    raw_label = (row.get("all_errortype2") or "").strip()
    if not raw_label:
        return None
    label = tax.canonicalize(raw_label)
    if label == tax.NO_ERROR:
        return None

    student_code = (row.get("studentAnswer") or "").strip()
    if not student_code:
        return None

    return {
        "dataset": DATASET,
        "submission_id": _submission_id(row, split, index),
        "problem_id": (row.get("questionId") or "").strip(),
        "user_id": (row.get("userId") or "").strip(),
        "problem": (row.get("question") or "").strip() or None,
        "reference_code": (row.get("exceptedAnswer") or "").strip() or None,
        "student_code": student_code,
        "language": LANGUAGE,
        "gold_labels": [label],
        "raw_label": raw_label,
        "orig_split": split,
        "fingerprint": _code_fingerprint(row),
    }


def load_submissions(
    root: Path,
    split_mode: str = "problem_disjoint",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    dedup: bool = True,
) -> list[dict]:
    """Load every erroneous PyMETA submission, tagged with a ``split``."""
    raw = load_raw(root)

    records: list[dict] = []
    for split in SPLITS:
        for index, row in enumerate(raw[split]):
            # structural guard: forbidden columns exist in `row` but are never copied
            record = _to_record(row, split, index)
            if record is not None:
                records.append(record)

    if dedup:
        records = _dedup_by_fingerprint(records)

    if split_mode == "official":
        for record in records:
            record["split"] = record["orig_split"]
    elif split_mode == "problem_disjoint":
        assign = assign_problem_splits(
            sorted({r["problem_id"] for r in records}), seed=seed, ratios=ratios
        )
        for record in records:
            record["split"] = assign[record["problem_id"]]
    else:
        raise ValueError(
            f"unknown split_mode {split_mode!r}; expected 'problem_disjoint' or 'official'"
        )

    for record in records:
        assert not (PYMETA_FORBIDDEN & set(record)), record.keys()
    return records


def _dedup_by_fingerprint(records: list[dict]) -> list[dict]:
    """Keep one record per (problem, program); prefer train, then dev, then test.

    Byte-identical resubmissions occur both within and across the released
    splits; keeping them would let the same program be scored twice and would put
    test programs into an SFT training set.
    """
    rank = {"train": 0, "dev": 1, "test": 2}
    best: dict[str, dict] = {}
    for record in records:
        key = record["fingerprint"]
        current = best.get(key)
        if current is None or rank[record["orig_split"]] < rank[current["orig_split"]]:
            best[key] = record
    return [best[k] for k in sorted(best)]


def assign_problem_splits(
    problem_ids: list[str],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, str]:
    """Deterministically assign whole problems to train/dev/test.

    Hash-based rather than shuffle-based so the assignment for a given problem is
    stable no matter which other problems are present.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    train_cut, dev_cut = ratios[0], ratios[0] + ratios[1]
    assignment: dict[str, str] = {}
    for problem_id in problem_ids:
        digest = hashlib.md5(f"{seed}:{problem_id}".encode("utf-8")).hexdigest()
        position = int(digest[:8], 16) / 0xFFFFFFFF
        if position < train_cut:
            assignment[problem_id] = "train"
        elif position < dev_cut:
            assignment[problem_id] = "dev"
        else:
            assignment[problem_id] = "test"
    return assignment
