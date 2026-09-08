"""COJ2022 loader (ErrorCLR, ``DaSESmartEdu/ErrorCLR`` -> ``COJ2022/``).

Joins three sources:

    error_info.csv       ID, User_ID, Problem_ID, Buggy_Line, Type, SubType,
                         Line_ID, Repaired_Line          -- 8,511 annotated errors
    source_codes.zip     source_codes/<ID>.c | .cpp      -- 42,073 programs
    submission_info.csv  ID, Problem_ID, Create_Time, User_ID, Result, Language

Two properties of this dataset drive the design and are easy to get wrong:

**Programs carry multiple errors.** 5,912 buggy programs hold 8,511 errors; 1,615
programs have more than one, and 943 have more than one *distinct* label. A
submission's gold is therefore a label **set**. "aligned" means the target is in
that set; a negative must be drawn from the complement of the whole set, never
just the complement of one label.

**43.6% of rows are ``<Type>::Undefined``** -- annotated at the coarse Type level
only. At ``granularity="subtype"`` those errors are dropped from the label set
(and any submission left with an empty set is dropped). At
``granularity="type"`` labels collapse to the 4 coarse types and everything is
usable.

COJ2022 ships no natural-language problem statement -- only ``Problem_ID`` --
which is why the COJ conditions are C1 (code only) and C2 (code + repaired
reference) rather than PyMETA's problem-bearing P1/P2.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from ..taxonomy import coj2022 as tax
from .leakage import COJ_FORBIDDEN
from .pymeta import assign_problem_splits

csv.field_size_limit(10**9)

DATASET = "coj2022"
GITHUB_BASE = "https://raw.githubusercontent.com/DaSESmartEdu/ErrorCLR/main/COJ2022"
FILES = ("error_info.csv", "submission_info.csv", "source_codes.zip",
         "error_types_info.md", "test_case.csv")

#: Hand-written problem statements for condition C3, keyed by the *hash*
#: Problem_ID used in submission_info.csv and test_case.csv -- NOT the
#: integer-like Problem_ID in error_info.csv. The two are 1:1 (498 <-> 498,
#: verified) but they are different id spaces, so C3 joins through the
#: submission id.
STATEMENTS_FILE = "coj2022_statements.json"

_EXT_LANGUAGE = {".c": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp"}


def raw_dir(root: Path) -> Path:
    return root / "data" / "raw" / "coj2022"


def load_statements(root: Path) -> dict[str, str]:
    """hash Problem_ID -> reconstructed problem statement, empty if absent."""
    path = root / "data" / STATEMENTS_FILE
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["statement"] for k, v in payload.get("statements", {}).items()}


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8", errors="replace"))))


def _load_sources(path: Path) -> dict[str, tuple[str, str]]:
    """``ID -> (source_text, language)`` from source_codes.zip."""
    sources: dict[str, tuple[str, str]] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            stem = name.rsplit("/", 1)[-1]
            key, _, ext = stem.rpartition(".")
            language = _EXT_LANGUAGE.get(f".{ext.lower()}")
            if not key or language is None:
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            sources[key] = (text, language)
    return sources


def _apply_repairs(
    source: str, repairs: list[tuple[int, str, str]]
) -> tuple[str | None, bool]:
    """Reconstruct the repaired program from the per-line annotations.

    Returns ``(repaired_source, fully_verified)``.

    ``Line_ID`` is **0-indexed** into ``source.splitlines()`` -- verified against
    the archive: 99.05% of the 7,350 rows with a non-empty ``Buggy_Line`` match
    exactly at ``lines[Line_ID]`` (1-indexed matches 0.1%). A row whose
    ``Buggy_Line`` is empty is an *insertion* at that position, not a
    replacement -- 1,105 rows, all in range.

    Every replacement is checked against ``Buggy_Line`` before being applied; a
    repair that does not match is skipped and ``fully_verified`` goes False, so a
    mis-indexed annotation degrades to a partial reference rather than to a
    silently corrupted one. Repairs are applied highest-index-first so that
    insertions do not shift the positions of later edits.
    """
    lines = source.splitlines()
    verified = True
    applied = 0

    for line_id, buggy, repaired in sorted(repairs, key=lambda r: -r[0]):
        text = repaired.rstrip("\n")
        if buggy.strip():
            if 0 <= line_id < len(lines) and lines[line_id].strip() == buggy.strip():
                if text.strip():
                    lines[line_id] = text
                else:
                    del lines[line_id]          # repair deletes the line
                applied += 1
            else:
                verified = False
        else:
            if 0 <= line_id <= len(lines) and text.strip():
                lines.insert(line_id, text)
                applied += 1
            else:
                verified = False

    if not applied:
        return None, False
    result = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    if not result.strip() or result == source:
        return None, False
    return result, verified


def load_submissions(
    root: Path,
    granularity: str = "subtype",
    split_mode: str = "problem_disjoint",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> list[dict]:
    """Load every annotated buggy COJ2022 program as a normalised record."""
    if granularity not in ("subtype", "type"):
        raise ValueError(f"granularity must be 'subtype' or 'type', got {granularity!r}")

    directory = raw_dir(root)
    missing = [f for f in FILES if not (directory / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing COJ2022 files {missing} under {directory}. "
            f"Run: python scripts/prepare_data.py --dataset coj2022"
        )

    errors = _read_csv(directory / "error_info.csv")
    sources = _load_sources(directory / "source_codes.zip")
    submissions = {r["ID"]: r for r in _read_csv(directory / "submission_info.csv")}
    statements = load_statements(root)

    labels_by_id: dict[str, list[str]] = defaultdict(list)
    repairs_by_id: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    meta_by_id: dict[str, dict] = {}

    for row in errors:
        submission_id = row["ID"]
        type_, subtype = row["Type"].strip(), row["SubType"].strip()
        if granularity == "type":
            label = type_
        else:
            if subtype == tax.UNDEFINED:
                continue          # coarse-only annotation: unusable at subtype level
            label = tax.make_label(type_, subtype)
        labels_by_id[submission_id].append(label)
        meta_by_id.setdefault(submission_id, row)

        line_id = (row.get("Line_ID") or "").strip()
        if line_id.isdigit():
            repairs_by_id[submission_id].append(
                (int(line_id), row.get("Buggy_Line") or "", row.get("Repaired_Line") or "")
            )

    records: list[dict] = []
    for submission_id, labels in labels_by_id.items():
        entry = sources.get(submission_id)
        if entry is None:
            continue              # 27 annotated IDs have no file in the archive
        source, language = entry
        if not source.strip():
            continue
        meta = meta_by_id[submission_id]
        info = submissions.get(submission_id, {})
        reference, reference_verified = _apply_repairs(source, repairs_by_id[submission_id])
        records.append({
            "dataset": DATASET,
            "submission_id": submission_id,
            "problem_id": (meta.get("Problem_ID") or info.get("Problem_ID") or "").strip(),
            "user_id": (meta.get("User_ID") or info.get("User_ID") or "").strip(),
            # COJ2022 publishes no statement; this is the hand-written
            # reconstruction used by condition C3, or None when the problem's
            # rule was not determinable from its test cases.
            "problem": statements.get(
                (info.get("Problem_ID") or "").strip()
            ) or None,
            "problem_id_hash": (info.get("Problem_ID") or "").strip() or None,
            "reference_code": reference,
            "reference_verified": reference_verified,
            "student_code": source,
            "language": language,
            "gold_labels": sorted(dict.fromkeys(labels)),
            "granularity": granularity,
            "fingerprint": submission_id,
        })

    assign = assign_problem_splits(
        sorted({r["problem_id"] for r in records}), seed=seed, ratios=ratios
    )
    if split_mode == "problem_disjoint":
        for record in records:
            record["split"] = assign[record["problem_id"]]
    elif split_mode == "random":
        sub_assign = assign_problem_splits(
            sorted(r["submission_id"] for r in records), seed=seed, ratios=ratios
        )
        for record in records:
            record["split"] = sub_assign[record["submission_id"]]
    else:
        raise ValueError(
            f"unknown split_mode {split_mode!r}; expected 'problem_disjoint' or 'random'"
        )

    for record in records:
        assert not (COJ_FORBIDDEN & set(record)), record.keys()
    records.sort(key=lambda r: r["submission_id"])
    return records


def label_space(granularity: str = "subtype") -> list[str]:
    return list(tax.TYPES) if granularity == "type" else list(tax.LABELS)
