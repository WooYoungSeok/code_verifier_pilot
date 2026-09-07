"""Build verifier pairs from the raw datasets.

    python scripts/build_pairs.py                       # defaults (see below)
    python scripts/build_pairs.py --cap 30 --split test # pilot-sized eval set
    python scripts/build_pairs.py --split train --cap none --out-suffix sft

Writes data/pairs/<dataset>_<split>[_<suffix>].jsonl plus a stats JSON.

Defaults worth knowing
----------------------
``--split-mode problem_disjoint``
    PyMETA's released splits are NOT problem-disjoint: every one of the 145 test
    questionIds also occurs in train, and 217 (questionId, studentAnswer) pairs
    are byte-identical across train and test. Evaluating an SFT'd verifier under
    the official splits scores it on problems -- sometimes on exact programs --
    it trained on. ``--split-mode official`` reproduces the released splits.

``--cap 30``
    At most 30 source submissions per error type, per the pilot budget. Rare
    classes contribute everything they have, which is why the headline metric is
    macro-averaged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifier_pilot.console import setup  # noqa: E402
from verifier_pilot.data import coj2022, pairs as pairs_mod, pymeta  # noqa: E402
from verifier_pilot.taxonomy import coj2022 as coj_tax, pymeta as pymeta_tax  # noqa: E402


def build_for(dataset: str, args, root: Path) -> tuple[list[dict], dict]:
    if dataset == "pymeta":
        records = pymeta.load_submissions(
            root, split_mode=args.split_mode, seed=args.seed, dedup=not args.no_dedup
        )
        label_space = pymeta_tax.ERROR_CLASSES
        hard_pool, describe = pymeta_tax.hard_negative_pool, pymeta_tax.describe
    else:
        mode = "problem_disjoint" if args.split_mode != "random" else "random"
        records = coj2022.load_submissions(
            root, granularity=args.granularity, split_mode=mode, seed=args.seed
        )
        label_space = coj2022.label_space(args.granularity)
        if args.granularity == "type":
            hard_pool = lambda label: [t for t in coj_tax.TYPES if t != label]  # noqa: E731
        else:
            hard_pool = coj_tax.hard_negative_pool
        describe = coj_tax.describe

    if args.split != "all":
        records = [r for r in records if r["split"] == args.split]
    if args.require_reference:
        records = [r for r in records if r.get("reference_code")]

    records = pairs_mod.cap_per_label(records, args.cap, seed=args.seed)
    built = pairs_mod.build_pairs(
        records, label_space, hard_pool, describe, seed=args.seed
    )
    stats = pairs_mod.summarise(built)
    stats.update({
        "dataset": dataset,
        "split": args.split,
        "split_mode": args.split_mode,
        "cap_per_label": args.cap,
        "seed": args.seed,
        "label_space_size": len(label_space),
        "source_submissions": len(records),
    })
    if dataset == "coj2022":
        stats["granularity"] = args.granularity
        stats["with_reference"] = sum(1 for p in built if p.get("reference_code"))
        stats["reference_verified"] = sum(1 for p in built if p.get("reference_verified"))
    return built, stats


def main() -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["pymeta", "coj2022", "all"], default="all")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--split-mode", choices=["problem_disjoint", "official", "random"],
                        default="problem_disjoint")
    parser.add_argument("--granularity", choices=["subtype", "type"], default="subtype",
                        help="COJ2022 label granularity")
    parser.add_argument("--cap", default="30",
                        help="max source submissions per error type, or 'none'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dedup", action="store_true",
                        help="keep byte-identical PyMETA resubmissions")
    parser.add_argument("--require-reference", action="store_true",
                        help="keep only submissions that have reference code (for P2/C2)")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    args.cap = None if str(args.cap).lower() in ("none", "0", "") else int(args.cap)
    datasets = ["pymeta", "coj2022"] if args.dataset == "all" else [args.dataset]
    out_dir = args.root / "data" / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for dataset in datasets:
        built, stats = build_for(dataset, args, args.root)
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        path = out_dir / f"{dataset}_{args.split}{suffix}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for pair in built:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

        all_stats[dataset] = stats
        print(f"\n=== {dataset} ({args.split}) ===")
        print(f"  source submissions : {stats['source_submissions']}")
        print(f"  pairs              : {stats['total_pairs']}  {stats['by_negative_type']}")
        print(f"  balance            : {stats['by_label']}")
        print(f"  problems           : {stats['problems']}")
        if dataset == "coj2022":
            print(f"  reference verified : {stats['reference_verified']}/{stats['total_pairs']}")
        print(f"  -> {path}")

    stats_path = out_dir / f"stats_{args.split}{'_' + args.out_suffix if args.out_suffix else ''}.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nstats -> {stats_path}")
    print("Next: python scripts/smoke_test.py   (verify API access before spending quota)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
