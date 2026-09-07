"""Aggregate results into the pilot tables.

    python scripts/report.py
    python scripts/report.py --bootstrap 1000 --compare
    python scripts/report.py --dataset pymeta --condition P1

Produces, per (dataset, condition):

  * the headline table -- macro F1, aligned precision/recall, random-neg F1,
    hard-neg F1 -- with cluster-bootstrap 95% CIs
  * per-error-type F1, so a verifier that is strong overall but blind to
    OffByOneError or LogicError is visible
  * paired model-vs-model comparisons with bootstrap p-values

Everything is read back from outputs/raw/*.jsonl, so the report can be
regenerated without re-running any model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifier_pilot.console import setup  # noqa: E402
from verifier_pilot.evaluation import bootstrap, metrics, runner  # noqa: E402


def discover(output_dir: Path) -> dict[tuple[str, str], dict[str, list[dict]]]:
    """(dataset, condition) -> model -> records, from the raw cache filenames."""
    found: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(dict)
    raw_dir = output_dir / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(f"no results under {raw_dir}. Run scripts/run_eval.py first.")
    for path in sorted(raw_dir.glob("*.jsonl")):
        stem = path.stem
        if stem.count("__") != 2:
            continue
        model, dataset, condition = stem.split("__")
        records = list(runner.load_cache(path).values())
        if records:
            found[(dataset, condition)][model] = records
    return found


def _cell(value, width: int = 7) -> str:
    """Fixed 4-decimal cell, right-aligned to ``width``."""
    return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.4f}"


def headline_table(by_model: dict[str, list[dict]], resamples: int, seed: int) -> str:
    header = (
        f"{'model':<28} {'macroF1':>9} {'95% CI':>17} {'alignP':>7} {'alignR':>7} "
        f"{'randF1':>7} {'hardF1':>7} {'d(h-r)':>7} {'n':>5} {'fail':>5}"
    )
    lines = [header, "-" * len(header)]
    rows = []
    for model, records in by_model.items():
        summary = metrics.summarise(records)
        overall = summary["overall"]
        ci = bootstrap.bootstrap_ci(records, "macro_f1", resamples=resamples, seed=seed)
        random_block = summary["by_negative_type"].get("random", {})
        hard_block = summary["by_negative_type"].get("hard", {})
        rows.append((overall.get("macro_f1") or -1, model, overall, ci, random_block, hard_block, summary))

    for _, model, overall, ci, random_block, hard_block, summary in sorted(rows, reverse=True):
        ci_text = (
            f"[{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"
            if ci.get("ci_low") is not None else "        n/a      "
        )
        delta = summary["by_negative_type"].get("hard_minus_random_f1")
        lines.append(
            f"{model:<28} {_cell(overall.get('macro_f1'), 9)} {ci_text:>17} "
            f"{_cell(overall.get('aligned_precision'))} "
            f"{_cell(overall.get('aligned_recall'))} "
            f"{_cell(random_block.get('macro_f1'))} "
            f"{_cell(hard_block.get('macro_f1'))} "
            f"{('n/a' if delta is None else f'{delta:+.3f}'):>7} "
            f"{overall.get('n_scored', 0):>5} {overall.get('n_failed', 0):>5}"
        )
    return "\n".join(lines)


def per_label_table(by_model: dict[str, list[dict]]) -> str:
    labels = sorted({
        label
        for records in by_model.values()
        for label in metrics.by_target_label(records)
    })
    models = list(by_model)
    width = max((len(l) for l in labels), default=10) + 1
    header = f"{'error type':<{width}}" + "".join(f"{m[:14]:>15}" for m in models)
    lines = [header, "-" * len(header)]
    tables = {m: metrics.by_target_label(r) for m, r in by_model.items()}
    for label in labels:
        row = f"{label:<{width}}"
        for model in models:
            block = tables[model].get(label) or {}
            value = block.get("macro_f1")
            support = block.get("n_scored", 0)
            row += f"{('n/a' if value is None else f'{value:.3f}') + f'({support})':>15}"
        lines.append(row)
    return "\n".join(lines)


def main() -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples; 0 disables CIs")
    parser.add_argument("--compare", action="store_true", help="paired model-vs-model tests")
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    output_dir = args.out or (args.root / "outputs")
    found = discover(output_dir)
    report: dict[str, dict] = {}
    chunks: list[str] = []

    for (dataset, condition), by_model in sorted(found.items()):
        if args.dataset and dataset != args.dataset:
            continue
        if args.condition and condition != args.condition:
            continue

        title = f"{dataset}  /  condition {condition}"
        block = [f"\n{'=' * 100}", title, "=" * 100, "",
                 headline_table(by_model, args.bootstrap, args.seed),
                 "", "per-error-type macro F1 (n scored)", "-" * 40,
                 per_label_table(by_model)]

        entry: dict = {
            model: metrics.summarise(records) for model, records in by_model.items()
        }
        if args.bootstrap:
            for model, records in by_model.items():
                entry[model]["bootstrap"] = {
                    metric: bootstrap.bootstrap_ci(records, metric, args.bootstrap, seed=args.seed)
                    for metric in ("macro_f1", "aligned_precision", "aligned_recall")
                }

        if args.compare and len(by_model) > 1:
            comparisons = bootstrap.compare_all(
                by_model, metric=args.metric, resamples=args.bootstrap, seed=args.seed
            )
            entry["_comparisons"] = comparisons
            block += ["", f"paired comparisons on {args.metric} "
                          f"(cluster bootstrap by submission, {args.bootstrap} resamples)", "-" * 78]
            for comparison in comparisons:
                if comparison.get("diff") is None:
                    continue
                stars = " *" if comparison.get("significant") else "  "
                block.append(
                    f"  {comparison['model_a']:<24} - {comparison['model_b']:<24} "
                    f"{comparison['diff']:+.4f} "
                    f"[{comparison['ci_low']:+.4f},{comparison['ci_high']:+.4f}] "
                    f"p={comparison['p_value']:.3f}{stars}"
                )
            block.append("  (* = 95% CI excludes zero)")

        report[f"{dataset}|{condition}"] = entry
        chunks.append("\n".join(block))

    text = "\n".join(chunks) if chunks else "(no matching results)"
    print(text)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text(text, encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nreport -> {output_dir / 'report.txt'} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
