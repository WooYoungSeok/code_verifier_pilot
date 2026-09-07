"""Run verifiers over the pair sets.

    # Experiment 1 -- zero-shot, closed models, primary condition
    python scripts/run_eval.py --models closed --datasets all

    # context ablation (Experiment 3)
    python scripts/run_eval.py --models closed --datasets pymeta --conditions P1 P2

    # open-weight zero-shot (needs a GPU)
    python scripts/run_eval.py --models open --datasets all

    # an SFT'd adapter
    python scripts/run_eval.py --models qwen2.5-coder-7b \
        --adapter runs/sft/pymeta_qwen/adapter --name qwen2.5-coder-7b-sft-pymeta \
        --datasets pymeta

Results append to outputs/raw/<model>__<dataset>__<condition>.jsonl and are
resumable: re-running skips completed pairs, so an interrupted run costs
nothing to continue. Use --refresh-failed to retry only the failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifier_pilot.clients import registry  # noqa: E402
from verifier_pilot.console import setup  # noqa: E402
from verifier_pilot.evaluation import metrics, runner  # noqa: E402
from verifier_pilot.prompts import DATASET_CONDITIONS  # noqa: E402


def load_pairs(root: Path, dataset: str, split: str, suffix: str = "") -> list[dict]:
    name = f"{dataset}_{split}{'_' + suffix if suffix else ''}.jsonl"
    path = root / "data" / "pairs" / name
    if not path.is_file():
        raise SystemExit(
            f"missing {path}. Run: python scripts/build_pairs.py --split {split}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["closed"])
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=["pymeta", "coj2022", "all"])
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="default: the primary condition per dataset (P1 / C1)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pairs-suffix", default="")
    parser.add_argument("--limit", type=int, default=None, help="cap calls per config (debug)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--refresh-failed", action="store_true")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (local models)")
    parser.add_argument("--name", default=None, help="override the result/model name")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--allow-param-fallback", action="store_true",
                        help="permit dropping a sampling parameter the API rejects. "
                             "Off by default: a rejected parameter aborts the run so the "
                             "change is a decision, not a silent side effect. Any drop is "
                             "recorded in the run manifest either way.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    output_dir = args.out or (args.root / "outputs")
    datasets = ["pymeta", "coj2022"] if "all" in args.datasets else args.datasets
    model_names = registry.resolve(args.models)

    plan: list[tuple[str, str]] = []
    for dataset in datasets:
        conditions = args.conditions or [DATASET_CONDITIONS[dataset][0]]
        for condition in conditions:
            if condition not in DATASET_CONDITIONS[dataset]:
                raise SystemExit(
                    f"condition {condition!r} is not defined for {dataset}; "
                    f"valid: {DATASET_CONDITIONS[dataset]}"
                )
            plan.append((dataset, condition))

    print(f"models    : {model_names}")
    print(f"plan      : {plan}")
    print(f"outputs   : {output_dir}")

    summary: dict[str, dict] = {}
    for model_name in model_names:
        client_kwargs: dict = {"strict_params": not args.allow_param_fallback}
        if args.adapter:
            client_kwargs["adapter_path"] = args.adapter
        if args.load_in_4bit:
            client_kwargs["load_in_4bit"] = True
        if args.name:
            client_kwargs["name"] = args.name

        print(f"\n{'=' * 66}\n{model_name}\n{'=' * 66}", flush=True)
        try:
            client = registry.build(model_name, **client_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"  [SETUP FAILED] {type(exc).__name__}: {exc}")
            continue

        try:
            for dataset, condition in plan:
                pairs = load_pairs(args.root, dataset, args.split, args.pairs_suffix)
                if condition in ("P2", "C2"):
                    before = len(pairs)
                    pairs = [p for p in pairs if p.get("reference_code")]
                    if len(pairs) < before:
                        print(f"  {condition}: {before - len(pairs)} pairs lack reference code, skipped")

                records = runner.run(
                    client, pairs, dataset, condition, output_dir,
                    resume=not args.no_resume,
                    refresh_failed=args.refresh_failed,
                    limit=args.limit, sleep_s=args.sleep,
                )
                result = metrics.summarise(records)
                overall = result["overall"]
                key = f"{client.name}|{dataset}|{condition}"
                summary[key] = result

                print(f"\n  --- {key} ---")
                if overall["n_scored"]:
                    print(f"  macro F1          : {overall['macro_f1']:.4f}")
                    print(f"  aligned precision : {overall['aligned_precision']:.4f}")
                    print(f"  aligned recall    : {overall['aligned_recall']:.4f}")
                    print(f"  accuracy          : {overall['accuracy']:.4f}")
                    for negative_type in ("random", "hard"):
                        block = result["by_negative_type"].get(negative_type)
                        if block and block.get("macro_f1") is not None:
                            print(f"  {negative_type:<6} negatives  : macro F1 {block['macro_f1']:.4f}")
                if overall["n_failed"]:
                    print(f"  FAILED CALLS      : {overall['n_failed']}  {result['failures']}")
        finally:
            client.close()

    if summary:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "summary.json"
        existing = {}
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.update(summary)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsummary -> {path}")
        print("Next: python scripts/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
