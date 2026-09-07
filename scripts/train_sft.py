"""Train a verifier LoRA adapter on one dataset.

    # 1. build train/dev pairs without a per-class cap
    python scripts/build_pairs.py --split train --cap none
    python scripts/build_pairs.py --split dev   --cap none

    # 2. train
    python scripts/train_sft.py --dataset pymeta  --model qwen2.5-coder-7b
    python scripts/train_sft.py --dataset coj2022 --model qwen2.5-coder-7b

    # 3. evaluate the adapter on the same test pairs the API models saw
    python scripts/run_eval.py --models qwen2.5-coder-7b \
        --adapter runs/sft/pymeta_qwen2.5-coder-7b/adapter \
        --name qwen2.5-coder-7b-sft --datasets pymeta

Train one adapter per dataset. PyMETA labels are Python interpreter exceptions
and COJ2022 labels are semantic implementation errors; a single verifier over
the union would have to treat two different levels of abstraction as one label
space. Compare them as two verifiers instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifier_pilot.clients.registry import REGISTRY  # noqa: E402
from verifier_pilot.console import setup  # noqa: E402
from verifier_pilot.prompts import DATASET_CONDITIONS  # noqa: E402
from verifier_pilot.sft import build_dataset  # noqa: E402
from verifier_pilot.sft.train_lora import LoraConfig_, train  # noqa: E402


def load_pairs(root: Path, dataset: str, split: str) -> list[dict]:
    path = root / "data" / "pairs" / f"{dataset}_{split}.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"missing {path}. Run: python scripts/build_pairs.py --split {split} --cap none"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["pymeta", "coj2022"])
    parser.add_argument("--model", default="qwen2.5-coder-7b", choices=[
        k for k, v in REGISTRY.items() if v.provider == "local"
    ])
    parser.add_argument("--condition", default=None, help="default: primary (P1 / C1)")
    parser.add_argument("--negatives", default="both", choices=["random", "hard", "both"],
                        help="'random' reproduces the original protocol exactly")
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true", help="LoRA in bf16 instead of QLoRA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="build and describe the SFT set, then stop")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    condition = args.condition or DATASET_CONDITIONS[args.dataset][0]
    model_id = REGISTRY[args.model].model_id
    out_dir = args.out or (args.root / "runs" / "sft" / f"{args.dataset}_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_examples = build_dataset.build_examples(
        load_pairs(args.root, args.dataset, "train"), condition,
        negatives=args.negatives, balance=not args.no_balance, seed=args.seed,
    )
    dev_pairs_path = args.root / "data" / "pairs" / f"{args.dataset}_dev.jsonl"
    eval_examples = None
    if dev_pairs_path.is_file():
        eval_examples = build_dataset.build_examples(
            load_pairs(args.root, args.dataset, "dev"), condition,
            negatives=args.negatives, balance=not args.no_balance, seed=args.seed,
        )

    print(f"dataset   : {args.dataset} (condition {condition})")
    print(f"model     : {args.model} -> {model_id}")
    print(f"train     : {build_dataset.describe(train_examples)}")
    if eval_examples:
        print(f"dev       : {build_dataset.describe(eval_examples)}")

    train_ids = {e["problem_id"] for e in train_examples}
    if eval_examples:
        overlap = train_ids & {e["problem_id"] for e in eval_examples}
        if overlap:
            print(f"  WARNING: {len(overlap)} problems appear in both train and dev")

    build_dataset.write_jsonl(train_examples, out_dir / "train.jsonl")
    if eval_examples:
        build_dataset.write_jsonl(eval_examples, out_dir / "dev.jsonl")
    print(f"sft data  -> {out_dir}")

    if args.dry_run:
        print("\n--- example ---")
        example = train_examples[0]
        print(example["messages"][1]["content"][:600])
        print(f"...\nanswer: {example['messages'][2]['content']}")
        print("\n(dry run: no training performed)")
        return 0

    config = LoraConfig_(
        model_id=model_id, output_dir=str(out_dir),
        epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len, lora_r=args.lora_r,
        load_in_4bit=not args.no_4bit, seed=args.seed,
    )
    adapter = train(config, train_examples, eval_examples)
    print(f"\nNext: python scripts/run_eval.py --models {args.model} "
          f"--adapter {adapter} --name {args.model}-sft-{args.dataset} "
          f"--datasets {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
