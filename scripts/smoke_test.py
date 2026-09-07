"""Verify API access and answer parsing with a handful of calls.

    python scripts/smoke_test.py                       # all closed models
    python scripts/smoke_test.py --models gemini-2.5-flash
    python scripts/smoke_test.py --models all --n 4

Run this before run_eval.py. It spends a few cents and catches the failures that
otherwise surface a thousand calls into a paid run: a missing or wrong key, a
model id that does not exist for your account, a rejected parameter, or an empty
response caused by the model spending its whole output budget on thinking.

Two fixtures with known answers are used, so a model that returns a valid label
for the wrong reason is still visible in the output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifier_pilot.clients import registry  # noqa: E402
from verifier_pilot.console import setup  # noqa: E402
from verifier_pilot.prompts import render  # noqa: E402
from verifier_pilot.taxonomy import pymeta as tax  # noqa: E402

FIXTURES = [
    {
        "name": "true IndexError, asked about IndexError",
        "expect": "aligned",
        "pair": {
            "pair_id": "smoke-1", "language": "python",
            "problem": "Read a list of integers and print the third element.",
            "student_code": "nums = [int(x) for x in input().split()]\nprint(nums[2])\n",
            "target_label": "IndexError",
            "target_description": tax.describe("IndexError"),
        },
    },
    {
        "name": "true NameError, asked about KeyError",
        "expect": "not_aligned",
        "pair": {
            "pair_id": "smoke-2", "language": "python",
            "problem": "Print the sum of two integers read from input.",
            "student_code": "a = int(input())\nb = int(input())\nprint(a + total)\n",
            "target_label": "KeyError",
            "target_description": tax.describe("KeyError"),
        },
    },
]


def main() -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["closed"])
    parser.add_argument("--n", type=int, default=len(FIXTURES), help="fixtures per model")
    parser.add_argument("--condition", default="P1", choices=["P1", "C1"])
    args = parser.parse_args()

    names = registry.resolve(args.models)
    fixtures = FIXTURES[: args.n]
    failures = 0

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        try:
            client = registry.build(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  [SETUP FAILED] {type(exc).__name__}: {exc}")
            failures += 1
            continue

        try:
            for fixture in fixtures:
                pair = dict(fixture["pair"])
                if args.condition == "C1":
                    pair.pop("problem", None)
                query = render(pair, args.condition)
                prediction = client.predict(query.system, query.user)

                if not prediction.ok:
                    print(f"  [FAIL] {fixture['name']}")
                    print(f"         {prediction.error_kind}: {prediction.error_detail}")
                    failures += 1
                    continue

                mark = "ok " if prediction.label == fixture["expect"] else "MISS"
                print(f"  [{mark}] {fixture['name']}")
                print(f"         got={prediction.label} expected={fixture['expect']} "
                      f"latency={prediction.latency_s}s attempts={prediction.attempts}")
                if prediction.usage:
                    print(f"         usage={prediction.usage}")
        finally:
            client.close()

    print("\n" + "=" * 60)
    if failures:
        print(f"{failures} call(s) failed -- fix these before running run_eval.py")
        print("Hints: missing key -> .env at the repo root (see .env.example)")
        print("       EmptyResponse on Gemini -> thinking budget; see clients/gemini.py")
        return 1
    print("All models reachable and answering. Next: python scripts/run_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
