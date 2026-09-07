"""Download the raw datasets into data/raw/.

    python scripts/prepare_data.py                    # both datasets
    python scripts/prepare_data.py --dataset pymeta   # one
    python scripts/prepare_data.py --force            # re-download

PyMETA  -> https://huggingface.co/datasets/CircleCat/pymeta  (train/dev/test.csv)
COJ2022 -> https://github.com/DaSESmartEdu/ErrorCLR          (COJ2022/*)

Roughly 60 MB in total. Nothing here is committed -- data/ is git-ignored.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifier_pilot.console import setup  # noqa: E402
from verifier_pilot.data import coj2022, pymeta  # noqa: E402

SOURCES: dict[str, list[tuple[str, str]]] = {
    "pymeta": [(f"{split}.csv", f"{pymeta.HF_BASE}/{split}.csv") for split in pymeta.SPLITS],
    "coj2022": [(name, f"{coj2022.GITHUB_BASE}/{name}") for name in coj2022.FILES],
}


def download(url: str, destination: Path, force: bool = False) -> bool:
    """Fetch ``url`` to ``destination``. Returns True if it downloaded."""
    if destination.is_file() and not force:
        size_mb = destination.stat().st_size / 1e6
        print(f"  [skip] {destination.name} ({size_mb:.1f} MB, already present)")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"  [get ] {destination.name} <- {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "verifier-pilot/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)
    print(f"         {destination.stat().st_size / 1e6:.1f} MB")
    return True


def main() -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["pymeta", "coj2022", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    names = ["pymeta", "coj2022"] if args.dataset == "all" else [args.dataset]
    directories = {"pymeta": pymeta.raw_dir(args.root), "coj2022": coj2022.raw_dir(args.root)}

    for name in names:
        print(f"\n=== {name} ===")
        for filename, url in SOURCES[name]:
            try:
                download(url, directories[name] / filename, force=args.force)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {filename}: {type(exc).__name__}: {exc}")
                return 1

    print("\n=== verifying ===")
    if "pymeta" in names:
        records = pymeta.load_submissions(args.root)
        print(f"  pymeta : {len(records):>6} erroneous submissions, "
              f"{len({r['problem_id'] for r in records})} problems")
    if "coj2022" in names:
        records = coj2022.load_submissions(args.root)
        print(f"  coj2022: {len(records):>6} buggy programs, "
              f"{len({r['problem_id'] for r in records})} problems")
    print("\nNext: python scripts/build_pairs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
