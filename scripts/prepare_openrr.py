"""Check/rearrange an already-downloaded OpenRR-5k directory.

No download is performed. Use this after extracting OpenRR archives manually on
the server.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED = {
    "train_blended": Path("train") / "blended",
    "train_transmission": Path("train") / "transmission",
    "val_blended": Path("val") / "blended",
    "val_transmission": Path("val") / "transmission",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OpenRR directory layout.")
    parser.add_argument("--root", default="data/openrr5k", help="OpenRR output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    missing = []
    for label, rel in EXPECTED.items():
        path = root / rel
        if not path.is_dir():
            missing.append(path)
        else:
            count = len([p for p in path.iterdir() if p.is_file()])
            print(f"{label}: {path} ({count} files)")
    if missing:
        print("Missing expected directories:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)
    print("OpenRR layout looks ready for RAP-ERRNet.")


if __name__ == "__main__":
    main()

