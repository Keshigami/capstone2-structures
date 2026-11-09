"""Classify an image: `uv run python src/predict.py <image> [--top 3] [--json]`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from structures.inference import Predictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Top-k historical-structure classes for an image")
    ap.add_argument("image", nargs="+", help="image file(s)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args()
    predictor = Predictor()
    for path in args.image:
        result = predictor.predict_path(path, top=args.top)
        if args.json:
            print(json.dumps({"image": path, **result}))
            continue
        print(path)
        for row in result["top"]:
            bar = "#" * int(round(row["probability"] * 30))
            print(f"  {row['label']:<16} {row['probability']*100:6.2f}%  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
