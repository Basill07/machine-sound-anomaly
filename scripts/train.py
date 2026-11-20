"""Train the detector on healthy machine sounds.

    python scripts/train.py --data data --out models/station3                  # sound profile (default)
    python scripts/train.py --method autoencoder --out models/station3_ae      # autoencoder baseline

`--data` must contain train/normal/*.wav (the demo data, or a MIMII/DCASE machine folder).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from msad.model import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/station3")
    ap.add_argument("--method", choices=["profile", "autoencoder"], default="profile")
    ap.add_argument("--epochs", type=int, default=40, help="autoencoder only")
    args = ap.parse_args()

    files = sorted((Path(args.data) / "train" / "normal").glob("*.wav"))
    if not files:
        sys.exit(f"No WAV files in {Path(args.data) / 'train' / 'normal'}. Run: python scripts/make_demo_data.py")
    t0 = time.perf_counter()
    kwargs = {"epochs": args.epochs} if args.method == "autoencoder" else {}
    det = train(files, method=args.method, **kwargs)
    det.save(args.out)
    print(f"Saved to {args.out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
