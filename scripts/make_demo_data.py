"""Create the synthetic Station 3 dataset (about 50 MB of WAV files).

    python scripts/make_demo_data.py --out data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf  # noqa: E402

from msad.synth import SR, generate_dataset, make_degrading_machine  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=150, help="healthy training clips")
    ap.add_argument("--test", type=int, default=40, help="test clips per class")
    args = ap.parse_args()
    out = generate_dataset(args.out, n_train=args.train, n_test=args.test)
    demo = out / "demo"
    demo.mkdir(exist_ok=True)
    for fault in ("bearing", "imbalance", "rattle"):
        sf.write(demo / f"degrading_{fault}.wav", make_degrading_machine(fault=fault), SR)
    print(f"Done: {out.resolve()}  (live demo recordings in {demo})")
