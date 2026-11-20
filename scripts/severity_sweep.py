"""How bad must a fault be before it's detected? (demo data only)

    python scripts/severity_sweep.py --model models/station3

Generates new clips at fixed fault severities and plots the detection rate.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from msad.model import load_detector  # noqa: E402
from msad.synth import FAULTS, make_clip  # noqa: E402

COLORS = {"bearing": "#d1495b", "imbalance": "#edae49", "rattle": "#00798c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/station3")
    ap.add_argument("--clips", type=int, default=20, help="clips per point")
    ap.add_argument("--out", default="docs/severity.png")
    args = ap.parse_args()

    det = load_detector(args.model)
    levels = np.round(np.arange(0.05, 0.65, 0.05), 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    for fault in FAULTS:
        rates = []
        for lvl in levels:
            hits = [det.is_anomaly(det.score(make_clip(fault, seed=900_000 + i, severity=lvl))) for i in range(args.clips)]
            rates.append(100 * np.mean(hits))
        print(fault.ljust(10), " ".join(f"{r:3.0f}" for r in rates))
        ax.plot(levels, rates, marker="o", lw=2, color=COLORS[fault], label=fault)
    ax.set_xlabel("Fault severity (0.15 = barely audible, 0.6 = obvious)")
    ax.set_ylabel("Detected (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"Detection rate vs. fault severity ({det.method} model, 0% false alarms)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"severity:  {' '.join(f'{l:3.2f}'[1:] for l in levels)}\nSaved {args.out}")


if __name__ == "__main__":
    main()
