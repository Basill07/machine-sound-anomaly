"""Test the detector on healthy and faulty clips it has never heard.

    python scripts/evaluate.py --data data --model models/station3

Prints AUC per fault type and saves plots to docs/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from msad.model import SR, load_audio, load_detector  # noqa: E402

COLORS = {"normal": "#6b7b8c", "bearing": "#d1495b", "imbalance": "#edae49", "rattle": "#00798c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="models/station3")
    ap.add_argument("--plots", default="docs")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    det = load_detector(args.model)
    print(f"Method: {det.method}")
    test_dir = Path(args.data) / "test"
    classes = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    if "normal" not in classes:
        sys.exit(f"Need {test_dir}/normal")

    scores = {c: np.array([det.score(load_audio(f)) for f in sorted((test_dir / c).glob("*.wav"))]) for c in classes}
    faults = [c for c in classes if c != "normal"]
    normal = scores["normal"]

    print(f"Threshold: {det.threshold:.4f}")
    print(f"False alarms on healthy clips: {np.mean(normal > det.threshold) * 100:.0f}% ({len(normal)} clips)\n")
    print(f"{'Fault':<12}{'AUC':>7}{'pAUC':>7}{'Detected':>10}")
    results = {"method": det.method, "threshold": det.threshold, "false_alarm_rate": float(np.mean(normal > det.threshold)), "faults": {}}
    for c in faults:
        y = np.r_[np.zeros(len(normal)), np.ones(len(scores[c]))]
        s = np.r_[normal, scores[c]]
        auc = roc_auc_score(y, s)
        pauc = roc_auc_score(y, s, max_fpr=0.1)  # how good it is when false alarms must stay rare
        detected = np.mean(scores[c] > det.threshold)
        results["faults"][c] = {"auc": round(auc, 3), "pauc": round(pauc, 3), "detected": round(float(detected), 3)}
        print(f"{c:<12}{auc:7.3f}{pauc:7.3f}{detected * 100:9.0f}%")

    all_faults = np.concatenate([scores[c] for c in faults])
    overall = roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(all_faults))], np.r_[normal, all_faults])
    results["overall_auc"] = round(overall, 3)
    print(f"{'all faults':<12}{overall:7.3f}")

    out = Path(args.plots)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{det.method}.json").write_text(json.dumps(results, indent=2))
    if args.no_plots:
        return

    # 1) Score distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    allv = np.concatenate(list(scores.values()))
    bins = np.geomspace(max(allv.min(), 1e-6) * 0.9, allv.max() * 1.1, 45)  # log scale: faults can score 100x higher
    for c in classes:
        ax.hist(scores[c], bins=bins, alpha=0.6, label=c, color=COLORS.get(c))
    ax.axvline(det.threshold, color="black", ls="--", lw=1.2, label="alarm threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Anomaly score (log scale)")
    ax.set_ylabel("Clips")
    ax.set_title("Healthy sounds score low, faults score high")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "score_distribution.png", dpi=150)
    plt.close(fig)

    # 2) ROC curves
    fig, ax = plt.subplots(figsize=(5, 5))
    for c in faults:
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(normal)), np.ones(len(scores[c]))], np.r_[normal, scores[c]])
        ax.plot(fpr, tpr, color=COLORS.get(c), lw=2, label=f"{c} (AUC {results['faults'][c]['auc']:.2f})")
    ax.plot([0, 1], [0, 1], color="#bbb", ls=":")
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Detection rate")
    ax.set_title("ROC per fault type")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)
    plt.close(fig)

    # 3) What the faults look like
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.4), sharey=True)
    for ax, c in zip(axes, ["normal"] + faults):
        f = sorted((test_dir / c).glob("*.wav"))[int(np.argmax(scores[c])) if c != "normal" else 0]
        audio = load_audio(f)[: 2 * SR]
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=64), ref=np.max)
        librosa.display.specshow(mel, sr=SR, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(c)
    fig.suptitle("Mel spectrograms: healthy vs. faulty (2 seconds each)")
    fig.tight_layout()
    fig.savefig(out / "spectrograms.png", dpi=150)
    plt.close(fig)
    print(f"\nPlots saved to {out}/")


if __name__ == "__main__":
    main()
