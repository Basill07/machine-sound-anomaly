"""Command line tools.

    python -m msad.cli record my_fan --seconds 180        # record healthy sound from your mic
    python -m msad.cli score data/test/bearing/bearing_000.wav
    python -m msad.cli monitor                            # live health score from the microphone
    python -m msad.cli monitor --file some_recording.wav  # replay a recording as if it were live
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .model import SR, load_audio, load_detector

WINDOW_S = 5.0  # each score looks at the last 5 seconds (same length as the training clips)
STEP_S = 1.0  # new score every second


def _bar(health: float, width: int = 30) -> str:
    filled = int(round(health / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _report(det, window: np.ndarray, t: float) -> None:
    s = det.score(window)
    h = det.health(s)
    status = "ALARM" if det.is_anomaly(s) else ("watch" if h < 70 else "ok   ")
    print(f"{t:6.0f}s  {_bar(h)} {h:5.1f}  {status}  (score {s:.2f})")
    if det.is_anomaly(s) and hasattr(det, "explain"):
        for hint in det.explain(window, top=2):
            print(f"          ↳ {hint}")


def cmd_record(args) -> None:
    """Record healthy machine sound and save it as 5-second training clips."""
    import sounddevice as sd

    out = Path(args.data) / "train" / "normal"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Recording {args.seconds:.0f}s of HEALTHY sound. Keep the machine running normally, avoid talking...")
    audio = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    clip = int(5 * SR)
    n = 0
    for i in range(0, len(audio) - clip + 1, clip):
        sf.write(out / f"{args.name}_{int(time.time())}_{n:03d}.wav", audio[i : i + clip], SR)
        n += 1
    print(f"Saved {n} clips to {out}. Now train: python scripts/train.py --data {args.data} --out models/{args.name}")


def cmd_score(args) -> None:
    det = load_detector(args.model)
    for f in args.files:
        audio = load_audio(f)
        s = det.score(audio)
        verdict = "ANOMALY" if det.is_anomaly(s) else "healthy"
        print(f"{Path(f).name}: score {s:.2f}, health {det.health(s):.0f}/100 -> {verdict}")
        if det.is_anomaly(s) and hasattr(det, "explain"):
            for hint in det.explain(audio):
                print(f"   ↳ {hint}")


def cmd_monitor(args) -> None:
    det = load_detector(args.model)
    win, step = int(WINDOW_S * SR), int(STEP_S * SR)
    print(f"Model: {det.method}, alarm above score {det.threshold:.2f}. Ctrl+C to stop.\n")

    if args.file:  # replay a recording in real time
        audio = load_audio(args.file)
        for end in range(win, len(audio) + 1, step):
            _report(det, audio[end - win : end], end / SR)
            time.sleep(0 if args.fast else STEP_S)
        return

    import sounddevice as sd

    buf = np.zeros(0, dtype=np.float32)
    t0 = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=step) as stream:
        try:
            while True:
                block, _ = stream.read(step)
                buf = np.r_[buf, block[:, 0]][-win:]
                if len(buf) == win:
                    _report(det, buf, time.time() - t0)
        except KeyboardInterrupt:
            print("\nStopped.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="msad", description="Machine sound anomaly detection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("record", help="Record healthy sound from the microphone as training clips")
    p.add_argument("name", help="machine name, e.g. my_fan")
    p.add_argument("--seconds", type=float, default=180, help="3 minutes = 36 clips; more is better")
    p.add_argument("--data", default=None, help="default: data_<name>")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("score", help="Score WAV files")
    p.add_argument("files", nargs="+")
    p.add_argument("--model", default="models/station3")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("monitor", help="Live health score from the microphone (or replay a file)")
    p.add_argument("--model", default="models/station3")
    p.add_argument("--file", help="replay this WAV instead of using the microphone")
    p.add_argument("--fast", action="store_true", help="with --file: don't wait between steps")
    p.set_defaults(func=cmd_monitor)

    args = ap.parse_args(argv)
    if getattr(args, "data", "x") is None:
        args.data = f"data_{args.name}"
    try:
        args.func(args)
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}. Train a model first: python scripts/train.py")


if __name__ == "__main__":
    main()
