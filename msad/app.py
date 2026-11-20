"""Web dashboard: live health from the microphone, replay of demo recordings, file upload.

    uvicorn msad.app:app
    open http://localhost:8000
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .model import SR, load_audio, load_detector

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MSAD_MODEL", ROOT / "models" / "station3"))
DEMO_DIR = Path(os.environ.get("MSAD_DEMO_DIR", ROOT / "data" / "demo"))
STATIC = Path(__file__).parent / "static"
MAX_SECONDS = 30
SILENCE_RMS = 1e-3

app = FastAPI(title="Machine Sound Anomaly Detection")


@lru_cache(maxsize=1)
def detector():
    if not (MODEL_DIR / "config.json").exists():
        raise HTTPException(503, f"No trained model at {MODEL_DIR}. Run: python scripts/train.py")
    return load_detector(MODEL_DIR)


def _result(audio: np.ndarray, with_spectrogram: bool = False) -> dict:
    if len(audio) < SR:
        raise HTTPException(400, "Need at least 1 second of audio")
    det = detector()
    if float(np.sqrt(np.mean(audio**2))) < SILENCE_RMS or np.mean(np.abs(audio) < 1e-5) > 0.2:
        # silence (or a dropout) is not machine sound: don't raise a false alarm
        return {"silent": True, "threshold": round(det.threshold, 3), "method": det.method}
    s = det.score(audio)
    out = {
        "score": round(s, 3),
        "threshold": round(det.threshold, 3),
        "health": round(det.health(s), 1),
        "anomaly": det.is_anomaly(s),
        "hints": det.explain(audio) if det.is_anomaly(s) and hasattr(det, "explain") else [],
        "method": det.method,
    }
    if with_spectrogram:
        out["spectrogram"] = _spectrogram_png(audio)
    return out


def _spectrogram_png(audio: np.ndarray) -> str:
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio[: 10 * SR], sr=SR, n_mels=64), ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    librosa.display.specshow(mel, sr=SR, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("Mel spectrogram")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/info")
def info():
    det = detector()
    demos = sorted(p.name for p in DEMO_DIR.glob("*.wav")) if DEMO_DIR.exists() else []
    return {"method": det.method, "threshold": det.threshold, "model": MODEL_DIR.name, "demos": demos}


@app.get("/demo/{name}")
def demo_file(name: str):
    path = (DEMO_DIR / name).resolve()
    if path.parent != DEMO_DIR.resolve() or not path.exists():
        raise HTTPException(404, "Unknown demo file")
    return FileResponse(path, media_type="audio/wav")


@app.post("/score-pcm")
async def score_pcm(request: Request):
    """Raw 16 kHz mono float32 samples (little endian) from the browser."""
    body = await request.body()
    if len(body) % 4 or len(body) > MAX_SECONDS * SR * 4:
        raise HTTPException(400, "Expected up to 30 s of float32 samples")
    return _result(np.frombuffer(body, dtype="<f4").astype(np.float32))


@app.post("/score-file")
async def score_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            audio = load_audio(tmp.name)
        except Exception as err:
            raise HTTPException(400, f"Could not read audio: {err}") from err
    return _result(audio[: MAX_SECONDS * SR], with_spectrogram=True)
