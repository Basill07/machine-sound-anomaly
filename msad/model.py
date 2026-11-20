"""Two anomaly detectors for machine sound. Both learn from HEALTHY sound only.

1. ProfileDetector (default)
   Describes a whole clip by its "sound profile": how loud each frequency band is on
   average, and how much it fluctuates over time. A Mahalanobis distance measures how far
   a new clip is from the healthy profiles. It catches slow changes (a 1 Hz wobble from an
   imbalance) and tonal changes, and it can explain which bands changed.

2. AutoEncoderDetector (baseline, as in the DCASE challenge)
   A small neural network learns to rebuild 160 ms slices of the healthy spectrogram.
   Unusual sound is rebuilt badly. Good at short events (rattles, clicks), blind to
   slow patterns longer than its window.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch import nn

SR = 16_000


def load_audio(path: str | Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio


def _split(files: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    files = list(files)
    np.random.default_rng(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_fraction))
    return files[n_val:], files[:n_val]


def _threshold(val_scores: list[float], percentile: float) -> float:
    return float(np.percentile(val_scores, percentile) * 1.05)


class BaseDetector:
    method: str
    threshold: float

    def score(self, audio: np.ndarray) -> float:
        raise NotImplementedError

    def is_anomaly(self, score: float) -> bool:
        return score > self.threshold

    def health(self, score: float) -> float:
        """0-100 for dashboards. About 100 for typical healthy sound, 50 at the alarm threshold, 0 far beyond it."""
        return float(np.clip(100 * (1.5 - score / self.threshold), 0, 100))


# ----------------------------------------------------------------------------
# 1) Sound profile + Mahalanobis distance
# ----------------------------------------------------------------------------

@dataclass
class ProfileConfig:
    # 16 wide bands: enough detail to see which part of the spectrum changed, and few enough
    # features (32) that ~25 healthy clips already give a stable model (64 bands needed 100+).
    n_mels: int = 16
    n_fft: int = 1024
    hop_length: int = 256


def _band_edges(cfg: ProfileConfig) -> np.ndarray:
    return librosa.mel_frequencies(n_mels=cfg.n_mels + 2, fmax=SR / 2)


def profile_features(audio: np.ndarray, cfg: ProfileConfig) -> np.ndarray:
    """[mean level per band, fluctuation per band] -> vector of length 2 * n_mels."""
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels)
    db = librosa.power_to_db(mel)
    return np.r_[db.mean(axis=1), db.std(axis=1)].astype(np.float64)


@dataclass
class ProfileDetector(BaseDetector):
    cfg: ProfileConfig
    mean: np.ndarray
    std: np.ndarray
    location: np.ndarray
    precision: np.ndarray
    threshold: float = 0.0
    method: str = field(default="profile", init=False)

    def _z(self, audio: np.ndarray) -> np.ndarray:
        return (profile_features(audio, self.cfg) - self.mean) / self.std

    def score(self, audio: np.ndarray) -> float:
        d = self._z(audio) - self.location
        return float(np.sqrt(d @ self.precision @ d))

    def explain(self, audio: np.ndarray, top: int = 3) -> list[str]:
        """Plain-language hints: which frequency bands differ most from healthy sound."""
        z = self._z(audio)
        n = self.cfg.n_mels
        edges = _band_edges(self.cfg)
        order = np.argsort(-np.abs(z))[:top]
        hints = []
        for i in order:
            band = i % n
            lo, hi = edges[band], edges[band + 2]
            what = ("louder" if z[i] > 0 else "quieter") if i < n else ("fluctuates more" if z[i] > 0 else "steadier")
            hints.append(f"{_hz(lo)}-{_hz(hi)}: {what} than healthy ({z[i]:+.1f} std)")
        return hints

    @classmethod
    def fit(cls, normal_files, cfg=None, val_fraction=0.15, threshold_percentile=98.0, seed=0, log=print):
        cfg = cfg or ProfileConfig()
        fit_files, val_files = _split(normal_files, val_fraction, seed)
        X = np.array([profile_features(load_audio(f), cfg) for f in fit_files])
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
        cov = LedoitWolf().fit((X - mean) / std)  # shrinkage: stable with few clips
        det = cls(cfg, mean, std, cov.location_, cov.precision_)
        det.threshold = _threshold([det.score(load_audio(f)) for f in val_files], threshold_percentile)
        log(f"Profile model from {len(fit_files)} healthy clips; threshold {det.threshold:.2f} (from {len(val_files)} held-back clips)")
        return det

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "profile.npz", mean=self.mean, std=self.std, location=self.location, precision=self.precision)
        (folder / "config.json").write_text(json.dumps({"method": self.method, "features": asdict(self.cfg), "threshold": self.threshold}, indent=2))

    @classmethod
    def _load(cls, folder: Path, meta: dict):
        p = np.load(folder / "profile.npz")
        return cls(ProfileConfig(**meta["features"]), p["mean"], p["std"], p["location"], p["precision"], meta["threshold"])


def _hz(f: float) -> str:
    return f"{f / 1000:.1f} kHz" if f >= 1000 else f"{f:.0f} Hz"


# ----------------------------------------------------------------------------
# 2) Autoencoder baseline
# ----------------------------------------------------------------------------

@dataclass
class AEConfig:
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    frames: int = 5  # 5 x 32 ms = 160 ms context

    @property
    def dim(self) -> int:
        return self.n_mels * self.frames


def ae_features(audio: np.ndarray, cfg: AEConfig) -> np.ndarray:
    """Log-mel spectrogram with `frames` neighbouring frames stacked: shape (n_vectors, n_mels * frames)."""
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels)
    log_mel = 10 * np.log10(mel + 1e-10)
    T = log_mel.shape[1] - cfg.frames + 1
    if T < 1:
        raise ValueError("Audio clip is too short")
    stacked = np.stack([log_mel[:, i : i + T] for i in range(cfg.frames)], axis=0)
    return stacked.transpose(2, 0, 1).reshape(T, cfg.dim).astype(np.float32)


class AutoEncoder(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, bottleneck: int = 8):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU())

        self.encoder = nn.Sequential(block(dim, hidden), block(hidden, hidden), block(hidden, bottleneck))
        self.decoder = nn.Sequential(block(bottleneck, hidden), block(hidden, hidden), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class AutoEncoderDetector(BaseDetector):
    model: AutoEncoder
    cfg: AEConfig
    mean: np.ndarray
    std: np.ndarray
    threshold: float = 0.0
    method: str = field(default="autoencoder", init=False)

    @torch.no_grad()
    def frame_scores(self, audio: np.ndarray) -> np.ndarray:
        x = torch.from_numpy((ae_features(audio, self.cfg) - self.mean) / self.std)
        self.model.eval()
        return ((self.model(x) - x) ** 2).mean(dim=1).numpy()

    def score(self, audio: np.ndarray) -> float:
        return float(self.frame_scores(audio).mean())

    @classmethod
    def fit(cls, normal_files, cfg=None, epochs=40, batch_size=512, val_fraction=0.15, threshold_percentile=98.0, seed=0, log=print):
        cfg = cfg or AEConfig()
        torch.manual_seed(seed)
        fit_files, val_files = _split(normal_files, val_fraction, seed)
        X = np.concatenate([ae_features(load_audio(f), cfg) for f in fit_files])
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
        X = torch.from_numpy((X - mean) / std)
        log(f"Autoencoder: {len(fit_files)} healthy clips ({len(X):,} vectors)")

        model = AutoEncoder(cfg.dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size=batch_size, shuffle=True)
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = ((model(batch) - batch) ** 2).mean()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                log(f"  epoch {epoch:3d}  loss {total / len(X):.4f}")

        det = cls(model, cfg, mean, std)
        det.threshold = _threshold([det.score(load_audio(f)) for f in val_files], threshold_percentile)
        log(f"  threshold {det.threshold:.4f}")
        return det

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), folder / "model.pt")
        np.savez(folder / "norm.npz", mean=self.mean, std=self.std)
        (folder / "config.json").write_text(json.dumps({"method": self.method, "features": asdict(self.cfg), "threshold": self.threshold}, indent=2))

    @classmethod
    def _load(cls, folder: Path, meta: dict):
        cfg = AEConfig(**meta["features"])
        model = AutoEncoder(cfg.dim)
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
        model.eval()
        n = np.load(folder / "norm.npz")
        return cls(model, cfg, n["mean"], n["std"], meta["threshold"])


# ----------------------------------------------------------------------------

METHODS = {"profile": ProfileDetector, "autoencoder": AutoEncoderDetector}


def train(normal_files, method: str = "profile", **kwargs) -> BaseDetector:
    return METHODS[method].fit(normal_files, **kwargs)


def load_detector(folder: str | Path) -> BaseDetector:
    folder = Path(folder)
    meta = json.loads((folder / "config.json").read_text())
    return METHODS[meta.get("method", "autoencoder")]._load(folder, meta)
