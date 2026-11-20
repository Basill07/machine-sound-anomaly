"""Synthetic machine sounds for the demo machine "Station 3" (a screwdriving cell).

Real factory recordings are rarely public, so this module creates realistic-ish sounds:
a motor hum with harmonics, a gearbox tone, and background noise, with natural
variation between clips (speed, loudness, noise level).

Fault types (each with random severity, so some are hard to hear):
  bearing    - periodic metallic clicks from a damaged ball bearing
  imbalance  - stronger 1x rotation tone that wobbles in loudness
  rattle     - short random bursts from a loose part

For real data, use the MIMII or DCASE datasets instead (see README).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
FAULTS = ("bearing", "imbalance", "rattle")


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise with more low than high frequencies, like a factory hall."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spectrum /= np.sqrt(np.maximum(freqs, 20.0))
    pink = np.fft.irfft(spectrum, n)
    return pink / (np.std(pink) + 1e-9)


def _healthy(n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    t = np.arange(n) / SR
    f0 = rng.uniform(47.0, 53.0)  # motor rotation (Hz), varies with load
    drift = 1 + 0.004 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t)  # tiny speed drift
    phase = 2 * np.pi * f0 * np.cumsum(drift) / SR

    sig = np.zeros(n)
    for k in range(1, 30):  # motor harmonics, getting quieter
        sig += (1.0 / k**1.2) * rng.uniform(0.7, 1.3) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    gear = rng.uniform(780, 820)  # gearbox mesh tone
    sig += 0.25 * np.sin(2 * np.pi * gear * np.cumsum(drift) / SR)
    sig += 0.1 * np.sin(2 * np.pi * 2 * gear * np.cumsum(drift) / SR)

    sig /= np.std(sig) + 1e-9
    sig += rng.uniform(0.3, 0.6) * _pink_noise(n, rng)  # hall noise, different in every clip
    return sig, f0


def _add_bearing(sig, f0, severity, rng):
    n = len(sig)
    rate = 3.2 * f0  # outer-race defect frequency
    period = int(SR / rate)
    t = np.arange(int(0.004 * SR)) / SR
    res = rng.uniform(2800, 4200)  # metal resonance
    click = np.sin(2 * np.pi * res * t) * np.exp(-t * 1500)
    out = sig.copy()
    for start in range(rng.integers(0, period), n - len(click), period):
        out[start : start + len(click)] += severity * rng.uniform(0.6, 1.4) * click
    return out


def _add_imbalance(sig, f0, severity, rng):
    n = len(sig)
    t = np.arange(n) / SR
    wobble = 1 + 0.6 * np.sin(2 * np.pi * rng.uniform(0.8, 2.5) * t)
    return sig + severity * 1.2 * wobble * np.sin(2 * np.pi * f0 * t)


def _add_rattle(sig, f0, severity, rng):
    out = sig.copy()
    n = len(sig)
    for _ in range(rng.integers(6, 16)):
        length = int(rng.uniform(0.03, 0.12) * SR)
        start = rng.integers(0, n - length)
        burst = rng.standard_normal(length) * np.hanning(length)
        # rattles are mostly mid/high frequency
        burst = np.diff(burst, prepend=0)
        out[start : start + length] += severity * 2.5 * burst
    return out


_FAULT_FN = {"bearing": _add_bearing, "imbalance": _add_imbalance, "rattle": _add_rattle}


def make_clip(kind: str = "normal", seconds: float = 5.0, seed: int | None = None, severity: float | None = None) -> np.ndarray:
    """One clip of machine sound. kind = 'normal' or one of FAULTS."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    sig, f0 = _healthy(n, rng)
    if kind != "normal":
        if severity is None:
            severity = rng.uniform(0.15, 0.6)  # from barely audible to obvious
        sig = _FAULT_FN[kind](sig, f0, severity, rng)
    sig *= rng.uniform(0.6, 1.0) * 0.25 / (np.max(np.abs(sig)) + 1e-9) * 3  # random loudness
    return np.clip(sig, -1, 1).astype(np.float32)


def make_degrading_machine(seconds: float = 40.0, fault: str = "bearing", start: float = 0.4, max_severity: float = 0.7, seed: int = 42) -> np.ndarray:
    """One continuous recording: healthy at first, then a fault slowly appears and gets worse.

    `start` = when the fault begins (fraction of the recording).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    sig, f0 = _healthy(n, rng)
    fault_part = _FAULT_FN[fault](sig, f0, 1.0, rng) - sig  # the fault sound alone, at severity 1
    ramp = np.clip((np.arange(n) / n - start) / (1 - start), 0, 1) * max_severity
    out = sig + ramp * fault_part
    out *= 0.75 / (np.max(np.abs(out)) + 1e-9)
    return np.clip(out, -1, 1).astype(np.float32)


def generate_dataset(out_dir: str | Path, n_train: int = 150, n_test: int = 40, seconds: float = 5.0, seed: int = 0) -> Path:
    """Write train/normal and test/{normal,bearing,imbalance,rattle} WAV files."""
    out = Path(out_dir)
    counter = seed * 100_000
    jobs = [("train", "normal", n_train)] + [("test", k, n_test) for k in ("normal", *FAULTS)]
    for split, kind, count in jobs:
        folder = out / split / kind
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            counter += 1
            clip = make_clip(kind, seconds, seed=counter)
            sf.write(folder / f"{kind}_{i:03d}.wav", clip, SR)
    return out
