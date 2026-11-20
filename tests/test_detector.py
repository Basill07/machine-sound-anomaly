"""Fast end-to-end tests on a small synthetic dataset.  Run: pytest -q"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from msad.model import AEConfig, ProfileDetector, ae_features, load_audio, load_detector, train
from msad.synth import FAULTS, SR, generate_dataset, make_clip, make_degrading_machine


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return generate_dataset(tmp_path_factory.mktemp("data"), n_train=60, n_test=20, seconds=5.0, seed=3)


@pytest.fixture(scope="module")
def profile(dataset):
    return train(sorted((dataset / "train" / "normal").glob("*.wav")), method="profile", log=lambda *_: None)


def _scores(det, folder):
    return np.array([det.score(load_audio(f)) for f in sorted(folder.glob("*.wav"))])


def test_synthetic_clips_are_valid():
    for kind in ("normal", *FAULTS):
        clip = make_clip(kind, seconds=2, seed=1)
        assert clip.dtype == np.float32 and len(clip) == 2 * SR
        assert np.max(np.abs(clip)) <= 1.0 and np.std(clip) > 0.01


def test_ae_feature_shape():
    cfg = AEConfig()
    x = ae_features(make_clip(seconds=2, seed=1), cfg)
    assert x.shape[1] == cfg.dim and x.shape[0] > 50


def test_profile_separates_faults(dataset, profile):
    normal = _scores(profile, dataset / "test" / "normal")
    for fault in FAULTS:
        bad = _scores(profile, dataset / "test" / fault)
        auc = roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(bad))], np.r_[normal, bad])
        assert auc > 0.7, f"{fault}: AUC {auc:.2f}"


def test_few_false_alarms(dataset, profile):
    normal = _scores(profile, dataset / "test" / "normal")
    assert np.mean(normal > profile.threshold) <= 0.2


def test_save_load_roundtrip(tmp_path, profile):
    profile.save(tmp_path)
    loaded = load_detector(tmp_path)
    assert isinstance(loaded, ProfileDetector)
    clip = make_clip("bearing", seconds=4, seed=5, severity=0.5)
    assert loaded.score(clip) == pytest.approx(profile.score(clip))


def test_explain_names_frequency_bands(profile):
    hints = profile.explain(make_clip("rattle", seconds=4, seed=9, severity=0.6))
    assert len(hints) == 3 and all("Hz" in h for h in hints)


def test_degrading_machine_gets_worse(profile):
    audio = make_degrading_machine(seconds=24, fault="imbalance", start=0.3, max_severity=0.8)
    first, last = audio[: 4 * SR], audio[-4 * SR :]
    assert profile.score(last) > profile.score(first)
    assert profile.is_anomaly(profile.score(last))
