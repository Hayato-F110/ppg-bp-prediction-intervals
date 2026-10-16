"""Feature extraction: on synthetic waveforms, values match how the waveform was built."""

import numpy as np
import pytest

from ppg_bp.features.extract import FEATURE_NAMES, segment_features, template_sqi
from tests.test_fiducials import BEAT_SEC, NO_SHOULDER, synthetic_ppg

FS = 125


def test_returns_every_feature_as_float():
    out = segment_features(synthetic_ppg(FS), FS)
    assert tuple(out) == FEATURE_NAMES
    assert all(isinstance(value, float) for value in out.values())
    assert np.isfinite(list(out.values())).all()


def test_values_match_the_synthetic_waveform():
    out = segment_features(synthetic_ppg(FS), FS)
    assert 10 <= out["n_beats"] <= 12
    assert out["ibi_median"] == pytest.approx(BEAT_SEC, abs=0.01)
    assert out["ibi_cv"] < 0.02  # perfectly periodic
    assert out["sqi_template"] > 0.99
    assert out["b_a"] < 0 < out["e_a"]
    assert 0 < out["t_ab"] < out["t_ac"] < out["t_ad"] < out["t_ae"] < out["t_af"]
    assert 0 < out["crest_time"] < BEAT_SEC / 2
    assert out["sw25"] > out["sw50"] > out["sw75"] > 0
    assert out["dw25"] > out["dw50"] > out["dw75"] > 0
    assert out["area_ratio"] > 1  # diastole is longer
    assert 0 < out["height_e"] < 1
    assert out["frac_e_found"] == out["frac_cd_found"] == 1.0


def test_features_do_not_depend_on_amplitude_or_offset():
    ppg = synthetic_ppg(FS)
    small, large = segment_features(ppg, FS), segment_features(500.0 * ppg + 30.0, FS)
    for name in FEATURE_NAMES:
        assert small[name] == pytest.approx(large[name], rel=1e-6, abs=1e-9), name


def test_missing_c_and_d_stay_missing():
    out = segment_features(synthetic_ppg(FS, NO_SHOULDER), FS)
    assert np.isnan(out["c_a"]) and np.isnan(out["t_ad"]) and np.isnan(out["aging_index"])
    assert out["frac_cd_found"] < 0.2
    assert np.isfinite(out["b_a"]) and np.isfinite(out["e_a"])


def test_noise_lowers_the_template_sqi():
    rng = np.random.default_rng(0)
    clean = synthetic_ppg(FS)
    noisy = clean + rng.normal(scale=0.5, size=len(clean))
    assert segment_features(noisy, FS)["sqi_template"] < segment_features(clean, FS)["sqi_template"] - 0.05


def test_irregular_rhythm_raises_interval_variability():
    t = np.arange(10 * FS) / FS
    warped = np.interp(t + 0.08 * np.sin(2 * np.pi * 0.3 * t), t, synthetic_ppg(FS))
    assert segment_features(warped, FS)["ibi_cv"] > 5 * segment_features(synthetic_ppg(FS), FS)["ibi_cv"]


@pytest.mark.parametrize("signal", [np.zeros(1250), np.linspace(0.0, 1.0, 1250)])
def test_segment_without_beats_gives_zero_quality(signal):
    out = segment_features(signal, FS)
    assert out["n_beats"] == 0
    assert out["sqi_template"] == 0.0
    assert np.isnan(out["b_a"])


def test_template_sqi_needs_three_beats():
    x = synthetic_ppg(FS)
    assert template_sqi(x, np.array([60, 160])) == 0.0


def test_template_sqi_follows_the_published_procedure():
    """Window as wide as the median peak interval, mean template, mean correlation (Orphanidou et al. 2015)."""
    x = synthetic_ppg(FS)
    peaks = np.array([123, 223, 323, 423, 523])  # every 0.8 s; the peak is 0.18 s after the start of the beat
    assert template_sqi(x, peaks) > 0.999
    spoiled = x.copy()
    spoiled[273:373] = -spoiled[273:373]  # the third beat is turned upside down
    windows = np.array([spoiled[p - 50 : p + 51] for p in peaks])
    expected = np.mean([np.corrcoef(w, windows.mean(axis=0))[0, 1] for w in windows])
    assert template_sqi(spoiled, peaks) == pytest.approx(expected)
