"""Fiducial points: on synthetic waveforms of known shape, the points appear where expected.

The comparison with annotated real data is in scripts/validate_fiducials.py (the data are not in the repository).
"""

import numpy as np
import pytest

from ppg_bp.features.fiducials import POINTS, analyse

BEAT_SEC = 0.8
DISTINCT = (0.12, 0.31)  # (height, position) of a small late-systolic wave; c and d appear as separate extrema
SHOULDER_ONLY = (0.08, 0.30)  # only a shoulder; c and d are not separate
NO_SHOULDER = (0.0, 0.31)


def synthetic_ppg(fs, shoulder=DISTINCT, seconds=10.0):
    """A pulse made of a systolic wave, a small late-systolic wave and a diastolic wave."""
    t = np.arange(int(seconds * fs)) / fs
    ppg = np.zeros_like(t)
    height, position = shoulder
    for k in range(-1, int(seconds / BEAT_SEC) + 2):
        phase = t - k * BEAT_SEC
        ppg += np.exp(-0.5 * ((phase - 0.18) / 0.07) ** 2)
        ppg += height * np.exp(-0.5 * ((phase - position) / 0.03) ** 2)
        ppg += 0.5 * np.exp(-0.5 * ((phase - 0.52) / 0.11) ** 2)
    return ppg


def inner(values):
    """Drop the beats at the ends of the segment (affected by the filter edges)."""
    return values[1:-1]


@pytest.mark.parametrize("fs", [125, 1000])
def test_beats_are_found_at_the_true_rate(fs):
    out = analyse(synthetic_ppg(fs), fs)
    assert 10 <= len(out["t_onset"]) <= 12
    assert np.allclose(inner(out["t_end"] - out["t_onset"]), BEAT_SEC, atol=0.02)
    assert np.all((out["t_onset"] < out["t_peak"]) & (out["t_peak"] < out["t_end"]))
    assert np.allclose(inner(out["t_peak"]) % BEAT_SEC, 0.18, atol=0.03)


@pytest.mark.parametrize("fs", [125, 1000])
def test_points_are_ordered_and_signs_are_physiological(fs):
    out = analyse(synthetic_ppg(fs), fs)
    times = np.column_stack([inner(out[f"t_{p}"]) for p in POINTS])
    assert not np.isnan(times).any()
    assert np.all(np.diff(times, axis=1) > 0)  # a < b < c < d < e < f
    assert np.all(inner(out["cd_distinct"]))
    assert np.all(inner(out["v_a"]) > 0)
    assert np.all(inner(out["v_b"]) < 0)
    assert np.all(inner(out["v_c"]) > inner(out["v_d"]))


def test_c_and_d_coincide_when_there_is_only_a_shoulder():
    """Where c and d do not appear as extrema, the same inflection point is returned and flagged."""
    out = analyse(synthetic_ppg(125, SHOULDER_ONLY), 125)
    assert not np.any(inner(out["cd_distinct"]))
    assert np.allclose(inner(out["t_c"]), inner(out["t_d"]))
    assert np.all((inner(out["t_b"]) < inner(out["t_c"])) & (inner(out["t_c"]) < inner(out["t_e"])))


def test_c_and_d_are_missing_without_a_shoulder():
    out = analyse(synthetic_ppg(125, NO_SHOULDER), 125)
    assert np.isnan(inner(out["t_c"])).all()
    assert np.isnan(inner(out["t_d"])).all()
    assert not np.isnan(inner(out["t_e"])).any()


def test_downsampling_keeps_times_and_ratios():
    results = {fs: analyse(synthetic_ppg(fs), fs) for fs in (125, 1000)}
    assert len(results[125]["t_a"]) == len(results[1000]["t_a"])
    for p in POINTS:
        hi, lo = inner(results[1000][f"t_{p}"]), inner(results[125][f"t_{p}"])
        assert np.abs(hi - lo).max() < 0.004, p  # within half a sample at 125 Hz (8 ms)
    for p in "bcde":
        hi = inner(results[1000][f"v_{p}"] / results[1000]["v_a"])
        lo = inner(results[125][f"v_{p}"] / results[125]["v_a"])
        assert np.abs(hi - lo).max() < 0.06, p


def test_amplitude_scale_does_not_change_times_or_ratios():
    ppg = synthetic_ppg(125)
    small, large = analyse(ppg, 125), analyse(1000.0 * ppg + 50.0, 125)
    assert np.allclose(small["t_a"], large["t_a"], equal_nan=True)
    assert np.allclose(small["v_b"] / small["v_a"], large["v_b"] / large["v_a"], equal_nan=True)


@pytest.mark.parametrize("signal", [np.zeros(1250), np.linspace(0.0, 1.0, 1250)])
def test_no_beats_in_signal_without_pulses(signal):
    out = analyse(signal, 125)
    assert len(out["t_onset"]) == 0
    assert len(out["t_a"]) == 0


def test_noise_does_not_crash():
    out = analyse(np.random.default_rng(0).normal(size=1250), 125)
    assert len(out["t_a"]) == len(out["t_onset"])
