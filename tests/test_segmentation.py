"""Segmentation, reference values and exclusion reasons, on synthetic waveforms with known answers."""

import numpy as np
import pytest

from ppg_bp.data.segmentation import (
    REASONS,
    has_flatline,
    label_segment,
    later_rules,
    segment_case,
    segment_quantities,
)

FS = 500
SEG_SEC = 10


def synthetic(seconds=SEG_SEC, rate_bpm=60.0, sbp=120.0, dbp=80.0):
    """Arterial pressure with known pulse rate, systolic and diastolic values, and a PPG in step with it."""
    t = np.arange(int(seconds * FS)) / FS
    phase = (t * rate_bpm / 60.0) % 1.0
    pulse = np.exp(-((phase - 0.25) ** 2) / (2 * 0.08**2))  # one wave per beat (0 to 1)
    abp = dbp + (sbp - dbp) * pulse
    ppg = 50.0 + 10.0 * pulse + 0.01 * np.sin(2 * np.pi * 7 * t)
    return ppg, abp


def test_labels_match_known_pressures():
    ppg, abp = synthetic(rate_bpm=75, sbp=130, dbp=70)
    reason, sbp, dbp, mbp, rate = label_segment(ppg, abp, FS)
    assert reason == "ok"
    assert sbp == pytest.approx(130, abs=0.5)
    assert dbp == pytest.approx(70, abs=0.5)
    assert dbp < mbp < sbp
    assert rate == pytest.approx(75, abs=1)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda p, a: (np.where(np.arange(len(p)) == 10, np.nan, p), a), "nan"),
        (lambda p, a: (p, np.where(np.arange(len(a)) == 10, np.nan, a)), "nan"),
        (lambda p, a: (p, a + 250), "abp_range"),
        (lambda p, a: (p, a - 70), "abp_range"),
    ],
)
def test_exclusion_by_value(mutate, expected):
    ppg, abp = mutate(*synthetic())
    assert label_segment(ppg, abp, FS)[0] == expected


def test_flatline_is_excluded():
    ppg, abp = synthetic()
    ppg[1000:1600] = ppg[1000]  # constant for 1.2 s
    assert has_flatline(ppg, FS)
    assert label_segment(ppg, abp, FS)[0] == "flatline"


def test_short_repeat_is_not_flatline():
    ppg, _ = synthetic()
    ppg[1000:1100] = ppg[1000]  # 0.2 s
    assert not has_flatline(ppg, FS)


def test_too_slow_pulse_is_excluded():
    ppg, abp = synthetic(rate_bpm=36)  # 6 beats in 10 s: pulse rate out of range
    assert label_segment(ppg, abp, FS)[0] == "pulse_rate"


def test_irregular_beats_are_excluded():
    ppg, abp = synthetic()
    abp[: len(abp) // 2] += 40  # first half 40 mmHg higher
    assert label_segment(ppg, abp, FS)[0] == "beat_irregular"


def test_damped_waveform_is_excluded():
    ppg, abp = synthetic(sbp=92, dbp=80)  # pulse pressure 12 mmHg
    assert label_segment(ppg, abp, FS)[0] in ("few_beats", "pulse_pressure")


def test_segment_case_counts_and_shapes():
    good_ppg, good_abp = synthetic(seconds=30)
    ppg, abp = good_ppg.copy(), good_abp.copy()
    ppg[SEG_SEC * FS + 5] = np.nan  # missing values in the second segment only
    ppg_out, rows, counts = segment_case(ppg, abp, FS, SEG_SEC, out_fs=125)
    assert sum(counts.values()) == 3
    assert counts["ok"] == 2 and counts["nan"] == 1
    assert set(counts) == set(REASONS)
    assert [r[0] for r in rows] == [0, 2]
    assert ppg_out.shape == (2, SEG_SEC * 125)
    assert ppg_out.dtype == np.float32
    assert not np.isnan(ppg_out).any()


def test_downsampled_ppg_matches_original():
    ppg, abp = synthetic(seconds=20)
    ppg_out, _, _ = segment_case(ppg, abp, FS, SEG_SEC, out_fs=125)
    # apart from the outer 0.5 s, nearly equal to every 4th sample of the original
    original = ppg[: SEG_SEC * FS : 4]
    assert np.max(np.abs(ppg_out[0][63:-63] - original[63:-63])) < 0.05


def test_no_valid_segment_returns_empty():
    ppg, abp = synthetic(seconds=20)
    ppg_out, rows, counts = segment_case(ppg, abp + 400, FS, SEG_SEC, out_fs=125)
    assert ppg_out.shape == (0, SEG_SEC * 125)
    assert rows == []
    assert counts["abp_range"] == 2


def test_wider_limits_of_the_later_rules_admit_a_slow_pulse():
    ppg, abp = synthetic(rate_bpm=36)  # excluded by the main rules (pulse rate below 40 per minute)
    reason, quantities = segment_quantities(ppg, abp, FS)
    assert reason == "" and quantities is not None
    assert later_rules(quantities) == "pulse_rate"
    assert later_rules(quantities, (20.0, 200.0), 20.0) == "ok"  # the widest rules of the sensitivity analysis


def test_first_rules_return_no_quantities():
    ppg, abp = synthetic()
    abp[100] = np.nan
    assert segment_quantities(ppg, abp, FS) == ("nan", None)
