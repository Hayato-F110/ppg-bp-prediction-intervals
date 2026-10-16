"""Cut continuous waveforms into 10-s segments and attach the reference pressures and the exclusion reason.

The segment length and the reference values follow PulseDB (Wang et al. 2023):
SBP and DBP are the means of the beat-by-beat values in the segment, and MBP is the time average of the arterial
pressure.
"""

import itertools

import numpy as np
from scipy.signal import find_peaks, resample_poly

# Exclusion reasons, checked from top to bottom; the first one that applies is recorded.
REASONS = (
    "ok",
    "nan",  # missing samples in the PPG or the arterial pressure
    "abp_range",  # arterial pressure outside a physiological range (open line, flush)
    "flatline",  # constant for 1 s or longer (sensor off, saturation)
    "few_beats",  # too few beats found in the arterial pressure
    "pulse_rate",  # pulse rate out of range
    "beat_irregular",  # large beat-to-beat SBP variation (movement, blood sampling, line handling)
    "pulse_pressure",  # pulse pressure too small (damped waveform)
)

MIN_PEAKS = 2  # a pulse interval needs two peaks

ABP_MIN, ABP_MAX = 20.0, 300.0
FLAT_SEC = 1.0
MIN_BEATS = 5
PULSE_RATE_RANGE = (40.0, 180.0)
MAX_BEAT_SBP_SD = 15.0
MIN_PULSE_PRESSURE = 20.0


def has_flatline(x: np.ndarray, fs: int) -> bool:
    """Whether the signal stays at the same value for FLAT_SEC seconds or longer."""
    same = np.diff(x) == 0
    if not same.any():
        return False
    edges = np.diff(np.concatenate(([0], same.astype(np.int8), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return bool(runs.max() >= int(FLAT_SEC * fs) - 1)


def beat_pressures(abp: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the systolic peak of each beat in the arterial pressure, and the diastolic trough between peaks."""
    peaks, _ = find_peaks(abp, distance=int(0.3 * fs), prominence=10)
    if len(peaks) < MIN_PEAKS:
        return peaks, np.array([]), np.array([])
    troughs = np.array([abp[a:b].min() for a, b in itertools.pairwise(peaks)])
    return peaks, abp[peaks], troughs


def label_segment(ppg: np.ndarray, abp: np.ndarray, fs: int) -> tuple[str, float, float, float, float]:  # noqa: PLR0911 (one return per exclusion rule)
    """Label one segment. Returns (reason, SBP, DBP, MBP, pulse rate); the numbers are NaN when excluded."""
    nan = (np.nan, np.nan, np.nan, np.nan)
    if np.isnan(ppg).any() or np.isnan(abp).any():
        return "nan", *nan
    if abp.min() < ABP_MIN or abp.max() > ABP_MAX:
        return "abp_range", *nan
    if has_flatline(ppg, fs) or has_flatline(abp, fs):
        return "flatline", *nan
    peaks, sbp_beats, dbp_beats = beat_pressures(abp, fs)
    if len(peaks) < MIN_BEATS:
        return "few_beats", *nan
    pulse_rate = 60.0 / (np.median(np.diff(peaks)) / fs)
    if not PULSE_RATE_RANGE[0] <= pulse_rate <= PULSE_RATE_RANGE[1]:
        return "pulse_rate", *nan
    if sbp_beats.std() > MAX_BEAT_SBP_SD:
        return "beat_irregular", *nan
    sbp, dbp = float(sbp_beats.mean()), float(dbp_beats.mean())
    if sbp - dbp < MIN_PULSE_PRESSURE:
        return "pulse_pressure", *nan
    return "ok", sbp, dbp, float(abp.mean()), float(pulse_rate)


def segment_case(
    ppg: np.ndarray, abp: np.ndarray, fs: int, seg_sec: int, out_fs: int
) -> tuple[np.ndarray, list[tuple], dict[str, int]]:
    """Cut one case into non-overlapping segments.

    Returns:
        ppg_out: PPG of the kept segments (resampled to out_fs, float32, shape n_kept x seg_sec*out_fs)
        table: one tuple per kept segment (index, sbp, dbp, mbp, pulse_rate)
        counts: number of segments per exclusion reason
    """
    seg = seg_sec * fs
    n_seg = len(ppg) // seg
    counts: dict[str, int] = dict.fromkeys(REASONS, 0)
    rows, kept = [], []
    for i in range(n_seg):
        sl = slice(i * seg, (i + 1) * seg)
        reason, sbp, dbp, mbp, rate = label_segment(ppg[sl], abp[sl], fs)
        counts[reason] += 1
        if reason == "ok":
            kept.append(i)
            rows.append((i, sbp, dbp, mbp, rate))

    out_len = seg_sec * out_fs
    if not kept:
        return np.empty((0, out_len), dtype=np.float32), rows, counts

    # Resample the continuous signal, so that filter edge effects do not appear at segment boundaries.
    # Gaps are interpolated before resampling; segments containing gaps were already excluded above.
    filled = ppg[: n_seg * seg].astype(np.float64)
    missing = np.isnan(filled)
    if missing.any():
        idx = np.arange(len(filled))
        filled[missing] = np.interp(idx[missing], idx[~missing], filled[~missing])
    down = resample_poly(filled, out_fs, fs).astype(np.float32)
    ppg_out = down[: n_seg * out_len].reshape(n_seg, out_len)[kept]
    return ppg_out, rows, counts
