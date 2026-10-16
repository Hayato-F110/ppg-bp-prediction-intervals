"""Features of a 10-s PPG segment.

Beat-level values are summarised by their median over the beats of the segment. The unit of amplitude
differs between devices, so only ratios and times are used. Values that cannot be found stay NaN.

    APG: amplitude ratios b/a to f/a, ageing index (b-c-d-e)/a, APG index (c+d-b)/a, times from a
    pulse shape: crest time, pulse widths, area ratio, steepest upslope, height at e (near the dicrotic notch)
    pulse interval: median and variability
    signal quality: template matching (sqi_template), skewness, share of beats with the points found
"""

import numpy as np
from scipy.stats import kurtosis, skew

from ppg_bp.features.fiducials import detect_apg_points, detect_beats, preprocess

MIN_INTERVALS = 2  # pulse intervals needed for their spread
MIN_BEATS_FOR_IQR = 4  # beats with b found needed for the interquartile range of b/a

MIN_BEATS_FOR_TEMPLATE = 3
WIDTH_LEVELS = (25, 50, 75)  # heights at which pulse widths are measured (% of the beat amplitude)

FEATURE_NAMES = (
    "n_beats",
    # APG
    "b_a",
    "c_a",
    "d_a",
    "e_a",
    "f_a",
    "aging_index",
    "apg_index",
    "b_a_iqr",
    "t_ab",
    "t_ac",
    "t_ad",
    "t_ae",
    "t_af",
    "t_be",
    "t_de",
    "t_ae_rel",
    # pulse shape
    "crest_time",
    "crest_time_rel",
    "upslope_time",
    "max_slope",
    "sw25",
    "dw25",
    "sw50",
    "dw50",
    "sw75",
    "dw75",
    "width50_rel",
    "dw_sw50",
    "area_ratio",
    "t_peak_e",
    "height_e",
    # pulse interval
    "ibi_median",
    "ibi_sd",
    "ibi_rmssd",
    "ibi_cv",
    # signal quality
    "sqi_template",
    "sqi_skewness",
    "sqi_kurtosis",
    "frac_e_found",
    "frac_cd_found",
    "frac_cd_distinct",
)


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan


def _crossing(beat: np.ndarray, level: float, start: int, step: int) -> float:
    """Position (samples, linear interpolation) where beat falls below level, moving from start by step."""
    i = start
    last = len(beat) - 1
    while 0 <= i + step <= last and beat[i + step] >= level:
        i += step
    j = i + step
    if not 0 <= j <= last:
        return float(i)
    return i + step * (beat[i] - level) / (beat[i] - beat[j])


def template_sqi(filtered: np.ndarray, peaks: np.ndarray) -> float:
    """Template-matching signal quality (Orphanidou et al. 2015, DOI 10.1109/JBHI.2014.2338351).

    A window as wide as the median peak interval is cut around each peak; the mean of all beats is the
    template, and the mean correlation of each beat with the template is returned.
    With fewer than MIN_BEATS_FOR_TEMPLATE beats whose window fits in the segment, 0 is returned (a rule
    added in this study: segments that cannot be assessed count as the lowest quality).
    """
    if len(peaks) < MIN_BEATS_FOR_TEMPLATE:
        return 0.0
    half = int(np.median(np.diff(peaks))) // 2
    centres = peaks[(peaks - half >= 0) & (peaks + half < len(filtered))]
    if len(centres) < MIN_BEATS_FOR_TEMPLATE or half < 1:
        return 0.0
    beats = np.array([filtered[c - half : c + half + 1] for c in centres])
    beats = beats - beats.mean(axis=1, keepdims=True)
    template = beats.mean(axis=0)
    template -= template.mean()
    denom = np.linalg.norm(beats, axis=1) * np.linalg.norm(template)
    if not np.all(denom > 0):
        return 0.0
    return float(np.mean(beats @ template / denom))


def _morphology(
    filtered: np.ndarray,
    d1: np.ndarray,
    onsets: np.ndarray,
    peaks: np.ndarray,
    ends: np.ndarray,
    t_e: np.ndarray,
    fs: float,
) -> dict[str, np.ndarray]:
    """Pulse-shape features of each beat (dict of arrays with one value per beat)."""
    keys = (
        "crest_time",
        "crest_time_rel",
        "upslope_time",
        "max_slope",
        "width50_rel",
        "dw_sw50",
        "area_ratio",
        "t_peak_e",
        "height_e",
    )
    keys += tuple(f"{side}{level}" for level in WIDTH_LEVELS for side in ("sw", "dw"))
    out = {key: np.full(len(onsets), np.nan) for key in keys}
    for k, (start, peak, end) in enumerate(zip(onsets, peaks, ends, strict=True)):
        beat = filtered[start : end + 1] - filtered[start]
        p = peak - start
        amp = beat[p]
        if amp <= 0 or p == 0:
            continue
        duration = (end - start) / fs
        out["crest_time"][k] = p / fs
        out["crest_time_rel"][k] = p / fs / duration
        upslope = int(np.argmax(d1[start : peak + 1]))
        out["upslope_time"][k] = upslope / fs
        out["max_slope"][k] = d1[start + upslope] / amp
        for level in WIDTH_LEVELS:
            height = float(amp * level / 100.0)
            out[f"sw{level}"][k] = (p - _crossing(beat, height, p, -1)) / fs
            out[f"dw{level}"][k] = (_crossing(beat, height, p, 1) - p) / fs
        out["width50_rel"][k] = (out["sw50"][k] + out["dw50"][k]) / duration
        if out["sw50"][k] > 0:
            out["dw_sw50"][k] = out["dw50"][k] / out["sw50"][k]
        rising = np.clip(beat[: p + 1], 0, None).sum()
        if rising > 0:
            out["area_ratio"][k] = np.clip(beat[p:], 0, None).sum() / rising
        if np.isfinite(t_e[k]):
            e = t_e[k] * fs - start
            out["t_peak_e"][k] = (e - p) / fs
            out["height_e"][k] = np.interp(e, np.arange(len(beat)), beat) / amp
    return out


def _interval_features(t_a: np.ndarray, onsets: np.ndarray, ends: np.ndarray) -> dict[str, float]:
    """Median and variability of the pulse interval, from the intervals between a points of adjacent beats."""
    adjacent = onsets[1:] == ends[:-1]
    intervals = np.diff(t_a)[adjacent]
    intervals = intervals[np.isfinite(intervals)]
    if len(intervals) < MIN_INTERVALS:
        median = float(intervals[0]) if len(intervals) else np.nan
        return {"ibi_median": median, "ibi_sd": np.nan, "ibi_rmssd": np.nan, "ibi_cv": np.nan}
    median = float(np.median(intervals))
    sd = float(intervals.std(ddof=1))
    return {
        "ibi_median": median,
        "ibi_sd": sd,
        "ibi_rmssd": float(np.sqrt(np.mean(np.diff(intervals) ** 2))) if len(intervals) > MIN_INTERVALS else np.nan,
        "ibi_cv": sd / median,
    }


def segment_features(ppg: np.ndarray, fs: float) -> dict:
    """Features of one segment; the keys are FEATURE_NAMES."""
    filtered, d1, d2, d3 = preprocess(ppg, fs)
    onsets, peaks, ends = detect_beats(filtered, d1, fs)
    out: dict[str, float] = dict.fromkeys(FEATURE_NAMES, np.nan)
    out["n_beats"] = float(len(onsets))
    out["sqi_skewness"] = float(skew(filtered))
    out["sqi_kurtosis"] = float(kurtosis(filtered))
    out["sqi_template"] = template_sqi(filtered, peaks)
    if len(onsets) == 0:
        return out

    apg = detect_apg_points(d1, d2, d3, onsets, peaks, ends, fs)
    a = apg["v_a"]
    ratios = {p: apg[f"v_{p}"] / a for p in "bcdef"}
    for p in "bcdef":
        out[f"{p}_a"] = _median(ratios[p])
    out["aging_index"] = _median(ratios["b"] - ratios["c"] - ratios["d"] - ratios["e"])
    out["apg_index"] = _median(ratios["c"] + ratios["d"] - ratios["b"])
    b_found = ratios["b"][np.isfinite(ratios["b"])]
    if len(b_found) >= MIN_BEATS_FOR_IQR:
        out["b_a_iqr"] = float(np.subtract(*np.percentile(b_found, [75, 25])))
    for p in "bcdef":
        out[f"t_a{p}"] = _median(apg[f"t_{p}"] - apg["t_a"])
    out["t_be"] = _median(apg["t_e"] - apg["t_b"])
    out["t_de"] = _median(apg["t_e"] - apg["t_d"])
    out["t_ae_rel"] = _median((apg["t_e"] - apg["t_a"]) / ((ends - onsets) / fs))

    for key, values in _morphology(filtered, d1, onsets, peaks, ends, apg["t_e"], fs).items():
        out[key] = _median(values)
    out |= _interval_features(apg["t_a"], onsets, ends)

    out["frac_e_found"] = float(np.isfinite(apg["t_e"]).mean())
    out["frac_cd_found"] = float(np.isfinite(apg["t_c"]).mean())
    out["frac_cd_distinct"] = float(apg["cd_distinct"].mean())
    return out
