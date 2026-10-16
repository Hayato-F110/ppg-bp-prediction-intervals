"""Beat segmentation of the PPG and detection of the points a to f of the acceleration plethysmogram
(APG, the second derivative of the PPG).

The point definitions and the filter band follow the pyPPG paper:
Goda, Charlton, Behar (2024) Physiol. Meas. 45 045001, DOI 10.1088/1361-6579/ad33a2, Table 3.
The implementation is our own (pyPPG's code is not used). Validation: scripts/validate_fiducials.py.

    a: maximum of the APG on the upslope
    b: first local minimum after a
    e: largest local maximum after the systolic peak within 60% of the beat (else up to the end of the beat)
    c: largest local maximum between b and e; without one, the inflection point (minimum slope of the APG)
    d: smallest local minimum between c and e; without one, the same inflection point as c
    f: first local minimum after e

In beats where c and d do not appear as extrema, c and d are the same point (cd_distinct is False).
"""

import itertools

import numpy as np
from scipy.signal import cheby2, find_peaks, sosfiltfilt

MIN_UPSLOPES = 2  # beats need an upslope before and after

BAND_HZ = (0.5, 12.0)  # as in the paper; 4th-order Chebyshev type II
STOPBAND_DB = 20.0  # not given in the paper; set in this study
SMOOTH_PPG_SEC = 0.050  # as in the paper; moving average after band-pass filtering
SMOOTH_DERIV_SEC = 0.010  # as in the paper; moving average after each derivative

MIN_BEAT_SEC, MAX_BEAT_SEC = 0.33, 1.5  # pulse rate 40-180 per minute
UPSLOPE_REL_HEIGHT = 0.4  # height of the first derivative taken as an upslope (relative to the upper beats)
ONSET_LOOKBACK_SEC = 0.3
A_MARGIN_SEC = 0.04  # a can come slightly before the beat onset (the PPG minimum)
E_LIMIT = 0.6  # upper limit of the search for e (share of the beat length); as in the paper

POINTS = ("a", "b", "c", "d", "e", "f")


def _smooth(x: np.ndarray, sec: float, fs: float) -> np.ndarray:
    """Moving average applied forwards and backwards (as in pyPPG's public code; no phase shift)."""
    n = round(sec * fs)
    if n <= 1:
        return x
    box = np.full(n, 1.0 / n)
    padded = np.pad(x, n - 1, mode="edge")
    return np.convolve(padded, np.convolve(box, box), mode="valid")


def preprocess(ppg: np.ndarray, fs: float) -> tuple[np.ndarray, ...]:
    """Band-pass filter and smooth; return (PPG, first, second and third derivatives)."""
    sos = cheby2(4, STOPBAND_DB, BAND_HZ, btype="bandpass", fs=fs, output="sos")
    filtered = _smooth(sosfiltfilt(sos, np.asarray(ppg, dtype=np.float64)), SMOOTH_PPG_SEC, fs)
    derivs = [filtered]
    for _ in range(3):
        derivs.append(_smooth(np.gradient(derivs[-1]) * fs, SMOOTH_DERIV_SEC, fs))
    return tuple(derivs)


def detect_beats(filtered: np.ndarray, d1: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the positions (samples) of the beat onsets, systolic peaks and beat ends.

    The upslopes (peaks of the first derivative) are found first; the minimum before each is the beat onset.
    Only beats between two onsets and of plausible length are returned.
    """
    empty = np.empty(0, dtype=np.intp)
    upslopes, props = find_peaks(d1, distance=max(1, int(MIN_BEAT_SEC * fs)), height=0)
    if len(upslopes) < MIN_UPSLOPES:
        return empty, empty, empty
    heights = props["peak_heights"]
    upslopes = upslopes[heights >= UPSLOPE_REL_HEIGHT * np.percentile(heights, 75)]

    # The onset is the minimum just before the upslope. Even if a lower point lies earlier in diastole,
    # the start of the upslope is taken (the paper's definition: "start of the systolic upslope").
    lookback = int(ONSET_LOOKBACK_SEC * fs)
    troughs, _ = find_peaks(-filtered)
    feet = []
    for i, u in enumerate(upslopes):
        lo = max(u - lookback, upslopes[i - 1] if i else 0)
        cand = _between(troughs, lo, u)
        feet.append(cand[-1] if len(cand) else -1)
    feet = np.array(feet)

    onsets, peaks, ends = [], [], []
    for start, end in itertools.pairwise(feet):
        if start < 0 or end < 0:
            continue
        if not MIN_BEAT_SEC <= (end - start) / fs <= MAX_BEAT_SEC:
            continue
        onsets.append(start)
        peaks.append(start + int(np.argmax(filtered[start:end])))
        ends.append(end)
    return np.array(onsets, dtype=np.intp), np.array(peaks, dtype=np.intp), np.array(ends, dtype=np.intp)


def _refine(y: np.ndarray, i: int) -> tuple[float, float]:
    """Refine the position and value of an extremum to sub-sample accuracy with a parabola through 3 points."""
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    left, mid, right = y[i - 1], y[i], y[i + 1]
    curvature = left - 2.0 * mid + right
    if curvature == 0:
        return float(i), float(mid)
    shift = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return i + shift, float(mid - 0.25 * (left - right) * shift)


def _between(candidates: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Candidates with lo < position < hi."""
    return candidates[np.searchsorted(candidates, lo, side="right") : np.searchsorted(candidates, hi, side="left")]


def _find_e(maxima: np.ndarray, d2: np.ndarray, after: int, start: int, end: int) -> int | None:
    """The e wave: the largest maximum of the APG after `after` within E_LIMIT of the beat, else up to its end.

    Neither rule is in the paper; both are rules of pyPPG's public code (fiducials.py). Returns None if not found.
    """
    cand = _between(maxima, after, start + E_LIMIT * (end - start) + 1)
    if len(cand) == 0:
        cand = _between(maxima, after, end)
    if len(cand) == 0:
        return None
    return cand[np.argmax(d2[cand])]


def detect_apg_points(
    d1: np.ndarray, d2: np.ndarray, d3: np.ndarray, onsets: np.ndarray, peaks: np.ndarray, ends: np.ndarray, fs: float
) -> dict[str, np.ndarray]:
    """Times (s) and APG values of a to f for each beat. Points not found are NaN.

    Keys: t_a, v_a, ..., t_f, v_f (arrays with one value per beat), and
    cd_distinct (whether c and d appeared as extrema).
    """
    n = len(onsets)
    out = {f"{kind}_{p}": np.full(n, np.nan) for p in POINTS for kind in ("t", "v")}
    out["cd_distinct"] = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    maxima, _ = find_peaks(d2)
    minima, _ = find_peaks(-d2)
    slope_minima, _ = find_peaks(-d3)
    margin = round(A_MARGIN_SEC * fs)

    def put(k: int, point: str, index: int, refine_on: np.ndarray = d2) -> None:
        pos, value = _refine(refine_on, index)
        out[f"t_{point}"][k] = pos / fs
        out[f"v_{point}"][k] = value if refine_on is d2 else np.interp(pos, np.arange(len(d2)), d2)

    for k, (start, peak, end) in enumerate(zip(onsets, peaks, ends, strict=True)):
        # a lies before the steepest point (peak of the first derivative)
        upslope = start + int(np.argmax(d1[start : peak + 1]))
        cand = _between(maxima, start - margin - 1, upslope + 1)
        if len(cand) == 0:
            continue
        a = cand[np.argmax(d2[cand])]
        put(k, "a", a)

        cand = _between(minima, a, end)
        if len(cand) == 0:
            continue
        b = cand[0]
        put(k, "b", b)

        e = _find_e(maxima, d2, max(b, peak), start, end)
        if e is None:
            continue
        put(k, "e", e)

        cand = _between(maxima, b, e)
        if len(cand):
            c = cand[np.argmax(d2[cand])]
            between = _between(minima, c, e)
            d = between[np.argmin(d2[between])]
            put(k, "c", c)
            put(k, "d", d)
            out["cd_distinct"][k] = True
        else:
            cand = _between(slope_minima, b, e)
            if len(cand):
                shoulder = cand[np.argmin(d3[cand])]
                put(k, "c", shoulder, refine_on=-d3)
                put(k, "d", shoulder, refine_on=-d3)

        cand = _between(minima, e, end)
        if len(cand):
            put(k, "f", cand[0])
    return out


def analyse(ppg: np.ndarray, fs: float) -> dict:
    """Beats and the points a to f of one PPG segment.

    Returns the keys of detect_apg_points and t_onset, t_peak and t_end (s).
    """
    filtered, d1, d2, d3 = preprocess(ppg, fs)
    onsets, peaks, ends = detect_beats(filtered, d1, fs)
    out = detect_apg_points(d1, d2, d3, onsets, peaks, ends, fs)
    out["t_onset"], out["t_peak"], out["t_end"] = onsets / fs, peaks / fs, ends / fs
    return out
