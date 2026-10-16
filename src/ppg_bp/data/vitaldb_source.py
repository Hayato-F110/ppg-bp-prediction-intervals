"""Download from VitalDB: the case table (subject IDs and attributes) and the waveforms.

All requests go to https://api.vitaldb.net (checked in vitaldb 1.7.2). The package sets no timeout, so a
default socket timeout is set here; otherwise a stalled connection would wait forever.
"""

import socket

import numpy as np
import pandas as pd
import vitaldb

PPG_TRACK = "SNUADC/PLETH"
ABP_TRACK = "SNUADC/ART"
NATIVE_FS = 500
CASES_URL = "https://api.vitaldb.net/cases"
CASE_COLUMNS = ["caseid", "subjectid", "age", "sex", "height", "weight", "bmi"]
NETWORK_TIMEOUT = 600  # seconds without data before a request fails (one case file can be large)


def _set_timeout() -> None:
    """Set the default socket timeout unless the caller has already set one."""
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(NETWORK_TIMEOUT)


def list_cases() -> pd.DataFrame:
    """Return the table of cases that have both a PPG and an arterial pressure track.

    One subject can have several operations (cases), so split the data by subjectid.
    """
    _set_timeout()
    cases = pd.read_csv(CASES_URL, compression="gzip")
    ids = vitaldb.find_cases([PPG_TRACK, ABP_TRACK])
    table = cases.loc[cases["caseid"].isin(ids), CASE_COLUMNS].copy()
    table["age"] = pd.to_numeric(table["age"], errors="coerce")
    return table.sort_values("caseid").reset_index(drop=True)


def load_case_signals(case_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the PPG and arterial pressure of one case at the native sampling rate (500 Hz)."""
    _set_timeout()
    # interval is in seconds; vitaldb annotates it as int but accepts a float
    values = vitaldb.load_case(case_id, [PPG_TRACK, ABP_TRACK], 1 / NATIVE_FS)  # pyright: ignore[reportArgumentType]
    if values is None:
        raise ValueError(f"case {case_id}: no signals returned")
    return values[:, 0], values[:, 1]
