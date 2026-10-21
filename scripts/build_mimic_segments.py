"""Download PPG and arterial pressure from the MIMIC-III Waveform Database Matched Subset and cut 10-s segments
as for VitalDB.

Data: https://physionet.org/content/mimic3wdb-matched/1.0/ (open access, ODbL v1.0).
Only the waveforms (PLETH and ABP, 125 Hz) are read; the credentialed clinical data are not used.
Segments, reference values and exclusion rules are those of VitalDB (src/ppg_bp/data/segmentation.py).

The database is 2.4 TB, so subjects are visited in a random order and at most MAX_HOURS hours per subject are read
from records with both PLETH and ABP. If stopped, a rerun skips the subjects already done.

The subjects of the paper were those reached before a time limit, so they depended on download speed. To use the
same subjects, pass the published list (data_lists/mimic3wdb_subjects.csv) with --subject-list.

usage: uv run python -X utf8 scripts/build_mimic_segments.py --subject-list data_lists/mimic3wdb_subjects.csv
       uv run python -X utf8 scripts/build_mimic_segments.py [--subjects 1000] [--hours-limit 7] [--workers 4]
output: data/processed/mimic3wdb/segments/<subject>.npz, subject_summary.csv
log: results/logs/mimic_build.log
"""

import argparse
import csv
import socket
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import wfdb

from ppg_bp.config import PROJECT_ROOT, load_config
from ppg_bp.data.segmentation import REASONS, segment_case
from ppg_bp.runlog import RunLog

DATABASE = "mimic3wdb-matched/1.0"
RECORDS_URL = f"https://physionet.org/files/{DATABASE}/RECORDS-waveforms"
SIGNALS = ("PLETH", "ABP")
# hours per subject; PhysioNet transfers are slow (about 70 KB/s with 8 connections),
# so more subjects are preferred to longer recordings
MAX_HOURS = 0.5
MIN_SEGMENT_SEC = 600  # shorter segments are skipped
MAX_SEGMENTS_CHECKED = 40  # at most this many segment headers are examined per record
MAX_RECORDS_PER_SUBJECT = 4

CONFIG = load_config()
OUT_DIR = PROJECT_ROOT / CONFIG["paths"]["processed"] / "mimic3wdb"
LOG = PROJECT_ROOT / CONFIG["paths"]["results"] / "logs" / "mimic_build.log"
FIELDS = ["subject", "record", "fs", "start_sec", "hours", "n_segments", *REASONS, "status"]
log = RunLog(LOG)


def list_records() -> dict:
    """Subject -> list of record paths (e.g. p00/p000020/p000020-2183-04-28-17-47)."""
    with urllib.request.urlopen(RECORDS_URL, timeout=120) as response:
        lines = response.read().decode().split()
    subjects = {}
    for line in lines:
        subjects.setdefault(line.split("/")[1], []).append(line)
    return subjects


def listed_records(path: Path) -> dict:
    """Subject -> [record] for the subjects of a published list (scripts/export_id_lists.py) with outcome ok.

    Within a record, the segments read are fixed (the first ones with both signals, up to MAX_HOURS), so reading the
    same record gives the same 10-s segments.
    """
    with path.open(encoding="utf-8") as f:
        return {row["subject"]: [row["record"]] for row in csv.DictReader(f) if row["status"] == "ok"}


def read_segment(seg_name: str, pn_dir: str, header, take: int):
    """Read the first take samples of a segment.

    Remote reading by wfdb sends many small requests and is slow. For the usual 16-bit single-file format, the
    needed bytes are fetched in one request and read locally; other formats are left to wfdb.
    """
    simple = (
        set(header.fmt) == {"16"}
        and len(set(header.file_name)) == 1
        and not any(header.byte_offset or [0])
        and set(header.samps_per_frame) == {1}
    )
    if not simple:
        return wfdb.rdrecord(seg_name, pn_dir=pn_dir, sampto=take, channel_names=list(SIGNALS))
    base = f"https://physionet.org/files/{pn_dir}/"
    n_bytes = take * header.n_sig * 2
    with tempfile.TemporaryDirectory() as folder:
        for name, headers in ((f"{seg_name}.hea", {}), (header.file_name[0], {"Range": f"bytes=0-{n_bytes - 1}"})):
            # The file name comes from the downloaded header; refuse anything but a plain name inside the folder
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError(f"unexpected file name in header: {name!r}")
            with urllib.request.urlopen(urllib.request.Request(base + name, headers=headers), timeout=600) as response:  # noqa: S310 (fixed https URL)
                (Path(folder) / name).write_bytes(response.read())
        return wfdb.rdrecord(str(Path(folder) / seg_name), sampto=take, channel_names=list(SIGNALS))


def read_pieces(_record_name: str, pn_dir: str, master) -> tuple:
    """Read, in order, the segments of a record that contain both PLETH and ABP.

    Reading several segments at once is slow, so they are read one by one. Returns (PPG, arterial pressure,
    start of the first segment). Each segment is trimmed to a multiple of 10 s before joining.
    """
    fs = round(master.fs)
    seg = CONFIG["segment"]["length_sec"] * fs
    quota = int(MAX_HOURS * 3600 * fs)
    ppg_parts, abp_parts, first_start, position, checked = [], [], None, 0, 0
    for seg_name, seg_len in zip(master.seg_name, master.seg_len, strict=False):
        start, position = position, position + seg_len
        if seg_name.endswith("_layout") or seg_name == "~" or seg_len < MIN_SEGMENT_SEC * fs:
            continue
        if checked >= MAX_SEGMENTS_CHECKED or quota < seg:
            break
        checked += 1
        header = wfdb.rdheader(seg_name, pn_dir=pn_dir)
        if not all(s in (header.sig_name or []) for s in SIGNALS):
            continue
        take = min(seg_len, quota) // seg * seg
        record = read_segment(seg_name, pn_dir, header, take)
        if record.p_signal is None or record.sig_name is None:
            raise AssertionError("the record has no signal data")
        ppg_parts.append(record.p_signal[:, record.sig_name.index("PLETH")])
        abp_parts.append(record.p_signal[:, record.sig_name.index("ABP")])
        first_start = start if first_start is None else first_start
        quota -= take
    if not ppg_parts:
        return None, None, 0
    return np.concatenate(ppg_parts), np.concatenate(abp_parts), first_start


def fetch_subject(subject: str, records: list) -> dict:
    """Download and save one subject; return one row of the summary."""
    row = {
        "subject": subject,
        "record": "",
        "fs": 0,
        "start_sec": 0,
        "hours": 0.0,
        "n_segments": 0,
        "status": "no_signals",
        **dict.fromkeys(REASONS, 0),
    }
    seg_sec, out_fs = CONFIG["segment"]["length_sec"], CONFIG["segment"]["fs_hz"]
    for path in records[:MAX_RECORDS_PER_SUBJECT]:
        directory, name = path.rsplit("/", 1)
        pn_dir = f"{DATABASE}/{directory}"
        try:
            master = wfdb.rdheader(name, pn_dir=pn_dir)
            if not (master.seg_name is not None):
                raise AssertionError("not a multi-segment record")
            layout = wfdb.rdheader(master.seg_name[0], pn_dir=pn_dir)
            if not all(s in (layout.sig_name or []) for s in SIGNALS):
                continue
            ppg, abp, start = read_pieces(name, pn_dir, master)
        except Exception as error:  # e.g. a network failure; try the next record
            row["status"] = f"error:{type(error).__name__}"
            continue
        if ppg is None:
            continue
        fs = round(master.fs)
        ppg_out, rows, counts = segment_case(ppg, abp, fs, seg_sec, out_fs)
        row |= {
            "record": path,
            "fs": fs,
            "start_sec": start // fs,
            "hours": len(ppg) / fs / 3600,
            "n_segments": sum(counts.values()),
            "status": "ok",
            **counts,
        }
        if rows:
            index, sbp, dbp, mbp, rate = (np.array(v) for v in zip(*rows, strict=False))
            np.savez(
                OUT_DIR / "segments" / f"{subject}.npz",
                ppg=ppg_out,
                index=index.astype(np.int32),
                sbp=sbp.astype(np.float32),
                dbp=dbp.astype(np.float32),
                mbp=mbp.astype(np.float32),
                pulse_rate=rate.astype(np.float32),
            )
            return row
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects", type=int, default=1000, help="stop when this many subjects have at least one kept segment"
    )
    parser.add_argument("--hours-limit", type=float, default=7.0, help="after this many hours, start no new subject")
    parser.add_argument("--workers", type=int, default=6, help="parallel connections (16 were refused on 2026-10-13)")
    parser.add_argument("--max-tried", type=int, default=None, help="maximum number of subjects to try (for testing)")
    parser.add_argument(
        "--subject-list",
        type=Path,
        default=None,
        help="read only the subjects of this list (data_lists/mimic3wdb_subjects.csv) with outcome ok, from the record "
        "given there; this reproduces the subjects of the paper, whose selection depended on download speed",
    )
    args = parser.parse_args()
    # wfdb (https, via requests) sets no timeout; without one a stalled connection waits forever
    socket.setdefaulttimeout(600)

    (OUT_DIR / "segments").mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "subject_summary.csv"
    done = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            done = {r["subject"]: r for r in csv.DictReader(f)}
    else:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    if args.subject_list:
        subjects = listed_records(args.subject_list)
        order = list(subjects)  # the order of the list (the order visited in the paper)
        args.subjects, args.hours_limit = len(subjects), float("inf")  # read every listed subject
        # a listed subject that failed (e.g. a network error) is tried again; its new row is appended
        done = {s: r for s, r in done.items() if r["status"] == "ok"}
    else:
        subjects = list_records()
        order = np.random.default_rng(CONFIG["seed"]).permutation(sorted(subjects))
    todo = [s for s in order if s not in done][: args.max_tried]
    n_ok = sum(1 for r in done.values() if int(r["ok"]) > 0)
    log(f"=== start: {len(subjects)} subjects, {len(done)} done ({n_ok} with kept segments), target {args.subjects}")

    start, tried = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for batch_start in range(0, len(todo), args.workers * 4):
            if n_ok >= args.subjects or time.time() - start > args.hours_limit * 3600:
                break
            batch = todo[batch_start : batch_start + args.workers * 4]
            for row in pool.map(lambda s: fetch_subject(s, subjects[s]), batch):
                tried += 1
                n_ok += int(row["ok"] > 0)
                with summary_path.open("a", encoding="utf-8", newline="") as f:
                    csv.DictWriter(f, FIELDS).writerow(row)
            log(f"tried {tried}, with kept segments {n_ok}, {(time.time() - start) / 60:.1f} min")
    log(f"=== done: {n_ok} subjects with kept segments")
    if args.subject_list:
        compare_with_list(summary_path, args.subject_list)


def compare_with_list(summary_path: Path, list_path: Path) -> None:
    """Report listed subjects whose start (s) or number of kept segments differs from the published list."""
    with summary_path.open(encoding="utf-8") as f:
        read = {r["subject"]: r for r in csv.DictReader(f)}  # the last row of each subject
    with list_path.open(encoding="utf-8") as f:
        listed = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    differ = [
        r["subject"]
        for r in listed
        if r["subject"] not in read
        or read[r["subject"]]["status"] != "ok"
        or int(read[r["subject"]]["start_sec"]) != int(r["start_sec"])
        or int(read[r["subject"]]["ok"]) != int(r["ok"])
    ]
    log(f"=== compared with {list_path.name}: {len(listed) - len(differ)} of {len(listed)} subjects match")
    if differ:
        log(f"differing or missing (run again to retry failures): {', '.join(differ[:20])}")


if __name__ == "__main__":
    main()
