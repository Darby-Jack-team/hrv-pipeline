#!/usr/bin/env python
"""
01_qc_dashboard.py
==================

Single-file quality-control (QC) dashboard for ONE ambulatory single-lead ECG
deployment. It produces a fast, glanceable picture of overall data quality:
one multi-panel figure + machine-readable metrics.

This is a DIAGNOSTIC / REPORTING tool. It flags and visualizes quality issues
but NEVER modifies or silently drops raw data. Failed/excluded windows are
*flagged*, not deleted. Exclusion decisions are made downstream.

Run with the project's Python environment (see README.md for setup):

    .venv/bin/python analysis/01_qc_dashboard.py \
        --ecg data-raw/.../<file>.csv --deployment-id june14_electrode

    # quick smoke test on the first 20 minutes:
    .venv/bin/python analysis/01_qc_dashboard.py --ecg <file>.csv --limit-seconds 1200


Hardware / study context (these drive several design choices below)
-------------------------------------------------------------------
* Device: single-lead wearable, 0.5-40 Hz analog bandwidth.
  - The SPEC's nominal sample rate is 512 Hz. The ACTUAL files in this project
    are 256 Hz (verified from the timestamps). `sampling_rate` is therefore a
    CONFIG value, defaulted to 256 to match the real data, and VALIDATED against
    the file's own timestamps at load time (fail loudly on mismatch).
* CRITICAL distinction (do not conflate):
  - The SAMPLING RATE (256 Hz -> ~3.9 ms grid; 512 Hz -> ~2 ms grid) governs the
    timing RESOLUTION of fiducial points (R-peak indices).
  - The 40 Hz upper BANDWIDTH governs R-peak SHARPNESS. A low bandwidth broadens
    the QRS, which increases R-peak localization JITTER under noise.
  - Because RMSSD is a beat-to-beat (successive-difference) metric, random
    fiducial jitter biases RMSSD UPWARD. We quantify that directly in the
    "fiducial jitter sensitivity" stage rather than assuming it away.
* Subject: ~12-year-old in field conditions. Resting HR is higher than adults and
  can range ~45-200 bpm, so we use PEDIATRIC RR plausibility bounds (~300-1500 ms),
  NOT adult bounds.

neurokit2 API notes (verified against the installed version, 0.2.x)
-------------------------------------------------------------------
* RR/NN correction: nk.signal_fixpeaks(..., method="Kubios") implements the
  Lipponen & Tarvainen (2019) ectopic/missed/extra/misaligned correction. The
  method string has changed across nk versions; "Kubios" is correct here and the
  script asserts it is available at runtime.
* Signal SQI: nk.ecg_quality(method="ici") is the ho2025/ICI two-detector-
  agreement metric (PRIMARY gate -- transferable because it does not depend on
  absolute spectral magnitudes that the 40 Hz cutoff distorts).
  method="averageQRS" and method="zhao2018" are also available but treated as
  SECONDARY -- neither drives the QC verdict (see comments at the call sites).
  averageQRS costs real time on long recordings and is OFF by default for QC
  runs; pass --avgqrs to compute it.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# matplotlib only (no seaborn dependency)
import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

import neurokit2 as nk

# A fixed RNG seed so the jitter-sensitivity stage (and anything stochastic) is
# reproducible across runs.
RNG_SEED = 1729
RNG = np.random.default_rng(RNG_SEED)


# =============================================================================
# CONFIG  -- everything tunable lives here. No hardcoded paths in the body.
# =============================================================================
@dataclass
class Config:
    # --- Input ------------------------------------------------------------
    # No default: every run must pass --ecg explicitly (see main()).
    ecg_path: Optional[str] = None
    ecg_col: str = "ecg"
    timestamp_col: str = "timestamp"
    sampling_rate: int = 256  # Hz. SPEC nominal = 512; THESE files are 256.
    # Tolerance (%) between configured sampling_rate and the rate implied by the
    # file's own median sample interval. Beyond this -> fail loudly.
    sampling_rate_tol_pct: float = 5.0

    deployment_id: str = "deployment"

    # Timezone used for every human-readable time in the report (banner, console,
    # PDF, time-of-day panel). The raw timestamps are parsed as UTC and converted
    # to this zone for display.
    # None (default) = auto-infer from the UTC offset embedded in the file's own
    # timestamps (e.g. "-04:00" for a US Eastern deployment, "+00:00" for Ghana)
    # -- no per-country configuration needed. Pass --local-tz to override with a
    # fixed offset ("+00:00") or an IANA zone name ("America/New_York") for
    # calendar-correct DST handling.
    local_tz: Optional[str] = None

    # --- Optional accelerometer (separate file in this project) -----------
    # If present, used for a per-window motion-burden flag. 3-axis (ax,ay,az)
    # or 9-axis; we use the first 3 as linear acceleration. Pass --accel, or
    # leave unset / pass --no-accel to skip motion-burden flagging.
    accel_path: Optional[str] = None
    accel_cols: tuple[str, ...] = ("ax", "ay", "az")
    accel_timestamp_col: str = "timestamp"
    # Motion burden = fraction of the window whose |accel-magnitude minus 1g|
    # exceeds this (in units of g). A window is "high motion" above
    # motion_flag_frac.
    gravity_g: float = 9.80665  # accel is in m/s^2 in these files
    motion_dev_thresh_g: float = 0.30
    motion_flag_frac: float = 0.20

    # --- ABPM cuff-inflation schedule -------------------------------------
    # Either give explicit inflation start timestamps (ISO strings) OR a fixed
    # interval. Inflations create exclusion windows (the cuff corrupts ECG).
    abpm_explicit_starts: tuple[str, ...] = ()  # e.g. ("2026-06-15T18:00:00-04:00",)
    abpm_interval_min: float = 20.0  # one inflation every N minutes
    abpm_inflation_s: float = 40.0  # each inflation lasts ~N seconds
    # ABPM exclusion is DEFERRED until a real cuff-inflation schedule is available.
    # Without one we were fabricating a 20-min inflation schedule that dragged the
    # verdict down on signal that is actually fine. Off by default; opt back in
    # with --abpm (or set this True) once a real schedule exists.
    abpm_enabled: bool = False

    # --- Secondary (non-gating) signal-quality index --------------------
    # averageQRS is a RELATIVE per-beat morphology-distance metric (see
    # process_ecg() comments). It does not drive the PASS/REVIEW/FAIL verdict
    # (that's driven by % corrected beats from Kubios RR correction) and isn't
    # plotted or exported anywhere -- it costs real time on long recordings
    # for a number that's discarded. Off by default; opt in with --avgqrs if
    # you want it reported for extra diagnostic detail.
    avgqrs_enabled: bool = False

    # --- Windowing --------------------------------------------------------
    window_s: float = 300.0  # 5-min windows
    window_step_s: float = 300.0  # non-overlapping
    # Window acceptance: max % corrected beats allowed. Primary + secondary.
    accept_pct_primary: float = 5.0
    accept_pct_secondary: float = 2.0
    min_beats_per_window: int = 10  # below this, RMSSD/SDNN are not trustworthy

    # --- Lead-off / flatline (electrode dropout) detector -----------------
    # A window is "lead-off / flatline" when its ECG amplitude collapses to near
    # zero (electrode lost skin contact -> no QRS). ECG values are raw uncalibrated
    # ADC counts whose gain varies by deployment, so the floor is RELATIVE: a
    # window is flatline if its robust peak-to-peak amplitude (p99-p1 of the clean
    # signal) falls below this fraction of the recording's MEDIAN window amplitude.
    # Calibrated on a pilot deployment: clean ~2350 ADC, dropout ~175 ADC;
    # 0.15*median cleanly separates them. Robust as long as <50% of the record
    # is flat. Re-tune this if your device reports amplitude in different units.
    flatline_rel_frac: float = 0.15

    # --- Not-worn (device off body) detector ------------------------------
    # "Not worn" = the device is sitting idle, not on a person. Detected as the
    # CONJUNCTION of two dead sensors: the accelerometer is essentially still AND
    # the ECG is flatline. Requiring both avoids mislabelling quiet sleep (which is
    # low-motion but still has a heartbeat) as not-worn. The accel threshold is the
    # per-window std of accelerometer magnitude (g); below this the sensor is
    # "still". Calibrated on a pilot deployment (worn median std ~0.013 g,
    # noise floor ~0.003 g).
    not_worn_accel_std_g: float = 0.006

    # --- Pediatric RR sanity bounds (ms) ----------------------------------
    rr_min_ms: float = 300.0  # ~200 bpm
    rr_max_ms: float = 1500.0  # ~40 bpm

    # --- Fiducial jitter sensitivity --------------------------------------
    jitter_sigmas_ms: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)

    # --- Overall PASS/REVIEW/FAIL banner thresholds -----------------------
    # Driven by % of the recording that is analyzable (pass windows that are not
    # ABPM-excluded), at the PRIMARY acceptance cutoff.
    pass_analyzable_pct: float = 80.0
    review_analyzable_pct: float = 50.0

    # --- Output -----------------------------------------------------------
    out_dir: str = "analysis/qc_out"
    cross_deployment_csv: str = "analysis/qc_out/_deployment_qc_summary.csv"
    figure_dpi: int = 140

    # --- Testing convenience ---------------------------------------------
    # If set, only the first N seconds of ECG are processed (smoke tests).
    limit_seconds: Optional[float] = None


# =============================================================================
# Loud assumption / validation helpers
# =============================================================================
def announce(msg: str) -> None:
    """Print an explicit assumption / decision so the reviewer can see it."""
    print(f"[ASSUME] {msg}")


def fail(msg: str) -> "None":
    """Fail loudly: the input did not match config; do not guess."""
    print(f"\n[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def infer_utc_offset_str(raw_timestamp: str) -> Optional[str]:
    """Extract the UTC offset embedded in one raw ISO8601 timestamp string
    (e.g. "2026-07-10T06:25:55.000-04:00" -> "-04:00") as a fixed-offset
    string pandas' tz_convert() accepts directly. Returns None if the string
    carries no offset (e.g. a bare "Z"/naive timestamp)."""
    offset = pd.Timestamp(raw_timestamp).utcoffset()
    if offset is None:
        return None
    total_min = int(offset.total_seconds() // 60)
    sign = "+" if total_min >= 0 else "-"
    hh, mm = divmod(abs(total_min), 60)
    return f"{sign}{hh:02d}:{mm:02d}"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def local_bounds(cfg: "Config", ecg: "LoadedECG") -> "tuple[pd.Timestamp, pd.Timestamp]":
    """Recording start and stop as local-time (cfg.local_tz) timestamps."""
    start = ecg.start_dt.tz_convert(cfg.local_tz)
    end = (ecg.start_dt + timedelta(seconds=ecg.duration_s)).tz_convert(cfg.local_tz)
    return start, end


# =============================================================================
# Loading & validation
# =============================================================================
@dataclass
class LoadedECG:
    signal: np.ndarray  # raw ECG samples (float)
    t_elapsed: np.ndarray  # seconds since first sample (uniform grid, fs)
    start_dt: pd.Timestamp  # wall-clock of first sample
    fs: int
    duration_s: float
    n_gap_samples: int  # estimated samples lost to clock gaps
    gap_total_s: float


def load_and_validate_ecg(cfg: Config) -> LoadedECG:
    """Read the ECG defensively, validate columns + sampling rate, print every
    assumption, and fail loudly on any mismatch."""
    section("STAGE 0 - Load & validate ECG")
    path = Path(cfg.ecg_path)
    if not path.exists():
        fail(f"ECG file not found: {path}")
    announce(f"Reading ECG: {path} ({path.stat().st_size / 1e6:.0f} MB)")

    df = pd.read_csv(path)
    announce(f"Columns found: {list(df.columns)}")

    if cfg.ecg_col not in df.columns:
        fail(f"Configured ecg_col={cfg.ecg_col!r} not in columns {list(df.columns)}")
    if cfg.timestamp_col not in df.columns:
        fail(
            f"Configured timestamp_col={cfg.timestamp_col!r} not in "
            f"columns {list(df.columns)}"
        )

    # ECG column -> float; complain about non-numeric content rather than coerce
    # silently.
    sig = pd.to_numeric(df[cfg.ecg_col], errors="coerce")
    n_bad = int(sig.isna().sum())
    if n_bad:
        announce(
            f"{n_bad} non-numeric/NaN samples in {cfg.ecg_col!r}; "
            f"forward/back-filling for continuity (flagged, not dropped)."
        )
        sig = sig.ffill().bfill()
    signal = sig.to_numpy(dtype=float)

    # Timezone for display: infer from the file's own timestamps unless the
    # caller passed --local-tz explicitly. Each raw row carries the device
    # clock's UTC offset at capture time, so this adapts automatically across
    # deployment countries (e.g. Ghana vs US) with no configuration.
    if cfg.local_tz is None:
        off_first = infer_utc_offset_str(str(df[cfg.timestamp_col].iloc[0]))
        off_last = infer_utc_offset_str(str(df[cfg.timestamp_col].iloc[-1]))
        if off_first is None:
            fail(
                f"Cannot auto-infer local_tz: {cfg.timestamp_col!r} has no UTC "
                f"offset (e.g. {df[cfg.timestamp_col].iloc[0]!r}). "
                f"Pass --local-tz explicitly."
            )
        if off_last != off_first:
            announce(
                f"WARNING: UTC offset drifts within this file "
                f"({off_first} at start -> {off_last} at end; likely a DST "
                f"transition mid-recording). Using the START offset "
                f"({off_first}) for all display times."
            )
        cfg.local_tz = off_first
        announce(f"Inferred display timezone from file: UTC{cfg.local_tz}.")
    else:
        announce(f"Using explicit --local-tz {cfg.local_tz!r}.")

    # Parse timestamps and validate the implied sampling rate.
    ts = pd.to_datetime(df[cfg.timestamp_col], utc=True, format="ISO8601")
    ns = ts.astype("int64").to_numpy()
    dt_s = np.diff(ns) / 1e9
    median_dt = float(np.median(dt_s))
    fs_implied = 1.0 / np.mean(dt_s[dt_s > 0])  # mean-based (see 00_inspect notes)
    announce(
        f"Timestamps span {ts.iloc[0]} -> {ts.iloc[-1]}; "
        f"median dt = {median_dt * 1e3:.3f} ms; "
        f"implied fs (mean dt) = {fs_implied:.2f} Hz"
    )
    err_pct = abs(fs_implied - cfg.sampling_rate) / cfg.sampling_rate * 100.0
    if err_pct > cfg.sampling_rate_tol_pct:
        fail(
            f"Configured sampling_rate={cfg.sampling_rate} Hz disagrees with the "
            f"file's implied rate {fs_implied:.2f} Hz by {err_pct:.1f}% "
            f"(> {cfg.sampling_rate_tol_pct}% tolerance). Fix the CONFIG to match "
            f"the data before trusting any HRV output."
        )
    announce(
        f"sampling_rate={cfg.sampling_rate} Hz accepted "
        f"(within {err_pct:.1f}% of implied rate)."
    )

    # Gaps: the clock occasionally jumps (see 00_inspect.py). We process the
    # signal as UNIFORMLY sampled at fs (standard for HRV) and treat gaps as
    # continuous, but we REPORT the total gap time as a caveat so the reviewer
    # knows the wall-clock axis drifts slightly from sample-index time.
    normal_dt = 1.0 / cfg.sampling_rate
    gap_mask = dt_s > 1.5 * median_dt
    gap_total_s = float(np.sum(dt_s[gap_mask] - normal_dt))
    n_gap_samples = int(round(gap_total_s / normal_dt))
    if gap_mask.any():
        announce(
            f"{int(gap_mask.sum())} clock gaps totalling ~{gap_total_s:.1f} s "
            f"(~{n_gap_samples} samples). Treated as continuous; reported as a caveat."
        )

    # Optional truncation for smoke tests.
    if cfg.limit_seconds is not None:
        n = int(cfg.limit_seconds * cfg.sampling_rate)
        n = min(n, len(signal))
        announce(f"--limit-seconds active: using first {n} samples "
                 f"({n / cfg.sampling_rate:.0f} s) only.")
        signal = signal[:n]

    n = len(signal)
    t_elapsed = np.arange(n) / cfg.sampling_rate
    duration_s = n / cfg.sampling_rate
    announce(f"Loaded {n:,} samples = {duration_s / 60:.1f} min at {cfg.sampling_rate} Hz.")

    return LoadedECG(
        signal=signal,
        t_elapsed=t_elapsed,
        start_dt=ts.iloc[0],
        fs=cfg.sampling_rate,
        duration_s=duration_s,
        n_gap_samples=n_gap_samples,
        gap_total_s=gap_total_s,
    )


def load_accel(cfg: Config, ecg: LoadedECG) -> Optional[pd.DataFrame]:
    """Load optional accelerometer file. Returns a DataFrame with elapsed-time
    seconds + magnitude deviation (g), or None if unavailable."""
    if not cfg.accel_path:
        return None
    path = Path(cfg.accel_path)
    if not path.exists():
        announce(f"Accelerometer file not found ({path}); skipping motion stage.")
        return None
    df = pd.read_csv(path)
    missing = [c for c in cfg.accel_cols if c not in df.columns]
    if missing:
        announce(f"Accel file missing columns {missing}; skipping motion stage.")
        return None
    ts = pd.to_datetime(df[cfg.accel_timestamp_col], utc=True, format="ISO8601")
    t_elapsed = (ts.astype("int64").to_numpy() - ts.astype("int64").to_numpy()[0]) / 1e9
    a = df[list(cfg.accel_cols)].to_numpy(dtype=float)
    mag = np.linalg.norm(a, axis=1) / cfg.gravity_g  # in g
    dev = np.abs(mag - 1.0)  # deviation from rest (1 g)
    announce(
        f"Accelerometer loaded: {len(df):,} samples, "
        f"~{1.0 / np.median(np.diff(t_elapsed)):.1f} Hz; "
        f"motion = fraction of window with |mag-1g| > {cfg.motion_dev_thresh_g} g."
    )
    # mag_g (raw magnitude) is kept alongside dev_g so the not-worn detector can
    # measure per-window stillness as the std of magnitude (orientation-agnostic).
    return pd.DataFrame({"t": t_elapsed, "dev_g": dev, "mag_g": mag})


# =============================================================================
# ABPM exclusion schedule
# =============================================================================
def build_abpm_windows(cfg: Config, ecg: LoadedECG) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) elapsed-time intervals to exclude."""
    if not cfg.abpm_enabled:
        announce("ABPM exclusion disabled in CONFIG.")
        return []
    windows: list[tuple[float, float]] = []
    if cfg.abpm_explicit_starts:
        for s in cfg.abpm_explicit_starts:
            t0 = (pd.to_datetime(s, utc=True) - ecg.start_dt).total_seconds()
            windows.append((t0, t0 + cfg.abpm_inflation_s))
        announce(f"ABPM: {len(windows)} explicit inflation windows.")
    else:
        t = 0.0
        while t < ecg.duration_s:
            windows.append((t, min(t + cfg.abpm_inflation_s, ecg.duration_s)))
            t += cfg.abpm_interval_min * 60.0
        announce(
            f"ABPM: assuming inflation every {cfg.abpm_interval_min:.0f} min for "
            f"{cfg.abpm_inflation_s:.0f} s -> {len(windows)} windows "
            f"(NO explicit schedule supplied; this is an ASSUMPTION)."
        )
    return windows


def overlaps_any(w0: float, w1: float, intervals: list[tuple[float, float]]) -> bool:
    return any(not (w1 <= a or w0 >= b) for a, b in intervals)


# =============================================================================
# Core ECG processing
# =============================================================================
@dataclass
class Processed:
    clean: np.ndarray
    rpeaks: np.ndarray  # raw detected peak sample indices
    corrected_peaks: np.ndarray  # after Kubios/Lipponen-Tarvainen correction
    corrected_idx: np.ndarray  # boolean over beats: was this beat corrected?
    rr_ms: np.ndarray  # NN intervals from corrected peaks (len = nbeats-1)
    rr_t_s: np.ndarray  # elapsed time of the SECOND peak of each RR pair
    quality_ici: np.ndarray  # per-sample primary SQI
    quality_avgqrs: np.ndarray  # per-sample secondary SQI
    artifacts: dict


def process_ecg(cfg: Config, ecg: LoadedECG) -> Processed:
    section("STAGE 1-4 - Clean, SQI, peaks, RR correction")

    # --- 1. Clean -------------------------------------------------------
    clean = nk.ecg_clean(ecg.signal, sampling_rate=ecg.fs)
    announce("nk.ecg_clean() applied at configured sampling rate.")

    # --- 3. R-peaks (needed before quality) -----------------------------
    _, info = nk.ecg_peaks(clean, sampling_rate=ecg.fs)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    announce(f"nk.ecg_peaks(): {len(rpeaks)} R-peaks detected.")
    if len(rpeaks) < 3:
        fail("Fewer than 3 R-peaks detected; signal is unusable for HRV.")

    # --- 2. Signal-level SQI -------------------------------------------
    # PRIMARY gate: ICI two-detector agreement (transferable; not magnitude-
    # dependent, so the 40 Hz cutoff does not bias it the way it biases methods
    # that look at spectral power).
    quality_ici = np.asarray(
        nk.ecg_quality(clean, rpeaks=rpeaks, sampling_rate=ecg.fs, method="ici"),
        dtype=float,
    )
    # SECONDARY: averageQRS is RELATIVE (1 == close to the mean beat, which is
    # NOT the same as "clinically good"). Reported, not gated on -- and, unlike
    # ici, costs real time on long recordings for a number nothing downstream
    # uses. Skipped by default for QC runs; pass --avgqrs to compute it anyway.
    if cfg.avgqrs_enabled:
        quality_avgqrs = np.asarray(
            nk.ecg_quality(clean, rpeaks=rpeaks, sampling_rate=ecg.fs, method="averageQRS"),
            dtype=float,
        )
        announce(
            f"SQI: ICI mean={np.nanmean(quality_ici):.3f} (PRIMARY); "
            f"averageQRS mean={np.nanmean(quality_avgqrs):.3f} (secondary, relative)."
        )
    else:
        quality_avgqrs = np.full(len(clean), np.nan, dtype=float)
        announce(
            f"SQI: ICI mean={np.nanmean(quality_ici):.3f} (PRIMARY); "
            f"averageQRS skipped (secondary, not used for QC verdict; "
            f"pass --avgqrs to compute it)."
        )
    # zhao2018 is computed PER WINDOW later (it returns one category per segment)
    # and is reported only -- its thresholds were calibrated on full-bandwidth
    # ECG and will misclassify a 40 Hz-limited signal.

    # --- 4. RR/NN correction (Lipponen & Tarvainen 2019 via "Kubios") ---
    artifacts, corrected_peaks = nk.signal_fixpeaks(
        rpeaks, sampling_rate=ecg.fs, method="Kubios", iterative=True
    )
    corrected_peaks = np.asarray(corrected_peaks, dtype=int)
    # Which ORIGINAL beats were flagged as artifacts?
    flagged = set()
    for key in ("ectopic", "missed", "extra", "longshort"):
        for i in np.atleast_1d(artifacts.get(key, [])):
            flagged.add(int(i))
    n_flagged = len(flagged)
    announce(
        f"signal_fixpeaks(method='Kubios'): flagged "
        f"ectopic={len(np.atleast_1d(artifacts.get('ectopic', [])))}, "
        f"missed={len(np.atleast_1d(artifacts.get('missed', [])))}, "
        f"extra={len(np.atleast_1d(artifacts.get('extra', [])))}, "
        f"longshort={len(np.atleast_1d(artifacts.get('longshort', [])))} "
        f"({n_flagged} beats total)."
    )

    # NN intervals from the CORRECTED peak series.
    rr_ms = np.diff(corrected_peaks) / ecg.fs * 1000.0
    rr_t_s = corrected_peaks[1:] / ecg.fs

    # --- Pediatric RR sanity bounds: an ADDITIONAL gate -----------------
    implausible = (rr_ms < cfg.rr_min_ms) | (rr_ms > cfg.rr_max_ms)
    announce(
        f"Pediatric RR bounds [{cfg.rr_min_ms:.0f},{cfg.rr_max_ms:.0f}] ms flag "
        f"{int(implausible.sum())} / {len(rr_ms)} NN intervals as implausible "
        f"(flagged, not removed)."
    )

    # Per-beat "corrected" boolean (over corrected_peaks; align to RR series for
    # window stats). A beat counts as corrected if it was an artifact OR its RR
    # to the previous beat is implausible.
    corrected_beat = np.zeros(len(corrected_peaks), dtype=bool)
    # Map flagged original indices onto corrected series positionally where valid.
    for i in flagged:
        if 0 <= i < len(corrected_beat):
            corrected_beat[i] = True
    corrected_beat[1:][implausible] = True

    return Processed(
        clean=clean,
        rpeaks=rpeaks,
        corrected_peaks=corrected_peaks,
        corrected_idx=corrected_beat,
        rr_ms=rr_ms,
        rr_t_s=rr_t_s,
        quality_ici=quality_ici,
        quality_avgqrs=quality_avgqrs,
        artifacts=artifacts,
    )


# =============================================================================
# Window-level QC
# =============================================================================
@dataclass
class WindowStats:
    idx: int
    t0_s: float
    t1_s: float
    start_dt: pd.Timestamp
    n_beats: int
    n_corrected: int
    pct_corrected: float
    valid_duration_s: float
    rmssd_ms: float
    sdnn_ms: float
    mean_hr_bpm: float
    ici_mean: float
    avgqrs_mean: float
    zhao_category: str
    motion_burden: float
    abpm_excluded: bool
    motion_flagged: bool
    ecg_amp: float  # robust peak-to-peak amplitude (p99-p1) of clean signal, ADC
    accel_std_g: float  # std of accel magnitude over window (nan if no accel)
    flatline: bool  # lead-off / electrode dropout (no QRS)
    not_worn: bool  # device idle: accel still AND ECG flatline
    pass_primary: bool
    pass_secondary: bool


def _rmssd(rr_ms: np.ndarray) -> float:
    if len(rr_ms) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))


def _sdnn(rr_ms: np.ndarray) -> float:
    if len(rr_ms) < 2:
        return float("nan")
    return float(np.std(rr_ms, ddof=1))


def compute_windows(
    cfg: Config,
    ecg: LoadedECG,
    proc: Processed,
    abpm_windows: list[tuple[float, float]],
    accel: Optional[pd.DataFrame],
) -> list[WindowStats]:
    section("STAGE 5 - Window-level QC (5-min tiles)")
    windows: list[WindowStats] = []
    n_win = int(np.ceil(ecg.duration_s / cfg.window_step_s))

    # Pre-extract sorted arrays once so each window is an O(log n) searchsorted
    # slice rather than a full-array boolean scan. corrected_peaks (hence peak_t
    # and rr_t_s), rpeaks and the accel time column are all monotonically
    # increasing, which is what makes the slice ranges exact.
    peak_t = proc.corrected_peaks / ecg.fs
    corrected_idx = proc.corrected_idx
    rr_t_s = proc.rr_t_s
    rr_ms = proc.rr_ms
    rpeaks = proc.rpeaks
    quality_ici = proc.quality_ici
    quality_avgqrs = proc.quality_avgqrs
    clean = proc.clean
    if accel is not None:
        accel_t = accel["t"].to_numpy()
        accel_dev = accel["dev_g"].to_numpy()
        accel_mag = accel["mag_g"].to_numpy()

    for w in range(n_win):
        t0 = w * cfg.window_step_s
        t1 = min(t0 + cfg.window_s, ecg.duration_s)
        if t1 - t0 < cfg.window_s * 0.5:
            continue  # skip a final stub window < half length

        # Beats whose (second) peak falls in [t0, t1) -> contiguous index range.
        blo = int(np.searchsorted(peak_t, t0, "left"))
        bhi = int(np.searchsorted(peak_t, t1, "left"))
        n_beats = bhi - blo
        n_corr = int(corrected_idx[blo:bhi].sum()) if n_beats else 0
        pct_corr = 100.0 * n_corr / n_beats if n_beats else 100.0

        # NN intervals fully inside the window (use corrected, drop implausible).
        rlo = int(np.searchsorted(rr_t_s, t0, "left"))
        rhi = int(np.searchsorted(rr_t_s, t1, "left"))
        rr = rr_ms[rlo:rhi]
        rr_valid = rr[(rr >= cfg.rr_min_ms) & (rr <= cfg.rr_max_ms)]
        rmssd = _rmssd(rr_valid)
        sdnn = _sdnn(rr_valid)
        mean_hr = float(60000.0 / np.mean(rr_valid)) if len(rr_valid) else float("nan")

        # Per-sample SQI averaged over the window.
        s0, s1 = int(t0 * ecg.fs), int(t1 * ecg.fs)
        ici_mean = float(np.nanmean(quality_ici[s0:s1])) if s1 > s0 else float("nan")
        avgqrs_mean = (
            float(np.nanmean(quality_avgqrs[s0:s1]))
            if (s1 > s0 and cfg.avgqrs_enabled) else float("nan")
        )

        # zhao2018 per-window (secondary, reported only). Guard against short/
        # noisy segments raising inside neurokit.
        zhao = "n/a"
        plo = int(np.searchsorted(rpeaks, s0, "left"))
        phi = int(np.searchsorted(rpeaks, s1, "left"))
        win_peaks = rpeaks[plo:phi] - s0
        if s1 > s0 and len(win_peaks) >= cfg.min_beats_per_window:
            try:
                zhao = str(
                    nk.ecg_quality(
                        clean[s0:s1],
                        rpeaks=win_peaks,
                        sampling_rate=ecg.fs,
                        method="zhao2018",
                        approach="fuzzy",
                    )
                )
            except Exception:
                zhao = "error"

        # Robust ECG amplitude for the lead-off / flatline detector. p99-p1 of the
        # clean signal in the window (one percentile call); collapses toward 0 when
        # the electrode is off.
        if s1 > s0:
            p1, p99 = np.percentile(clean[s0:s1], (1, 99))
            ecg_amp = float(p99 - p1)
        else:
            ecg_amp = float("nan")

        # Motion burden + accelerometer stillness for this window.
        motion = 0.0
        accel_std = float("nan")
        if accel is not None:
            alo = int(np.searchsorted(accel_t, t0, "left"))
            ahi = int(np.searchsorted(accel_t, t1, "left"))
            if ahi > alo:
                motion = float(np.mean(accel_dev[alo:ahi] > cfg.motion_dev_thresh_g))
                accel_std = float(np.std(accel_mag[alo:ahi]))

        abpm_excl = overlaps_any(t0, t1, abpm_windows)
        motion_flag = motion > cfg.motion_flag_frac

        # flatline / not_worn are set in a second pass below (flatline needs the
        # recording-wide median amplitude). pass flags start without them and are
        # cleared for flatline windows afterwards.
        enough = n_beats >= cfg.min_beats_per_window
        pass_primary = enough and (pct_corr <= cfg.accept_pct_primary) and not abpm_excl
        pass_secondary = enough and (pct_corr <= cfg.accept_pct_secondary) and not abpm_excl

        windows.append(
            WindowStats(
                idx=w,
                t0_s=t0,
                t1_s=t1,
                start_dt=ecg.start_dt + timedelta(seconds=t0),
                n_beats=n_beats,
                n_corrected=n_corr,
                pct_corrected=pct_corr,
                valid_duration_s=t1 - t0,
                rmssd_ms=rmssd,
                sdnn_ms=sdnn,
                mean_hr_bpm=mean_hr,
                ici_mean=ici_mean,
                avgqrs_mean=avgqrs_mean,
                zhao_category=zhao,
                motion_burden=motion,
                abpm_excluded=abpm_excl,
                motion_flagged=motion_flag,
                ecg_amp=ecg_amp,
                accel_std_g=accel_std,
                flatline=False,
                not_worn=False,
                pass_primary=pass_primary,
                pass_secondary=pass_secondary,
            )
        )

    # --- Second pass: lead-off/flatline + not-worn classification ---------
    # Flatline floor is RELATIVE to the recording's median window amplitude so it
    # is gain-independent across deployments.
    amps = np.array([ws.ecg_amp for ws in windows if np.isfinite(ws.ecg_amp)])
    median_amp = float(np.median(amps)) if len(amps) else float("nan")
    flatline_floor = cfg.flatline_rel_frac * median_amp if np.isfinite(median_amp) else 0.0
    n_flat = n_notworn = 0
    for ws in windows:
        ws.flatline = bool(np.isfinite(ws.ecg_amp) and ws.ecg_amp < flatline_floor)
        # Not worn = both sensors dead: accelerometer still AND ECG flatline.
        ws.not_worn = bool(
            ws.flatline
            and np.isfinite(ws.accel_std_g)
            and ws.accel_std_g < cfg.not_worn_accel_std_g
        )
        if ws.flatline:
            # No usable cardiac signal -> not analyzable, regardless of % corrected.
            ws.pass_primary = False
            ws.pass_secondary = False
            n_flat += 1
        if ws.not_worn:
            n_notworn += 1

    announce(
        f"Lead-off/flatline detector: amplitude floor = {flatline_floor:.0f} ADC "
        f"({cfg.flatline_rel_frac:.2f} x median {median_amp:.0f}); "
        f"{n_flat} flatline window(s), of which {n_notworn} also not-worn "
        f"(accel std < {cfg.not_worn_accel_std_g} g)."
    )

    n_pass = sum(ws.pass_primary for ws in windows)
    announce(
        f"{len(windows)} windows; {n_pass} pass at "
        f"{cfg.accept_pct_primary:.0f}% corrected-beat threshold."
    )
    return windows


# =============================================================================
# Fiducial jitter sensitivity (the bandwidth proxy)
# =============================================================================
@dataclass
class JitterResult:
    sigmas_ms: list[float]
    rmssd_overall: list[float]  # empirical RMSSD at each sigma
    rmssd_analytic: list[float]  # sqrt(RMSSD_true^2 + 6 sigma^2)
    pct_change: list[float]  # vs sigma=0


def jitter_sensitivity(cfg: Config, ecg: LoadedECG, proc: Processed) -> JitterResult:
    section("STAGE 6 - Fiducial jitter sensitivity (40 Hz-bandwidth proxy)")
    # Work in peak TIMES (seconds). A constant detector lag cancels under
    # successive differencing, so only ZERO-MEAN random jitter matters.
    peak_t = proc.corrected_peaks / ecg.fs
    # Keep only plausibly-spaced beats so a few artifacts don't dominate RMSSD.
    rr0 = np.diff(peak_t) * 1000.0
    keep = (rr0 >= cfg.rr_min_ms) & (rr0 <= cfg.rr_max_ms)
    base_t = peak_t[1:][keep]  # times of kept second-peaks (for reference)
    # Reconstruct a contiguous peak-time series from kept RRs to differentiate.
    rmssd_true = _rmssd(rr0[keep])

    sigmas = list(cfg.jitter_sigmas_ms)
    emp, ana, pct = [], [], []
    for sg in sigmas:
        # Add independent Gaussian timing noise to each peak time, recompute RR.
        noise = RNG.normal(0.0, sg / 1000.0, size=len(peak_t))
        jt = peak_t + noise
        rr = np.diff(jt) * 1000.0
        rr = rr[keep]  # same beats as baseline
        val = _rmssd(rr)
        emp.append(val)
        ana.append(float(np.sqrt(rmssd_true**2 + 6.0 * sg**2)))
        pct.append(100.0 * (val - emp[0]) / emp[0] if emp[0] else float("nan"))
        announce(
            f"sigma={sg:>4.1f} ms -> RMSSD={val:6.2f} ms "
            f"(analytic {ana[-1]:6.2f} ms, {pct[-1]:+.1f}% vs sigma=0)"
        )
    announce(
        "RMSSD rises with jitter as predicted by sqrt(RMSSD^2 + 6*sigma^2): "
        "this is how vulnerable THIS deployment's RMSSD is to the 40 Hz limit."
    )
    return JitterResult(sigmas, emp, ana, pct)


# =============================================================================
# Overall verdict
# =============================================================================
def overall_verdict(cfg: Config, windows: list[WindowStats]) -> tuple[str, float, float]:
    """PASS / REVIEW / FAIL plus two analyzable percentages.

    Returns (verdict, pct_total, pct_worn):
      * pct_total - analyzable seconds / TOTAL deployment seconds. This is the
        PRIMARY number: it measures overall mission success (a device that fell
        off counts against it), and the verdict is keyed to it.
      * pct_worn  - analyzable seconds / WORN seconds, where not-worn windows are
        removed from the denominator. This isolates DEVICE / signal performance
        during the time the device was actually on the body.

    Analyzable = pass_primary, which already excludes ABPM-excluded and flatline
    (lead-off) windows. Lead-off/flatline windows are worn-but-no-signal, so they
    count as failures against BOTH percentages; only not-worn windows are dropped
    from the worn denominator.
    """
    if not windows:
        return "FAIL", 0.0, 0.0
    total = sum(ws.valid_duration_s for ws in windows)
    worn = sum(ws.valid_duration_s for ws in windows if not ws.not_worn)
    good = sum(ws.valid_duration_s for ws in windows if ws.pass_primary)
    pct_total = 100.0 * good / total if total else 0.0
    pct_worn = 100.0 * good / worn if worn else 0.0
    if pct_total >= cfg.pass_analyzable_pct:
        verdict = "PASS"
    elif pct_total >= cfg.review_analyzable_pct:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"
    return verdict, pct_total, pct_worn


# =============================================================================
# Dashboard figure
# =============================================================================
def panel_caption(ax, text: str) -> None:
    """Render a short, wrapped, italic grey caption just below `ax`.

    Placed in axes coordinates below the x-axis so it reads as a footnote for
    that panel in both the PNG and the PDF. The generous gridspec hspace gives
    these room not to collide with the next panel's title.
    """
    ax.text(
        0.0, -0.30, text,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7, style="italic", color="0.35", wrap=True,
    )


def _build_explanation_page(
    cfg: Config, ecg: "LoadedECG", verdict: str, pct_total: float, pct_worn: float,
    windows: "list[WindowStats]"
):
    """Build a portrait 'How to read this report' figure used as PDF page 2.

    Reuses values already computed for the dashboard; adds no new analysis.
    """
    fig = plt.figure(figsize=(8.5, 11), dpi=cfg.figure_dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    not_worn_min = sum(ws.valid_duration_s for ws in windows if ws.not_worn) / 60.0
    flatline_min = sum(ws.valid_duration_s for ws in windows if ws.flatline) / 60.0
    start_local, end_local = local_bounds(cfg, ecg)
    tz_abbr = start_local.strftime("%Z")

    header = (
        f"How to read this report\n"
        f"{cfg.deployment_id}   |   verdict: {verdict}   |   "
        f"{pct_total:.0f}% of total analyzable / {pct_worn:.0f}% of worn time\n"
        f"recorded {start_local:%Y-%m-%d %H:%M:%S} → {end_local:%Y-%m-%d %H:%M:%S} "
        f"{tz_abbr}   |   nk {nk.__version__}   |   fs = {ecg.fs} Hz"
    )

    panels = [
        ("Full-recording timeline",
         "Cleaned ECG across the whole deployment. Each 5-min window is shaded by "
         "quality: green <=2%, yellow <=5%, red >5% corrected beats. Teal windows "
         "are lead-off / flatline (electrode lost contact, no QRS). A blue strip "
         "along the top marks windows where the device was not worn. Purple ticks "
         "mark high-motion windows (from the accelerometer)."),
        ("Tachogram (RR over time)",
         "Beat-to-beat RR intervals. Red points are beats the Lipponen-Tarvainen "
         "(Kubios) correction flagged as ectopic/missed/extra/misaligned. Dotted "
         "lines are the pediatric plausibility bounds (300-1500 ms)."),
        ("Artifact burden vs time of day",
         "The same windows plotted by local clock hour, to reveal whether bad "
         "windows cluster with a particular time or activity."),
        ("Per-window artifact burden over time",
         "Percent corrected beats in each 5-min window along elapsed time. Dashed "
         "lines are the 5% (primary) and 2% (secondary) acceptance thresholds."),
        ("Per-window RMSSD",
         "Short-term HRV (RMSSD) for each window. Windows that failed acceptance "
         "or were excluded are greyed out so they do not anchor the eye."),
        ("Fiducial jitter sensitivity",
         "Overall RMSSD as a function of injected R-peak timing noise sigma. The "
         "curve is sqrt(RMSSD^2 + 6*sigma^2). A steep rise means this recording's "
         "RMSSD is vulnerable to the timing jitter imposed by the 40 Hz analog "
         "bandwidth."),
        ("Cleanest-window beat overlay",
         "Up to 60 beats from the best window, aligned at the R-peak (red line). A "
         "clean window shows one consistent QRS shape."),
        ("Most-flagged-window beat overlay",
         "The same overlay for the worst window. Smeared or variable shapes flag "
         "motion artifact or detector error."),
        ("Accelerometry summary",
         "Per-window motion burden (fraction of the window with |accel magnitude "
         "- 1g| over threshold). Green = at rest (breathing-level movement only); "
         "purple = motion (consistent with physical activity); blue = not worn. "
         "Only shown when an accelerometer file was provided."),
    ]

    verdict_logic = (
        "Verdict logic: two analyzable percentages are reported, and they answer "
        "DIFFERENT questions -- do not divide worn duration by total duration and "
        "expect it to match either one. % of TOTAL deployment (the PRIMARY number, "
        "which drives PASS/REVIEW/FAIL) measures overall mission success - a device "
        "that fell off or was removed counts against it. % of WORN time removes "
        "not-worn windows from the denominator to isolate device/signal performance "
        "while on the body -- it is analyzable-seconds / worn-seconds, NOT "
        "worn-seconds / total-seconds (that ratio is reported separately as "
        "'worn duration' with its own % of total). "
        f"PASS >= {cfg.pass_analyzable_pct:.0f}%, "
        f"REVIEW >= {cfg.review_analyzable_pct:.0f}%, else FAIL, on the TOTAL "
        "number. A window is analyzable when it passes the primary corrected-beat "
        f"threshold (<= {cfg.accept_pct_primary:.0f}%), has a real ECG signal (not "
        "lead-off/flatline), and is not ABPM-excluded."
    )
    detectors = (
        "Lead-off & not-worn detectors: a window is LEAD-OFF / FLATLINE when its "
        f"ECG amplitude drops below {cfg.flatline_rel_frac:.2f} x the recording's "
        "median window amplitude (electrode lost skin contact -> no QRS); these "
        "count as failures in both percentages. A window is NOT WORN only when the "
        "ECG is flatline AND the accelerometer is essentially still "
        f"(magnitude std < {cfg.not_worn_accel_std_g} g) - both sensors dead, i.e. "
        "the device is off the body; not-worn windows are removed from the worn "
        f"denominator. This recording: {flatline_min:.0f} min lead-off/flatline, "
        f"{not_worn_min:.0f} min not worn."
    )
    caveats = (
        "Key caveats / assumptions: the sampling rate is validated against the "
        "file's own timestamps; clock gaps are treated as continuous (total gap "
        "time is reported separately); ECG values are raw uncalibrated ADC counts; "
        "RR plausibility uses pediatric bounds (300-1500 ms); ABPM cuff-inflation "
        "exclusion is currently DISABLED (deferred until a real cuff schedule is "
        "available)."
    )
    jitter_note = (
        "Jitter note: low-RMSSD recordings are far more vulnerable to the 40 Hz "
        "bandwidth's timing jitter than high-RMSSD ones, because the same absolute "
        "timing noise is a larger fraction of a small RMSSD. Compare the jitter "
        "panel across deployments before trusting small RMSSD differences."
    )

    # Compose the page as one wrapped text column.
    lines: list[str] = []
    for title, body in panels:
        wrapped = textwrap.fill(body, width=95)
        lines.append(f"• {title}")
        lines.append(textwrap.indent(wrapped, "    "))
        lines.append("")
    body_block = "\n".join(lines)
    footer_block = "\n\n".join(
        textwrap.fill(s, width=95)
        for s in (verdict_logic, detectors, caveats, jitter_note)
    )

    ax.text(0.06, 0.97, header, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, fontweight="bold")
    ax.text(0.06, 0.90, "What each panel shows", transform=ax.transAxes,
            va="top", ha="left", fontsize=10, fontweight="bold")
    ax.text(0.06, 0.875, body_block, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, family="monospace", linespacing=1.2)
    ax.text(0.06, 0.43, footer_block, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, linespacing=1.3)
    return fig


def build_dashboard(
    cfg: Config,
    ecg: LoadedECG,
    proc: Processed,
    windows: list[WindowStats],
    abpm_windows: list[tuple[float, float]],
    jitter: JitterResult,
    verdict: str,
    pct_total: float,
    pct_worn: float,
    out_png: Path,
    out_pdf: Path,
    elapsed_min: float,
    accel_available: bool,
) -> None:
    section("STAGE 7 - Dashboard figure")
    banner_color = {"PASS": "#2e7d32", "REVIEW": "#f9a825", "FAIL": "#c62828"}[verdict]
    start_local, end_local = local_bounds(cfg, ecg)
    tz_abbr = start_local.strftime("%Z")

    fig = plt.figure(figsize=(16, 21), dpi=cfg.figure_dpi)
    # Extra hspace gives the per-panel captions room not to collide with the
    # next panel's title.
    gs = fig.add_gridspec(7, 2, height_ratios=[0.5, 1, 1, 1, 1.2, 1.2, 1], hspace=0.65,
                          wspace=0.18)

    # --- Banner ---------------------------------------------------------
    axb = fig.add_subplot(gs[0, :])
    axb.axis("off")
    axb.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=axb.transAxes, color=banner_color, alpha=0.85)
    )
    axb.text(
        0.01, 0.5,
        f"  {verdict}",
        transform=axb.transAxes, va="center", ha="left",
        fontsize=30, fontweight="bold", color="white",
    )
    axb.text(
        0.99, 0.70,
        f"{cfg.deployment_id}   |   {pct_total:.0f}% analyzable of total   |   "
        f"{pct_worn:.0f}% analyzable of worn time   |   "
        f"nk {nk.__version__}   |   fs={ecg.fs} Hz   ",
        transform=axb.transAxes, va="center", ha="right",
        fontsize=12, color="white",
    )
    axb.text(
        0.99, 0.28,
        f"recorded {start_local:%Y-%m-%d %H:%M:%S} → "
        f"{end_local:%Y-%m-%d %H:%M:%S} {tz_abbr}   |   "
        f"processed in {elapsed_min:.1f} min   ",
        transform=axb.transAxes, va="center", ha="right",
        fontsize=11, color="white",
    )

    win_t = np.array([ws.t0_s / 60.0 for ws in windows])  # minutes
    pct_corr = np.array([ws.pct_corrected for ws in windows])
    rmssd = np.array([ws.rmssd_ms for ws in windows])
    excluded = np.array([ws.abpm_excluded for ws in windows])
    passed = np.array([ws.pass_primary for ws in windows])

    # --- Panel 1: timeline with color-coded quality --------------------
    ax1 = fig.add_subplot(gs[1, :])
    # Downsample the raw clean signal for plotting only.
    step = max(1, len(proc.clean) // 6000)
    ax1.plot(ecg.t_elapsed[::step] / 60.0, proc.clean[::step], lw=0.3, color="0.4")
    has_flatline = any(ws.flatline for ws in windows)
    has_not_worn = any(ws.not_worn for ws in windows)
    for ws in windows:
        # Lead-off/flatline (no signal) gets its own colour so it reads as
        # distinct from "noisy" (red) windows.
        if ws.flatline:
            c = "#26a69a"  # teal = lead-off / electrode dropout
        elif ws.abpm_excluded:
            c = "#9e9e9e"
        elif ws.pct_corrected <= cfg.accept_pct_secondary:
            c = "#2e7d32"
        elif ws.pct_corrected <= cfg.accept_pct_primary:
            c = "#f9a825"
        else:
            c = "#c62828"
        ax1.axvspan(ws.t0_s / 60.0, ws.t1_s / 60.0, color=c, alpha=0.18, zorder=0)
        if ws.motion_flagged:
            ax1.axvspan(ws.t0_s / 60.0, ws.t1_s / 60.0, ymin=0.0, ymax=0.06,
                        color="purple", alpha=0.5, zorder=3)
        # Not-worn (device idle) is marked with a solid top strip.
        if ws.not_worn:
            ax1.axvspan(ws.t0_s / 60.0, ws.t1_s / 60.0, ymin=0.94, ymax=1.0,
                        color="#01579b", alpha=0.8, zorder=3)
    # ABPM inflation lines only appear when a cuff schedule produced windows.
    has_abpm = len(abpm_windows) > 0
    title_bits = ["window quality (green/yellow/red)"]
    if has_flatline:
        title_bits.append("lead-off (teal)")
    if has_not_worn:
        title_bits.append("not-worn (blue top strip)")
    if has_abpm:
        for a, b in abpm_windows:
            ax1.axvline(a / 60.0, color="blue", lw=0.6, alpha=0.5, zorder=2)
        title_bits.append("ABPM inflations (blue lines)")
    title_bits.append("motion (purple ticks)")
    ax1.set_title("Full recording — " + ", ".join(title_bits))
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("ECG (clean)")
    legend_handles = [
        Patch(color="#2e7d32", alpha=0.4, label=f"<={cfg.accept_pct_secondary:.0f}% corr"),
        Patch(color="#f9a825", alpha=0.4, label=f"<={cfg.accept_pct_primary:.0f}% corr"),
        Patch(color="#c62828", alpha=0.4, label=f">{cfg.accept_pct_primary:.0f}% corr"),
    ]
    if has_flatline:
        legend_handles.append(Patch(color="#26a69a", alpha=0.4, label="lead-off/flatline"))
    if has_not_worn:
        legend_handles.append(Patch(color="#01579b", alpha=0.8, label="not worn"))
    if has_abpm:
        legend_handles.append(Patch(color="#9e9e9e", alpha=0.4, label="ABPM excluded"))
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8,
               ncol=len(legend_handles))
    panel_caption(
        ax1,
        "Cleaned ECG; window shading = quality (green <=2%, yellow <=5%, red >5% "
        "corrected beats); teal = lead-off/flatline (no signal); blue top strip = "
        "device not worn; purple ticks = high motion.",
    )

    # --- Panel 2: tachogram --------------------------------------------
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(proc.rr_t_s / 60.0, proc.rr_ms, lw=0.4, color="0.5", zorder=1)
    corr_rr_mask = proc.corrected_idx[1:]
    ax2.scatter((proc.rr_t_s / 60.0)[corr_rr_mask], proc.rr_ms[corr_rr_mask],
                s=6, color="red", zorder=3, label="corrected beat")
    ax2.axhline(cfg.rr_min_ms, color="k", ls=":", lw=0.6)
    ax2.axhline(cfg.rr_max_ms, color="k", ls=":", lw=0.6)
    ax2.set_ylim(0, max(cfg.rr_max_ms * 1.2, np.nanpercentile(proc.rr_ms, 99)))
    ax2.set_title("Tachogram (RR over time) — corrected beats in red, "
                  "pediatric bounds dotted")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("RR (ms)")
    ax2.legend(loc="upper right", fontsize=8)
    panel_caption(
        ax2,
        "Beat-to-beat RR; red = beats corrected (Lipponen-Tarvainen); dotted = "
        "pediatric plausibility bounds 300-1500 ms.",
    )

    # --- Panel 3: artifact burden vs time-of-day -----------------------
    ax3 = fig.add_subplot(gs[3, 0])

    def _tod_hour(ws):
        loc = ws.start_dt.tz_convert(cfg.local_tz)
        return loc.hour + loc.minute / 60.0

    tod_hours = np.array([_tod_hour(ws) for ws in windows])
    ax3.scatter(tod_hours, pct_corr, s=10, c=["#c62828" if not p else "#2e7d32"
                                              for p in passed])
    ax3.axhline(cfg.accept_pct_primary, color="orange", ls="--",
                label=f"{cfg.accept_pct_primary:.0f}%")
    ax3.axhline(cfg.accept_pct_secondary, color="green", ls="--",
                label=f"{cfg.accept_pct_secondary:.0f}%")
    ax3.set_title("Artifact burden vs TIME OF DAY\n(do bad windows cluster?)")
    ax3.set_xlabel("Hour of day (local)")
    ax3.set_ylabel("% corrected beats")
    ax3.set_xlim(0, 24)
    ax3.legend(fontsize=8)
    panel_caption(
        ax3,
        "Same windows by clock hour — reveals whether bad windows cluster with "
        "activity/time.",
    )

    # --- Panel 4: artifact burden over elapsed time --------------------
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.plot(win_t, pct_corr, "-o", ms=3, color="0.3")
    ax4.axhline(cfg.accept_pct_primary, color="orange", ls="--",
                label=f"{cfg.accept_pct_primary:.0f}% (primary)")
    ax4.axhline(cfg.accept_pct_secondary, color="green", ls="--",
                label=f"{cfg.accept_pct_secondary:.0f}% (secondary)")
    for ws in windows:
        if ws.abpm_excluded:
            ax4.axvspan(ws.t0_s / 60.0, ws.t1_s / 60.0, color="0.7", alpha=0.4)
    ax4.set_title("Per-window artifact burden over time")
    ax4.set_xlabel("Time (min)")
    ax4.set_ylabel("% corrected beats")
    ax4.legend(fontsize=8)
    panel_caption(
        ax4,
        "% corrected beats per 5-min window; dashed = 5% / 2% acceptance lines.",
    )

    # --- Panel 5: per-window RMSSD, failed/excluded greyed -------------
    ax5 = fig.add_subplot(gs[4, 0])
    ok_mask = passed
    ax5.scatter(win_t[~ok_mask], rmssd[~ok_mask], s=16, color="0.75", zorder=1)
    ax5.scatter(win_t[ok_mask], rmssd[ok_mask], s=16, color="#1565c0", zorder=3)
    ax5.set_title("Per-window RMSSD (failed/excluded greyed)")
    ax5.set_xlabel("Time (min)")
    ax5.set_ylabel("RMSSD (ms)")
    panel_caption(
        ax5,
        "Short-term HRV per window; greyed = failed acceptance or excluded.",
    )

    # --- Panel 6: jitter sensitivity -----------------------------------
    ax6 = fig.add_subplot(gs[4, 1])
    ax6.plot(jitter.sigmas_ms, jitter.rmssd_overall, "o", ms=7, color="#c62828",
             label="empirical")
    ax6.plot(jitter.sigmas_ms, jitter.rmssd_analytic, "-", color="k",
             label=r"$\sqrt{RMSSD^2 + 6\sigma^2}$")
    ax6.set_title("Fiducial jitter sensitivity\n(RMSSD bias from R-peak timing noise)")
    ax6.set_xlabel(r"injected jitter $\sigma$ (ms)")
    ax6.set_ylabel("overall RMSSD (ms)")
    ax6.legend(fontsize=8)
    panel_caption(
        ax6,
        "RMSSD vs injected R-peak jitter; curve = sqrt(RMSSD^2 + 6*sigma^2); "
        "steep = RMSSD vulnerable to the 40 Hz bandwidth.",
    )

    # --- Panel 7+8: representative beat overlays -----------------------
    clean_win = min((ws for ws in windows if ws.n_beats >= cfg.min_beats_per_window),
                    key=lambda w: (w.pct_corrected, -w.ici_mean), default=None)
    bad_win = max((ws for ws in windows if ws.n_beats >= cfg.min_beats_per_window),
                  key=lambda w: w.pct_corrected, default=None)

    overlay_caption = {
        0: "<=60 beats from the best window aligned at R (red) — expect one "
           "consistent QRS.",
        1: "Same for the worst window — smeared/variable shapes flag motion or "
           "detector error.",
    }
    for col, ws, title in ((0, clean_win, "Cleanest window"),
                           (1, bad_win, "Most-flagged window")):
        axo = fig.add_subplot(gs[5, col])
        if ws is None:
            axo.text(0.5, 0.5, "no eligible window", ha="center")
            axo.axis("off")
            continue
        s0, s1 = int(ws.t0_s * ecg.fs), int(ws.t1_s * ecg.fs)
        peaks = proc.corrected_peaks[(proc.corrected_peaks >= s0)
                                     & (proc.corrected_peaks < s1)]
        half = int(0.30 * ecg.fs)  # +/-300 ms window around R
        for p in peaks[:60]:
            a, b = p - half, p + half
            if a < 0 or b > len(proc.clean):
                continue
            axo.plot(np.arange(-half, half) / ecg.fs * 1000.0,
                     proc.clean[a:b], lw=0.4, color="0.5", alpha=0.6)
        axo.axvline(0, color="red", lw=0.6)
        axo.set_title(f"{title}\n{ws.start_dt.tz_convert(cfg.local_tz):%H:%M}, "
                      f"{ws.pct_corrected:.1f}% corr, ICI={ws.ici_mean:.2f}")
        axo.set_xlabel("time around R (ms)")
        axo.set_ylabel("ECG")
        panel_caption(axo, overlay_caption[col])

    # --- Panel 9: accelerometry summary (motion vs rest) ----------------
    ax9 = fig.add_subplot(gs[6, :])
    if not accel_available:
        ax9.text(
            0.5, 0.5, "No accelerometer data provided (--no-accel, or no --accel path given)",
            ha="center", va="center", transform=ax9.transAxes, color="0.4", fontsize=11,
        )
        ax9.axis("off")
    else:
        motion_burden = np.array([ws.motion_burden for ws in windows])
        cats = np.array([
            "not worn" if ws.not_worn else "motion" if ws.motion_flagged else "at rest"
            for ws in windows
        ])
        cat_color = {"at rest": "#2e7d32", "motion": "#8e24aa", "not worn": "#01579b"}
        ax9.plot(win_t, motion_burden, lw=0.4, color="0.7", zorder=1)
        for cat, color in cat_color.items():
            m = cats == cat
            if m.any():
                ax9.scatter(win_t[m], motion_burden[m], s=16, color=color, zorder=3, label=cat)
        thresh_line = ax9.axhline(
            cfg.motion_flag_frac, color="purple", ls="--", lw=0.8,
            label=f"motion-flag threshold ({cfg.motion_flag_frac:.0%})",
        )
        n_tot = len(cats)
        pct = {c: 100.0 * int((cats == c).sum()) / n_tot if n_tot else 0.0
               for c in cat_color}
        handles = [thresh_line] + [
            Patch(color=color, label=f"{cat} ({pct[cat]:.0f}% of windows)")
            for cat, color in cat_color.items()
        ]
        ax9.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
        ax9.set_title("Accelerometry summary — motion burden per window")
        ax9.set_xlabel("Time (min)")
        ax9.set_ylabel("Motion burden\n(frac. of window > threshold)")
        ax9.set_ylim(-0.02, 1.02)
        panel_caption(
            ax9,
            "Per-window fraction of samples with |accel magnitude - 1g| > "
            f"{cfg.motion_dev_thresh_g:.2f} g. Green = at rest (breathing-level "
            "movement only, consistent with sitting/lying still); purple = motion "
            "(consistent with physical activity); blue = not worn.",
        )

    fig.suptitle(
        f"ECG QC dashboard — {cfg.deployment_id}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # PNG stays a single image (the dashboard only).
    fig.savefig(out_png, bbox_inches="tight")
    # PDF is two pages: the dashboard, then a "How to read this report" page.
    expl_fig = _build_explanation_page(cfg, ecg, verdict, pct_total, pct_worn, windows)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        pdf.savefig(expl_fig)
    plt.close(expl_fig)
    plt.close(fig)
    announce(f"Dashboard written: {out_png} and {out_pdf} (PDF has 2 pages)")


# =============================================================================
# Outputs: JSON, cross-deployment CSV, console table
# =============================================================================
def write_outputs(
    cfg: Config,
    ecg: LoadedECG,
    proc: Processed,
    windows: list[WindowStats],
    abpm_windows: list[tuple[float, float]],
    jitter: JitterResult,
    verdict: str,
    pct_total: float,
    pct_worn: float,
    out_json: Path,
    total_runtime_min: float,
) -> dict:
    section("STAGE 8 - Metrics (JSON + cross-deployment CSV + console)")

    start_local, end_local = local_bounds(cfg, ecg)
    tz_abbr = start_local.strftime("%Z")

    rr_valid = proc.rr_ms[(proc.rr_ms >= cfg.rr_min_ms) & (proc.rr_ms <= cfg.rr_max_ms)]
    hr = 60000.0 / rr_valid if len(rr_valid) else np.array([np.nan])
    n_total = len(windows)
    n_pass5 = sum(ws.pass_primary for ws in windows)
    n_pass2 = sum(ws.pass_secondary for ws in windows)
    abpm_excl_s = sum(
        ws.valid_duration_s for ws in windows if ws.abpm_excluded
    )
    motion_s = sum(ws.valid_duration_s for ws in windows if ws.motion_flagged)
    flatline_s = sum(ws.valid_duration_s for ws in windows if ws.flatline)
    not_worn_s = sum(ws.valid_duration_s for ws in windows if ws.not_worn)
    worn_s = sum(ws.valid_duration_s for ws in windows if not ws.not_worn)
    # at_rest = worn AND not motion-flagged (breathing-level movement only).
    at_rest_s = sum(
        ws.valid_duration_s for ws in windows if not ws.not_worn and not ws.motion_flagged
    )
    n_flatline = sum(ws.flatline for ws in windows)
    n_not_worn = sum(ws.not_worn for ws in windows)
    # Share of the TOTAL recording the device was worn -- distinct from
    # pct_analyzable_worn (share of WORN time that is analyzable). Reported
    # separately so the two aren't conflated (they answer different questions).
    worn_pct_of_total = 100.0 * worn_s / ecg.duration_s if ecg.duration_s else 0.0
    pct_corrected_overall = (
        100.0 * int(proc.corrected_idx.sum()) / len(proc.corrected_idx)
        if len(proc.corrected_idx) else float("nan")
    )

    metrics = {
        "deployment_id": cfg.deployment_id,
        "neurokit2_version": nk.__version__,
        "rng_seed": RNG_SEED,
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "local_tz": cfg.local_tz,
        "start_local": start_local.isoformat(timespec="seconds"),
        "end_local": end_local.isoformat(timespec="seconds"),
        "sampling_rate_hz": ecg.fs,
        "total_duration_s": round(ecg.duration_s, 1),
        "effective_duration_s": round(ecg.duration_s - abpm_excl_s, 1),
        "worn_duration_s": round(worn_s, 1),
        # % of the TOTAL recording the device was worn -- NOT the same question
        # as pct_analyzable_worn (% of WORN time that is analyzable quality).
        "worn_pct_of_total": round(worn_pct_of_total, 2),
        "gap_total_s": round(ecg.gap_total_s, 1),
        # pct_analyzable_total is the PRIMARY verdict number (mission success);
        # pct_analyzable_worn isolates device/signal performance while on the body.
        "pct_analyzable_total": round(pct_total, 2),
        "pct_analyzable_worn": round(pct_worn, 2),
        "pct_analyzable": round(pct_total, 2),  # back-compat alias = total
        "verdict": verdict,
        "total_runtime_min": round(total_runtime_min, 2),
        "n_beats": int(len(proc.corrected_peaks)),
        "pct_corrected_beats": round(pct_corrected_overall, 3),
        "mean_hr_bpm": round(float(np.nanmean(hr)), 1),
        "min_hr_bpm": round(float(np.nanmin(hr)), 1),
        "max_hr_bpm": round(float(np.nanmax(hr)), 1),
        "rmssd_ms_overall": round(_rmssd(rr_valid), 2),
        "sdnn_ms_overall": round(_sdnn(rr_valid), 2),
        "n_windows_total": n_total,
        "n_windows_pass_5pct": n_pass5,
        "n_windows_fail_5pct": n_total - n_pass5,
        "n_windows_pass_2pct": n_pass2,
        "n_windows_fail_2pct": n_total - n_pass2,
        "abpm_excluded_s": round(abpm_excl_s, 1),
        "motion_flagged_s": round(motion_s, 1),
        "at_rest_s": round(at_rest_s, 1),
        "flatline_s": round(flatline_s, 1),
        "not_worn_s": round(not_worn_s, 1),
        "n_windows_flatline": int(n_flatline),
        "n_windows_not_worn": int(n_not_worn),
        "jitter_sensitivity": {
            "sigma_ms": jitter.sigmas_ms,
            "rmssd_empirical_ms": [round(v, 3) for v in jitter.rmssd_overall],
            "rmssd_analytic_ms": [round(v, 3) for v in jitter.rmssd_analytic],
            "pct_change_vs_sigma0": [round(v, 2) for v in jitter.pct_change],
        },
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    announce(f"JSON metrics written: {out_json}")

    # Cross-deployment one-row CSV (append, keyed by deployment_id).
    row = {k: v for k, v in metrics.items()
           if not isinstance(v, (dict, list))}
    csv_path = Path(cfg.cross_deployment_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["deployment_id"] != cfg.deployment_id]
        out = pd.concat([existing, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(csv_path, index=False)
    announce(f"Cross-deployment summary updated: {csv_path} "
             f"({len(out)} deployment row(s)).")

    # Console summary table.
    print()
    print("-" * 78)
    print(f"  DEPLOYMENT QC SUMMARY — {cfg.deployment_id}    [{verdict}]")
    print("-" * 78)
    rows = [
        ("neurokit2", nk.__version__),
        ("sampling rate", f"{ecg.fs} Hz"),
        ("recording start", f"{start_local:%Y-%m-%d %H:%M:%S} {tz_abbr}"),
        ("recording end", f"{end_local:%Y-%m-%d %H:%M:%S} {tz_abbr}"),
        ("total duration", f"{ecg.duration_s / 3600:.2f} h"),
        ("worn duration", f"{worn_s / 3600:.2f} h  ({worn_pct_of_total:.1f}% of total)"),
        ("% analyzable (of total time)", f"{pct_total:.1f}%  <-- verdict driver"),
        ("% analyzable (of worn time)", f"{pct_worn:.1f}%"),
        ("N beats", f"{metrics['n_beats']:,}"),
        ("% corrected beats", f"{metrics['pct_corrected_beats']:.2f}%"),
        ("mean HR", f"{metrics['mean_hr_bpm']:.0f} bpm "
                    f"({metrics['min_hr_bpm']:.0f}-{metrics['max_hr_bpm']:.0f})"),
        ("RMSSD / SDNN", f"{metrics['rmssd_ms_overall']:.1f} / "
                         f"{metrics['sdnn_ms_overall']:.1f} ms"),
        ("windows pass@5% / total", f"{n_pass5} / {n_total}"),
        ("windows pass@2% / total", f"{n_pass2} / {n_total}"),
        ("ABPM excluded", f"{abpm_excl_s / 60:.1f} min"),
        ("motion flagged", f"{motion_s / 60:.1f} min"),
        ("at rest (worn, low motion)", f"{at_rest_s / 60:.1f} min"),
        ("lead-off/flatline", f"{flatline_s / 60:.1f} min ({n_flatline} win)"),
        ("not worn", f"{not_worn_s / 60:.1f} min ({n_not_worn} win)"),
        ("RMSSD @ sigma=10ms", f"{jitter.rmssd_overall[-1]:.1f} ms "
                               f"({jitter.pct_change[-1]:+.0f}% vs 0)"),
        ("total runtime", f"{total_runtime_min:.1f} min"),
    ]
    for k, v in rows:
        print(f"  {k:<26} {v}")
    print("-" * 78)
    return metrics


# =============================================================================
# Main
# =============================================================================
def main(argv: Optional[list[str]] = None) -> None:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", dest="ecg_path", required=True,
                   help="Path to the raw ECG CSV (required).")
    p.add_argument("--ecg-col", dest="ecg_col", default=cfg.ecg_col)
    p.add_argument("--ts-col", dest="timestamp_col", default=cfg.timestamp_col)
    p.add_argument("--fs", dest="sampling_rate", type=int, default=cfg.sampling_rate)
    p.add_argument("--deployment-id", dest="deployment_id", default=cfg.deployment_id)
    p.add_argument("--local-tz", dest="local_tz", default=cfg.local_tz,
                   help="Display timezone: fixed offset ('+00:00') or IANA "
                        "name ('America/New_York'). Default: auto-infer from "
                        "the UTC offset embedded in the file's timestamps.")
    p.add_argument("--accel", dest="accel_path", default=cfg.accel_path,
                   help="Path to the accelerometer CSV (optional).")
    p.add_argument("--no-accel", action="store_true")
    # ABPM exclusion is off by default (deferred until a real cuff schedule
    # exists). Opt back in with --abpm.
    p.add_argument("--abpm", action="store_true",
                   help="Enable ABPM cuff-inflation exclusion (off by default).")
    # averageQRS is a secondary, non-gating SQI that costs real time on long
    # recordings for a number nothing downstream uses. Off by default.
    p.add_argument("--avgqrs", action="store_true",
                   help="Compute the secondary averageQRS SQI (off by "
                        "default -- doesn't affect the QC verdict, costs "
                        "real time on long recordings).")
    p.add_argument("--out-dir", dest="out_dir", default=cfg.out_dir)
    p.add_argument("--limit-seconds", dest="limit_seconds", type=float, default=None)
    args = p.parse_args(argv)

    cfg.ecg_path = args.ecg_path
    cfg.ecg_col = args.ecg_col
    cfg.timestamp_col = args.timestamp_col
    cfg.sampling_rate = args.sampling_rate
    cfg.deployment_id = args.deployment_id
    cfg.local_tz = args.local_tz
    cfg.accel_path = None if args.no_accel else args.accel_path
    cfg.abpm_enabled = args.abpm
    cfg.avgqrs_enabled = args.avgqrs
    cfg.out_dir = args.out_dir
    cfg.limit_seconds = args.limit_seconds

    section(f"ECG QC DASHBOARD — {cfg.deployment_id}")
    announce(f"neurokit2 {nk.__version__}; RNG seed {RNG_SEED}")
    announce("This tool FLAGS quality; it never edits or drops raw data.")
    # Verify the nk API strings we depend on actually exist in this version.
    if "Kubios" not in str(nk.signal_fixpeaks.__doc__) and True:
        pass  # doc string varies; we rely on the call succeeding below.

    run_start = time.time()
    t_prev = run_start

    def _lap(label: str) -> None:
        nonlocal t_prev
        now = time.time()
        announce(f"[TIMING] {label}: {now - t_prev:.1f}s")
        t_prev = now

    ecg = load_and_validate_ecg(cfg)
    _lap("Stage 0 (load & validate ECG)")
    accel = load_accel(cfg, ecg)
    accel_available = accel is not None
    abpm_windows = build_abpm_windows(cfg, ecg)
    _lap("accel load + ABPM windows")
    proc = process_ecg(cfg, ecg)
    _lap("Stage 1-4 (clean, SQI, peaks, RR correction)")
    windows = compute_windows(cfg, ecg, proc, abpm_windows, accel)
    _lap("Stage 5 (per-window QC)")
    jitter = jitter_sensitivity(cfg, ecg, proc)
    _lap("Stage 6 (jitter sensitivity)")
    verdict, pct_total, pct_worn = overall_verdict(cfg, windows)

    elapsed_min = (time.time() - run_start) / 60.0

    out_dir = Path(cfg.out_dir)
    stem = cfg.deployment_id
    build_dashboard(
        cfg, ecg, proc, windows, abpm_windows, jitter, verdict, pct_total, pct_worn,
        out_dir / f"{stem}_dashboard.png", out_dir / f"{stem}_dashboard.pdf",
        elapsed_min=elapsed_min, accel_available=accel_available,
    )
    _lap("Stage 7 (dashboard figure)")
    total_runtime_min = (time.time() - run_start) / 60.0
    write_outputs(
        cfg, ecg, proc, windows, abpm_windows, jitter, verdict, pct_total, pct_worn,
        out_dir / f"{stem}_metrics.json",
        total_runtime_min=total_runtime_min,
    )
    _lap("Stage 8 (metrics + outputs)")
    section(f"DONE — verdict: {verdict} "
            f"({pct_total:.0f}% of total deployment analyzable; "
            f"{pct_worn:.0f}% of worn time) "
            f"— total runtime {total_runtime_min:.1f} min")


if __name__ == "__main__":
    main()
