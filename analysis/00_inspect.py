#!/usr/bin/env python
"""
00_inspect.py — first-look sanity report for a raw ECG log.

Run with the project venv so pandas/numpy are available:

    .venv/bin/python analysis/00_inspect.py [path/to/file.csv]

Reports, in plain language:
  - column names and dtypes
  - timestamp resolution (smallest tick the clock actually uses)
  - median sample interval (dt) and a dt histogram
  - implied sampling rate (fs)
  - total record duration
  - amplitude range and a guess at units
  - any gaps (places where the clock jumped more than expected)

It makes no changes to the data; it only reads and describes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FILE = (
    "data-raw/Darby_test_June13/_darbytestJune13-log-1-ecg_256hz_cid67.csv"
)


def human_duration(seconds: float) -> str:
    """Turn a number of seconds into 'Hh Mm Ss'."""
    seconds = float(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{int(h)}h")
    if m or h:
        parts.append(f"{int(m)}m")
    parts.append(f"{s:.3f}s")
    return " ".join(parts)


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main(path: str) -> None:
    csv = Path(path)
    if not csv.exists():
        sys.exit(f"File not found: {csv}")

    print(f"Inspecting: {csv}")
    print(f"Size on disk: {csv.stat().st_size / 1e6:.1f} MB")

    # --- Load -------------------------------------------------------------
    # The timestamp is ISO-8601 with a timezone offset, e.g.
    #   2026-06-14T13:50:12.004-04:00
    # Parse it as a real datetime so we can reason about time. Everything
    # else is read as-is.
    df = pd.read_csv(csv)
    raw_cols = list(df.columns)

    ts_col = raw_cols[0]
    # Assume the first non-timestamp column is the signal.
    signal_col = next((c for c in raw_cols[1:]), None)

    # Parse timestamps to UTC so DST/offset quirks don't distort dt.
    ts = pd.to_datetime(df[ts_col], utc=True, format="ISO8601")

    # --- Columns and dtypes ----------------------------------------------
    rule("Columns and dtypes")
    print(f"{len(raw_cols)} columns, {len(df):,} rows")
    for c in raw_cols:
        print(f"  - {c!r}: {df[c].dtype}")
    print(f"  (timestamp column read as text, parsed to: {ts.dtype})")

    # --- Timestamp resolution --------------------------------------------
    # How fine is the clock, really? Look at how many decimal places the
    # raw text uses, and the smallest nonzero step the clock actually takes.
    rule("Timestamp resolution")
    sample_ts_text = str(df[ts_col].iloc[0])
    print(f"First timestamp: {sample_ts_text}")
    print(f"Last  timestamp: {df[ts_col].iloc[-1]}")

    frac = df[ts_col].astype(str).str.extract(r"\.(\d+)")[0].dropna()
    if not frac.empty:
        decimals = frac.str.len().max()
        print(f"Fractional-second digits in text: up to {decimals} "
              f"(i.e. recorded to the {'ms' if decimals == 3 else f'1e-{decimals}s'} place)")

    ns = ts.astype("int64").to_numpy()  # nanoseconds since epoch, UTC
    dt_ns = np.diff(ns)
    nonzero = dt_ns[dt_ns != 0]
    if nonzero.size:
        tick_ns = int(np.gcd.reduce(np.abs(nonzero)))
        print(f"Smallest clock tick actually used: {tick_ns / 1e6:.3f} ms "
              f"(GCD of all sample-to-sample steps)")

    # --- dt: median interval and histogram --------------------------------
    rule("Sample interval (dt) and implied fs")
    dt_s = dt_ns / 1e9
    median_dt = float(np.median(dt_s))
    mean_dt = float(np.mean(dt_s))
    print(f"Median dt: {median_dt * 1e3:.4f} ms")
    print(f"Mean   dt: {mean_dt * 1e3:.4f} ms")
    print(f"Implied fs (from median dt): {1.0 / median_dt:.3f} Hz")
    print(f"Implied fs (from mean dt):   {1.0 / mean_dt:.3f} Hz")

    # Histogram of the distinct dt values. ECG clocks usually quantize the
    # interval to a few discrete millisecond values, so show them by count.
    dt_ms_rounded = np.round(dt_s * 1e3, 3)
    vals, counts = np.unique(dt_ms_rounded, return_counts=True)
    order = np.argsort(counts)[::-1]
    print("\ndt histogram (most common steps first):")
    print(f"  {'dt (ms)':>10}  {'count':>12}  {'share':>7}")
    total = counts.sum()
    shown = 0
    for i in order:
        if shown >= 12 and counts[i] / total < 0.001:
            break
        bar = "#" * int(40 * counts[i] / counts.max())
        print(f"  {vals[i]:>10.3f}  {counts[i]:>12,}  "
              f"{100 * counts[i] / total:>6.2f}%  {bar}")
        shown += 1
    if shown < len(vals):
        print(f"  ... and {len(vals) - shown} rarer dt value(s)")

    # --- Record duration --------------------------------------------------
    rule("Record duration")
    span_s = (ns[-1] - ns[0]) / 1e9
    print(f"Start: {ts.iloc[0]}")
    print(f"End:   {ts.iloc[-1]}")
    print(f"Wall-clock span: {human_duration(span_s)}  ({span_s:.3f} s)")
    print(f"Samples: {len(df):,}")
    print(f"Samples / fs(median): {human_duration(len(df) * median_dt)} "
          f"of signal if perfectly uniform")

    # --- Amplitude range / units -----------------------------------------
    rule("Amplitude range and units")
    if signal_col is None:
        print("No signal column found.")
    else:
        s = pd.to_numeric(df[signal_col], errors="coerce")
        n_nan = int(s.isna().sum())
        smin, smax = float(s.min()), float(s.max())
        print(f"Signal column: {signal_col!r}")
        print(f"  min = {smin:.3f}")
        print(f"  max = {smax:.3f}")
        print(f"  range (max-min) = {smax - smin:.3f}")
        print(f"  mean = {s.mean():.3f}, std = {s.std():.3f}")
        if n_nan:
            print(f"  non-numeric / missing values: {n_nan:,}")
        # Unit guess: the column name says nothing about units, and the
        # values are signed integers-ish in the hundreds-to-thousands range,
        # which is typical of raw/uncalibrated ADC counts rather than mV.
        looks_intish = np.allclose(s.dropna(), np.round(s.dropna()))
        print("  units: not labeled in the file. Values are "
              + ("whole-number-like " if looks_intish else "fractional ")
              + "and signed,")
        if abs(smin) < 50 and abs(smax) < 50:
            print("         within a small +/- range — could be millivolts (mV).")
        else:
            print("         spanning a wide signed range — most likely raw ADC")
            print("         counts (uncalibrated), not millivolts. Confirm with the device spec.")

    # --- Gaps -------------------------------------------------------------
    # A "gap" is a sample-to-sample jump much larger than the normal step.
    # Flag anything bigger than 1.5x the most common dt (and any negative
    # steps, which would mean the clock went backwards).
    rule("Gaps and clock issues")
    normal_dt = float(vals[counts.argmax()]) / 1e3  # most common dt, seconds
    gap_threshold = 1.5 * normal_dt
    gap_idx = np.where(dt_s > gap_threshold)[0]
    back_idx = np.where(dt_s < 0)[0]

    print(f"Normal step (mode dt): {normal_dt * 1e3:.3f} ms; "
          f"flagging jumps > {gap_threshold * 1e3:.3f} ms")
    if back_idx.size:
        print(f"!! {back_idx.size} backwards time step(s) — clock went in reverse.")
    if gap_idx.size == 0 and back_idx.size == 0:
        print("No gaps found. Sampling is continuous within the flagging threshold.")
    else:
        missing_total = 0.0
        print(f"{gap_idx.size} gap(s) found. Showing up to 20:")
        for i in gap_idx[:20]:
            jump = dt_s[i]
            missing = jump / normal_dt - 1.0
            missing_total += missing
            print(f"  row {i:>12,} -> {i + 1:<12,}  "
                  f"jump {jump * 1e3:>10.3f} ms  "
                  f"(~{missing:.1f} samples missing)  at {ts.iloc[i]}")
        if gap_idx.size > 20:
            for i in gap_idx[20:]:
                missing_total += dt_s[i] / normal_dt - 1.0
            print(f"  ... and {gap_idx.size - 20} more gap(s)")
        print(f"Estimated total missing samples across all gaps: "
              f"~{missing_total:.0f} "
              f"(~{human_duration(missing_total * normal_dt)})")

    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE)
