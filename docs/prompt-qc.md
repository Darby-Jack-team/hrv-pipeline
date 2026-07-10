# Prompt Qc

# Task
Write a single, self-contained Python QC script for one ambulatory single-lead ECG
deployment. Goal: a fast, glanceable picture of overall data quality for the deployment —
one multi-panel figure plus key metrics. This is a DIAGNOSTIC/REPORTING tool, not an
analysis step: it flags and visualizes quality but never modifies or silently drops the
raw data. Exclusion decisions are downstream.

# Environment & stack
- Python, run in the project's virtual environment. Use neurokit2 for all ECG processing.
- matplotlib for plotting (no seaborn dependency). numpy/pandas/scipy fine.
- Record the neurokit2 version in the output for reproducibility. Set a fixed RNG seed.

# Hardware context (drives several design choices — document these in comments)
- Device: single-lead wearable, 0.5–40 Hz analog bandwidth, sampled at 512 Hz.
- Subject: ~12-year-old; field conditions. Resting HR is higher than adults; HR can range
  ~45–200 bpm, so use pediatric RR plausibility bounds, NOT adult ones.
- CRITICAL distinction the script must respect: sampling rate (512 Hz, ~2 ms grid) governs
  fiducial timing RESOLUTION; the 40 Hz upper BANDWIDTH governs R-peak SHARPNESS. The low
  bandwidth broadens the R peak, which increases R-peak localization jitter under noise.
  Because RMSSD is a beat-to-beat metric, that jitter biases RMSSD UPWARD. The script must
  quantify this (see "Fiducial jitter sensitivity" below). Do not conflate the two.

# Input (parameterize in a CONFIG block at the top of the file — no hardcoded paths)
- ECG file path, ECG column name, timestamp column/handling, sampling_rate.
- Optional accelerometer columns (3- or 9-axis) for a motion-burden flag if present.
- Optional ABPM cuff-inflation schedule: either explicit timestamps or a fixed interval
  (default every 20 min) + inflation duration, to build exclusion windows.
- Window length (default 300 s), window step (default 300 s, non-overlapping).
- Window acceptance threshold: max % corrected beats per window (default 5%; also compute
  at 2% as a second cutoff).
- Pediatric RR sanity bounds (default ~300–1500 ms), parameterized.
- I do NOT know the exact raw file format. Read it defensively, validate columns/sampling
  rate, and PRINT every assumption you make. Fail loudly with a clear message if the input
  doesn't match the config rather than guessing.

# QC stages (in order)
1. Clean: nk.ecg_clean() at the configured sampling rate.
2. Signal-level SQI:
  - PRIMARY gate: two-detector agreement (bSQI). Use nk.ecg_quality(method="ici") (the
     ho2025/ici method — two independent detectors agreeing on R-peak location). This is
     the most transferable SQI because it doesn't depend on absolute spectral magnitudes
     that the 40 Hz cutoff distorts.
  - ALSO compute and report nk.ecg_quality(method="averageQRS") and method="zhao2018",
     but treat them as secondary. In comments, note: averageQRS is relative (1 = close to
     mean beat, NOT necessarily "good"); zhao2018's category thresholds were calibrated on
     full-bandwidth ECG and will misclassify a 40 Hz-limited signal, so report its output
     but do not use it as the deciding gate.
3. R-peaks: nk.ecg_peaks().
4. RR/NN correction: apply Lipponen & Tarvainen (2019) ectopic/missed/extra/misaligned
   correction via nk.signal_fixpeaks(). Check the installed neurokit2 version's API for the
   correct method string for the Kubios/Lipponen-Tarvainen algorithm (it has changed across
   versions) and use it. Apply pediatric RR sanity bounds as an additional gate. Track which
   beats were corrected and why.
5. Window level: tile the recording into 5-min windows. Per window compute: % corrected
   beats, N normal beats, valid duration, RMSSD, SDNN, mean HR. Mark each window as
   pass/fail against the acceptance threshold(s). Do NOT delete failed windows — flag them.

# Device/study-specific gates
- ABPM exclusion: flag windows overlapping any cuff-inflation interval. Report excluded time.
- Motion: if accel present, compute a per-window motion-burden metric and flag high-motion
  windows. Report motion burden alongside artifact burden (these may correlate).
- Differential-artifact check: plot/report artifact burden as a function of time-of-day so
  the reviewer can SEE whether bad windows cluster in time (this matters because motion
  artifact tends to coincide with activity/exposure). Surface it; do not "fix" it.

# Fiducial jitter sensitivity (the bandwidth proxy — implement this)
- Take the corrected R-peak series. Add zero-mean Gaussian timing noise with SD sigma for
  sigma in {0, 2, 5, 10} ms (configurable). Recompute RMSSD (overall and per-window) at each
  sigma. Report % change in RMSSD vs sigma=0.
- As a sanity check, overlay the analytic prediction RMSSD_measured ≈ sqrt(RMSSD_true^2 +
  6*sigma^2) (a constant detector lag cancels in the successive differencing; only random
  jitter contributes, with variance 6*sigma^2 per RR difference) and confirm the empirical
  curve tracks it. This characterizes how vulnerable THIS deployment's RMSSD is to the 40 Hz
  bandwidth limitation.

# Outputs
1. One multi-panel "deployment dashboard" figure saved to file (PNG + PDF), with a clear
   PASS / REVIEW / FAIL banner driven by configurable overall thresholds (e.g., % of
   recording that is analyzable). Panels:
  - Full-recording timeline with signal quality color-coded (green/yellow/red) over time,
     ABPM inflations marked as vertical lines, motion-flagged spans shaded.
  - Tachogram (RR over time) with corrected beats marked.
  - Per-window artifact burden over time, acceptance threshold(s) drawn as horizontal lines.
  - Per-window RMSSD over time, with failed/excluded windows greyed out.
  - Jitter-sensitivity panel: RMSSD vs sigma (empirical points + analytic curve).
  - A few representative beat overlays: a clean window vs a flagged window, for visual
     sanity-checking the detector.
2. A machine-readable metrics summary (JSON): deployment_id, neurokit2 version, total &
   effective duration, % analyzable, sampling rate, N beats, mean/range HR, % corrected
   beats, N windows total/pass/fail (at 5% and 2%), ABPM-excluded time, motion-flagged time,
   overall RMSSD/SDNN, and the jitter-sensitivity table.
3. A one-row summary appended to a cross-deployment CSV keyed by deployment_id, so these can
   be aggregated into a study-wide QC table later.
4. A printed human-readable summary table to console.

# Code quality
- Top config block, type hints, docstrings, defensive input validation, explicit printed
  assumptions, fixed seed, sensible figure DPI. No hardcoded paths. Keep it one file.
- Do not fabricate behavior for input formats you can't verify — validate and fail loudly.