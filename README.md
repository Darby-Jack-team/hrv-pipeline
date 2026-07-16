# Fibion Flash HRV pipeline

A Python pipeline for quality-controlling and analyzing ambulatory single-lead
ECG recordings from the Fibion Flash wearable, computing heart-rate
variability (HRV) metrics with [NeuroKit2](https://neuropsychology.github.io/NeuroKit/).

## What it does

Each deployment produces a raw CSV of `timestamp, amplitude` samples at
256 Hz, optionally alongside a 3-axis accelerometer log. The pipeline:

1. **Validates** the file — verifies the actual sampling rate against the
   timestamps (never coerces it), and quarantines the file with a logged
   reason if it's out of tolerance.
2. **Detects R-peaks** with parabolic sub-sample interpolation, then
   artifact-corrects the RR series (NeuroKit's Kubios method).
3. **Quantifies signal quality** per ~5-minute window (SQI, lead-off/
   flatline, not-worn, and — optionally — motion and ABPM cuff-inflation
   exclusion), and computes an overall "analyzable %" verdict.
4. **Computes HRV** — time-, frequency-, and nonlinear-domain metrics —
   with frequency-domain metrics always computed per-epoch, never over the
   whole record.
5. **Reports** a two-page PDF/PNG QC dashboard plus machine-readable JSON
   metrics per deployment, and appends a one-row summary to a
   cross-deployment CSV.

It's a diagnostic tool: it flags and visualizes quality issues but never
silently drops or edits raw data — exclusion decisions are made downstream.

## Repo layout

```
analysis/
  00_inspect.py       first-look sanity report for a raw ECG log
  01_qc_dashboard.py  the QC dashboard / HRV pipeline (main entry point)
  qc_out/             generated dashboards + metrics (gitignored)
docs/                 project notes and prompt drafts
data-raw/             raw device exports (gitignored, never committed)
```

## Setup

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd hrv-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install neurokit2 numpy scipy pandas pyarrow matplotlib pytest
```

Then drop your raw device exports under `data-raw/<deployment-name>/` (this
directory is gitignored — nothing under it is ever committed) and you're
ready to run the pipeline below.

Before committing anything, double-check no raw data snuck into git:

```bash
git ls-files | grep -i csv   # must return nothing
```

## Input data

Raw exports come from the [Movesense device](https://bitbucket.org/movesense/movesense-device-lib/downloads/)
as one CSV per signal, named `..._log-1-<signal>_<rate>hz_cid<n>.csv`, e.g.:

- `..._ecg_256hz_cid67.csv` — single-lead ECG (`timestamp`, `ecg` columns)
- `..._acc_26hz_cid64.csv` — 3-axis accelerometer (`timestamp`, `ax`, `ay`, `az`)
- `..._hr_13hz_cid74.csv` — device-computed heart rate (not used by this pipeline)

Point `--ecg` / `--accel` at the ECG and accelerometer files for one
deployment; column and sampling-rate names can be overridden via CLI flags
if your export differs (see Usage below).

## Usage

With the venv active (`source .venv/bin/activate`) and raw files in place under
`data-raw/<deployment>/`:

**1. Sanity-check the raw file** (optional, but cheap and catches format
surprises before the full run):

```bash
.venv/bin/python analysis/00_inspect.py data-raw/<deployment>/<ecg_file>.csv
```

**2. Run the QC dashboard** on the full deployment:

```bash
.venv/bin/python analysis/01_qc_dashboard.py \
  --ecg   data-raw/<deployment>/<ecg_file>.csv \
  --accel data-raw/<deployment>/<accel_file>.csv \
  --deployment-id <deployment>
```

It prints its progress stage-by-stage as it validates the file, detects
beats, and windows the recording, ending in a summary table and an overall
verdict:

```
------------------------------------------------------------------------------
  DEPLOYMENT QC SUMMARY — example_deployment    [PASS]
------------------------------------------------------------------------------
  serial number              254530002246
  neurokit2                  0.2.13
  sampling rate              256 Hz
  recording start            2026-07-16 08:19:47 UTC
  recording end              2026-07-17 07:41:03 UTC
  total duration              23.35 h
  worn duration                22.08 h  (94.6% of total)
  % analyzable (of total time) 83.9%  <-- verdict driver
  % analyzable (of worn time) 88.7%
  N beats                    99,581
  % corrected beats          2.84%
  mean HR                    81 bpm (40-200)
  RMSSD / SDNN               93.4 / 165.6 ms
  windows pass@5% / total    235 / 280
  windows pass@2% / total    215 / 280
  ...
  total runtime               85.5 min
==============================================================================
DONE — verdict: PASS (84% of total deployment analyzable; 89% of worn time)
==============================================================================
```

`worn duration` and `% analyzable (of worn time)` answer different questions —
worn duration is a share of the *whole recording*; the analyzable percentages
are shares of their own denominator (total or worn), not of each other. Don't
expect worn-duration-as-%-of-total to match `% analyzable (of worn time)`.

`% analyzable (of total time)` — the share of the deployment's SQI-passing
windows against the *whole* recording — drives the verdict: **PASS** (≥80%),
**REVIEW** (50–80%), or **FAIL** (<50%). These thresholds live in the
`Config` dataclass at the top of `01_qc_dashboard.py`, not behind a flag.

**3. Check the outputs** in `analysis/qc_out/`:

- `<deployment>_dashboard.png` / `.pdf` — the two-page visual QC dashboard
- `<deployment>_metrics.json` — the same numbers, machine-readable
- `_deployment_qc_summary.csv` — one row per deployment, appended/updated
  each run (keyed by `--deployment-id`), for comparing across a study

Useful flags (see `--help` for the full list):

| Flag | Purpose |
|---|---|
| `--fs` | Override the expected sampling rate (default 256 Hz) |
| `--local-tz` | Display timezone (`+00:00`, `America/New_York`, ...) — by default it's auto-inferred from the UTC offset already embedded in the file's timestamps, so this rarely needs setting |
| `--no-accel` | Skip accelerometer-based motion flagging |
| `--abpm` | Enable ABPM cuff-inflation exclusion windows (off by default) |
| `--avgqrs` | Compute the secondary averageQRS SQI (off by default — doesn't affect the QC verdict, costs real time on long recordings) |
| `--limit-seconds` | Process only the first N seconds — useful for a quick smoke test |
| `--out-dir` | Change the output directory (default `analysis/qc_out/`) |

## Notes

- Raw data (`data-raw/`) and generated results (`analysis/qc_out/`) are
  gitignored and must never be committed — they contain subject health data.
- Before using any NeuroKit2 API, verify the signature against the installed
  version rather than assuming it from memory — the API has changed across
  versions (see `docs/CLAUDE.md`).
- `docs/prior-draft.md` is an earlier, unverified draft; don't build on it
  without auditing.
