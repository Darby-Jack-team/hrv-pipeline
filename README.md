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

Sanity-check a raw file before running the full pipeline:

```bash
.venv/bin/python analysis/00_inspect.py data-raw/<deployment>/<ecg_file>.csv
```

Run the QC dashboard on a full deployment:

```bash
.venv/bin/python analysis/01_qc_dashboard.py \
  --ecg   data-raw/<deployment>/<ecg_file>.csv \
  --accel data-raw/<deployment>/<accel_file>.csv \
  --deployment-id <deployment>
```

Outputs land in `analysis/qc_out/`: `<deployment>_dashboard.{png,pdf}`,
`<deployment>_metrics.json`, and an appended row in
`_deployment_qc_summary.csv`.

Useful flags (see `--help` for the full list):

| Flag | Purpose |
|---|---|
| `--fs` | Override the expected sampling rate (default 256 Hz) |
| `--local-tz` | Display timezone (`+00:00`, `America/New_York`, ...) — by default it's auto-inferred from the UTC offset already embedded in the file's timestamps, so this rarely needs setting |
| `--no-accel` | Skip accelerometer-based motion flagging |
| `--abpm` | Enable ABPM cuff-inflation exclusion windows (off by default) |
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
