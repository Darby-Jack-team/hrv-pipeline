# CLAUDE

# Project: Fibion Flash HRV pipeline (Python compute, R stats)

## Objective
Parse single-channel raw-ECG CSVs (timestamp + amplitude). Quantify usable
cardiac-data fraction; compute time-, frequency-, nonlinear-domain HRV.

## Stack
- Python + NeuroKit2 for all signal processing and HRV.
- R/tidyverse ONLY for downstream stats on output/hrv_table.parquet.

## Hard requirements (non-negotiable, not options)
- fs is LOCKED at 256 Hz but VERIFIED per file, never coerced:
  - estimate fs from timestamps (median Δt AND Δt distribution);
  - assert fs ∈ [247.5, 252.5]; else QUARANTINE the file with a logged reason.
- Fiducial timing = sample_index / 250.0, NOT raw timestamps.
- Apply parabolic sub-sample R-peak interpolation (see beats.py). Required.
- Artifact-correct RR before any HRV metric (NeuroKit Kubios method).
- Frequency-domain HRV computed per ~5-min epoch, never whole-record.
- Report the rejected/quarantined fraction alongside every HRV value.

## Verify, don't recall
Before using any NeuroKit2 function, introspect the INSTALLED version
(help(), inspect.signature) — do not assume signatures from training data.

## Notes
- docs/prior-draft.md is an UNVERIFIED earlier draft. Audit; don't build on it.
- Never modify or commit anything in data-raw/.