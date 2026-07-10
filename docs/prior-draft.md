# ECG → HRV pipeline (Movesense / Fibion Flash)

Extract HRV from single-lead ECG recorded on Movesense sensors (Fibion Flash). Detect clean cardiac segments, compute HRV (whole-record \+ per-segment), QC, append to a master registry. Primary stack: **R** (RHRV \+ a Pan-Tompkins detector); a Python/NeuroKit variant exists for the detection+quality core.

## How to test

- Run R scripts with `Rscript`. Validate on a head slice / one self-recording in `data/` before any full run. R packages: tidyverse, RHRV, signal, lubridate.  
- Python path uses a reticulate venv (numpy, pandas, polars, scipy, matplotlib, neurokit2) on **Python ≥ 3.10**.

## Device facts (do not hardcode — infer from the file)

- **Sampling rate is 200 OR 256 Hz** depending on export. Always infer: `fs <- round(1/median(diff(time)))`. (256 Hz files show \~3.9 ms steps with jitter; 200 Hz files show exact 5 ms steps.)  
- **Two value encodings.** Raw integer stream: LSB \= **0.38147 µV/count**, 18-bit signed, full scale ±50 mV (so 2^17 ≈ 131072 counts \= 50 mV). A separate **mV** export is already calibrated — do NOT re-scale it. Detect from the `#unit` header (`mV` vs counts) or by magnitude (|median| \< 50 ⇒ mV).  
- Device band: **0.5–40 Hz**. The 0.5 Hz high-pass causes a \~0.25 s startup transient (front end can rail near ±50 mV, then relaxes). LPF at 40 Hz.

## File-format heterogeneity (the corpus is mixed)

- Delimiter: **tab OR comma**. Detect.  
- Time column: **numeric elapsed-seconds OR ISO-8601 string**. ISO uses unpadded day \+ colon offset (e.g. `2026-06-9T15:48:26.000+00:00`) — parse with `lubridate::ymd_hms()`, NOT base `as.POSIXct` (the colon offset breaks `%z`).  
- Header is `#`\-comment metadata lines (`# created`, `# device`, `# serial`, `# bandwidth`, `# gender`, `# age`, then `#name/#datatype/#unit` with no space).

## QRS detection — validated approach

- Morphology can be **S-dominant** (S-trough deeper than R-peak). Confirmed on real data: R ≈ \+0.33 mV, S ≈ −0.53 mV.  
- DO NOT use `abs()` \+ findpeaks: it tracks whichever of R/S is larger and **flips between fiducials** when dominance changes → injects RR jitter that corrupts RMSSD/HF (measures detector jitter as vagal tone).  
- DO: Pan-Tompkins energy (bandpass 5–15 Hz → derivative → square → 150 ms moving integration) with a **MAD-robust threshold** (`median + 4·MAD`, which the ±50 mV transients can't inflate, unlike a quantile threshold), then refine the fiducial to a **single GLOBAL polarity** chosen once for the whole record. Validated: synthetic S-dominant 80 bpm → 39/39 beats, RR CV 0.006, S-trough chosen consistently.

## Clean-segment gating \+ HRV

- Amplitude pre-screen is REQUIRED before any detector: ±50 mV transients wreck adaptive thresholds. Reject epochs with |amplitude| above an artifact ceiling (\~5 mV) or below a QRS floor.  
- HRV via RHRV: CreateHRVData → LoadBeatVector → BuildNIHR → FilterNIHR → CreateTimeAnalysis (+ InterpolateNIHR → CreateFreqAnalysis → CalculatePowerBand).  
- **Frequency domain (LF/HF) needs ≥ 5 min** of stationary data (Task Force 1996). Below that, report time-domain only; leave LF/HF as NA.  
- **VERIFY RHRV slot names with `str()` on a real object** before trusting them: expected `TimeAnalysis[[1]]$SDNN/$rMSSD/$pNN50`, `FreqAnalysis[[1]]$LF/$HF` (vectors per window). These have only been inspection-checked, not run.  
- `BuildNIHR` drops the first beat — account for it in beat counts.

## QC — keep it honest

- Coverage \= detected beats vs beats implied by the **observed median RR** over recorded duration. NOT vs an assumed 70 bpm (that just measures HR ≈ 70).  
- Quality \= **template-correlation SQI** (median correlation of each beat to the record's median beat). NOT amplitude CV mislabeled as "SNR" (amplitude CV is driven by respiration/posture, not just contact).  
- Master CSV append is **schema-validated** — refuses to append if columns differ. The old script's CSV has a different (invalid-metric) schema; archive it, don't merge. Re-run files through the new pipeline rather than reconciling.

## Known gotchas (already hit — don't repeat)

- R `lapply(x, function(k){...})` run interactively chunk-by-chunk throws "object 'k' not found"; use `for` loops \+ preallocated vectors, source whole.  
- reticulate: `use_virtualenv()` only ATTACHES; must `virtualenv_create()` first. Build against Python ≥ 3.10 (a NeuroKit dep uses `X | None` → TypeError on 3.9). Restart R before the first `use_virtualenv()` call.

## Scope caveats

- Self-recordings validate the PLUMBING completely but are one person at rest: they do NOT calibrate thresholds (`thr_k`, artifact ceiling, `rr_cv_max`) for the study population (arrhythmia, low-amplitude, pediatric HR, motion). Keep those tunable; revisit against real field recordings before locking.  
- Large files (multi-hour, \~1 GB): don't load whole. Use the chunked/`fst` random-access path; this in-memory pipeline is for per-session recordings.

## Files

Every runnable R script and Quarto/Rmd document must begin with `here::i_am("<this file's path relative to the project root>")` (e.g. a file at `analysis/03_model.R` starts with `here::i_am("analysis/03_model.R")`). Build all subsequent paths with `here::here(...)`. Never use `setwd()` or absolute paths. After `i_am()`, assert identity: `stopifnot(basename(here::here()) == "hrv-pipeline")`.

- `hrv_pipeline.R` — main: read → detect → segment → RHRV → QC → master CSV.  
- `detect_clean_ecg.R` — standalone clean-segment scanner (R).  
- `ecg_clean_detect.py` \+ `run_ecg.R` — Python/NeuroKit detection via reticulate.

