# Plan: Drop ABPM by default + verbose explanatory PDF

## Context
The QC dashboard (`analysis/01_qc_dashboard.py`) currently assumes an ABPM cuff-inflation
schedule (every 20 min) whenever no explicit schedule is given. On the Juliette running
recording this fabricated 24 exclusion windows (120 min) and dragged the verdict down to
REVIEW (72.6% analyzable) even though the signal itself is fine. ABPM handling will be
revisited later when real cuff schedules are available.

Two changes requested:
1. **Drop ABPM by default** so the verdict reflects actual signal quality, while keeping
   all ABPM code intact for later re-enablement.
2. **Make the report PDF more verbose** with brief explanations of each plot — delivered as
   **both** short inline captions under every panel *and* a dedicated explanation page in
   the PDF (user chose "Both").

Only one file changes: `analysis/01_qc_dashboard.py`.

## Change 1 — ABPM off by default (keep code for later)
- **CONFIG** (`Config` dataclass): set `abpm_enabled: bool = False` with a comment noting it
  is deferred until a real cuff schedule exists. Leave `abpm_explicit_starts`,
  `abpm_interval_min`, `abpm_inflation_s` in place untouched.
- **CLI** (`main`): replace the existing `--no-abpm` (store_true) with an opt-IN
  `--abpm` (store_true); set `cfg.abpm_enabled = args.abpm`. This inverts the default but
  preserves the ability to turn ABPM back on from the command line.
- `build_abpm_windows()` already returns `[]` and prints "ABPM exclusion disabled in CONFIG"
  when `abpm_enabled` is False — no change needed; it will now be the default path.
- **Dashboard panel 1** (`build_dashboard`): make ABPM references conditional on
  `abpm_windows` being non-empty — only draw the blue inflation lines and include the
  "ABPM excluded" legend patch when there are windows; adjust the panel title string so it
  doesn't mention ABPM when none are present.
- Verdict logic (`overall_verdict`) is unchanged; with no exclusions the analyzable % rises
  (Juliette should move toward PASS).

## Change 2 — Verbose PDF: inline captions + explanation page

### (a) Inline captions under each panel (appear in PNG and PDF)
- Add a small helper `panel_caption(ax, text)` that renders a wrapped, italic, grey
  (`color="0.35"`, `fontsize=7`, `style="italic"`) caption just below each panel via
  `ax.text(0.0, -0.30, text, transform=ax.transAxes, va="top", wrap=True)`.
- Apply to all 8 content panels (timeline, tachogram, time-of-day, artifact-burden,
  per-window RMSSD, jitter, both beat overlays).
- Give the captions room: bump `gridspec` `hspace` (~0.45 -> ~0.65) and increase figure
  height (~18 -> ~21 in) so captions don't collide with the next panel's title.
- One-line caption text per panel (final wording tuned in code), e.g.:
  - Timeline: "Cleaned ECG; window shading = quality (green <=2%, yellow <=5%, red >5%
    corrected beats); purple ticks = high motion."
  - Tachogram: "Beat-to-beat RR; red = beats corrected (Lipponen-Tarvainen); dotted =
    pediatric plausibility bounds 300-1500 ms."
  - Time-of-day: "Same windows by clock hour — reveals whether bad windows cluster with
    activity/time."
  - Artifact burden: "% corrected beats per 5-min window; dashed = 5% / 2% acceptance lines."
  - Per-window RMSSD: "Short-term HRV per window; greyed = failed acceptance or excluded."
  - Jitter: "RMSSD vs injected R-peak jitter; curve = sqrt(RMSSD^2 + 6*sigma^2); steep =
    RMSSD vulnerable to the 40 Hz bandwidth."
  - Cleanest overlay: "<=60 beats from the best window aligned at R (red) — expect one
    consistent QRS."
  - Most-flagged overlay: "Same for the worst window — smeared/variable shapes flag motion
    or detector error."

### (b) Explanation page appended to the PDF (PDF only; PNG stays single image)
- In `build_dashboard`, switch the PDF write to multi-page using
  `matplotlib.backends.backend_pdf.PdfPages`:
  - Save the PNG from the dashboard figure exactly as today.
  - Open `PdfPages(out_pdf)`, write the dashboard figure as page 1, then build a second
    portrait figure (axis off) as page 2 and write it, then close.
- Page 2 content — a "How to read this report" page that reuses values already passed into
  `build_dashboard` (`cfg`, `verdict`, `analyzable_pct`, `nk.__version__`):
  - Header line with `deployment_id`, verdict, % analyzable, nk version, fs.
  - One short paragraph per panel (the longer form of the captions above).
  - "Verdict logic": PASS/REVIEW/FAIL is driven by % of recording that is analyzable
    (`pass_analyzable_pct` / `review_analyzable_pct`), where a window is analyzable if it
    passes the primary corrected-beat threshold and is not ABPM-excluded.
  - "Key caveats / assumptions": sampling rate is validated against the file's own
    timestamps; clock gaps are treated as continuous; ECG values are raw uncalibrated ADC
    counts; pediatric RR bounds 300-1500 ms; **ABPM exclusion is currently disabled
    (deferred)**.
  - "Jitter note": low-RMSSD recordings are far more vulnerable to the 40 Hz-bandwidth
    timing jitter than high-RMSSD ones (contrast across deployments).

## Files
- `analysis/01_qc_dashboard.py` — the only file modified. Key functions touched:
  `Config`, `main` (CLI), `build_abpm_windows` (no change), `build_dashboard` (captions +
  multi-page PDF + conditional ABPM drawing), and a new `panel_caption()` helper and a new
  explanation-page builder (e.g. `_build_explanation_page(cfg, verdict, analyzable_pct)`).

## Verification
1. Fast smoke test (first 20 min):
   `conda run -n graphs-ecg python analysis/01_qc_dashboard.py --limit-seconds 1200
   --deployment-id smoke_test` — confirm log prints "ABPM exclusion disabled in CONFIG",
   `ABPM excluded 0.0 min`, run completes exit 0.
2. Full Juliette re-run (no `--abpm`):
   `conda run -n graphs-ecg python analysis/01_qc_dashboard.py
   --ecg data-raw/Juliette_test_June16_electrode/_julietteJune16-run-electrode-log-1-ecg_256hz_cid67.csv
   --accel data-raw/Juliette_test_June16_electrode/_julietteJune16-run-electrode-log-1-acc_26hz_cid64.csv
   --deployment-id Juliette_June16_run`
  - Expect verdict to improve (toward PASS) now that 120 min of spurious exclusions are gone.
3. Inspect outputs:
  - Open `analysis/qc_out/Juliette_June16_run_dashboard.png` — confirm a caption sits under
     each panel and no ABPM blue lines / legend patch appear.
  - Confirm the PDF has **2 pages** (dashboard + explanation), e.g. check `/Count` via
     `strings analysis/qc_out/Juliette_June16_run_dashboard.pdf | grep -m1 "/Count"`.
  - Spot-check `Juliette_June16_run_metrics.json` shows `abpm_excluded_s: 0`.
