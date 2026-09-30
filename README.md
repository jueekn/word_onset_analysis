# word_onset_analysis

Word-onset spectral power and phase connectivity in human intracranial EEG,
across 12 ROIs (used in Burke et. al. 2013), from the Penn Computational Memory Lab free-recall
datasets (FR1 / catFR1 / pyFR).

The core contrast is **`word_on`**: high-frequency activity while a word is on
the screen (0–600 ms after onset) versus the blank inter-stimulus interval
before it (−700 to −100 ms), for the same trials. It extends Long et al. (2020),
*Feed-forward, feed-back, and distributed feature representation during visual
word recognition*, to a larger electrode sample with bipolar re-referencing, and
adds phase-connectivity analyses.

---

## Requirements

Python ≥ 3.11 and `pip install -r requirements.txt`. Two of the pins come
from GitHub rather than PyPI: `ptsa` (needs a C++ compiler and FFTW —
`brew install fftw` on macOS, then `CPPFLAGS=-I$(brew --prefix fftw)/include
LDFLAGS=-L$(brew --prefix fftw)/lib pip install ...`) and `bidsreader`, CML's
reader for the BIDS exports of its datasets.

## Data

Everything is read from the lab's public BIDS datasets on OpenNeuro
(FR1 `ds004789`, catFR1 `ds004809`, pyFR `ds004865`); nothing needs the lab
cluster. `cml_data.py` lists a dataset on S3 and downloads what a session needs
on first use — events / channel / electrode tables (~1 MB) and, for the
compute stages, the bipolar recording (300–700 MB per session) — into
`config.yaml paths.bids_cache` (default `~/scratch/word_onset_analysis/bids_data`;
`CML_BIDS_CACHE` overrides). Files are never re-downloaded.
You are asked once per run before anything is fetched; set
`CML_AUTO_APPROVE=1` for unattended runs.

Not every rhino session is on OpenNeuro yet (as of 2026-09: FR1 149/153 of the
original subjects, catFR1 31/33, pyFR 42/80). The session list is whatever
OpenNeuro lists, so the cohort grows as the release is updated.

## How it runs

```bash
snakemake --cores 8                               # full run (config.yaml `runs`: alpha + high gamma, PPC)
snakemake --cores 4 --config smokescreen=true     # first 3 subjects, *.smokescreen outputs
snakemake simulations --cores 4                   # validity checks -> figures/simulations/
# add --config longetal=true for the Long et al. 2020 replication (power only) -> *.longetal
```

or by hand:

```bash
python prepare_sessions.py --n-subjects 3 --workers 4   # session list, data check, events
python build_roi_power.py     --band high_gamma --workers 4
python build_roi_synchrony.py --band high_gamma --workers 4
python build_power_synchrony.py --band high_gamma
```

`prepare_sessions.py` writes the session list and per-session artifacts under
`config.yaml paths.scratch_dir` (default `~/scratch/word_onset_analysis/scratch`,
outside the repo; smokescreen / longetal runs use sibling `scratch.*` folders).
`--subjects` / `--n-subjects` restrict it. A session is included only if its
recalls also match for the encoding and retrieval contrasts (kept so the cohort
matches earlier runs; see `load_events.recall_matching_ok`).

Every build script has `--stage compute` (one pickle per session under
`SCRATCH_DIR/word_on/…`, `--workers N` in parallel) and `--stage plot`
(figures + CSVs); `--stage both` is the default. Compute is cached on the file
existing, **not** on the settings that produced it — after changing anything
upstream (band, notch, buffers, windows) delete the output directory.

Simulations (`--simulation-tag`, config/simulation_config.yaml) keep each real
session's montage and events and replace only the signal, so they exercise the
whole real pipeline.

---

## Figures (word on − word off only)

Each plot stage redraws one figure per analysis with a row per band already
plotted into its directory (alpha, high gamma), paired t per ROI, BH-FDR.

| directory | figure | |
|---|---|---|
| `burke_roi_power/` | `roi_power_word_on.png` | Cohen's *d* of power per ROI |
| | `responsiveness_word_on_<band>_multitaper.png` | per-electrode \|t\| and number of responsive electrodes (p < 1e-8); `--fine-labels` adds a figure by raw label (top 15) |
| | `power_timecourse_word_on_high_gamma_multitaper_50ms.png` | *d* vs latency per ROI — **gamma only** |
| `burke_roi_synchrony/` | `roi_synchrony_word_on_ppc.png` | on − off PPC per ROI, collapsed over distance |
| | `roi_synchrony_distance_word_on_<band>_ppc.png` | the same per distance bin, per ROI |
| `power_synchrony/` | `power_synchrony_word_on_ppc.png` | across-electrode r of power *d* vs synchrony change |

Synchrony is raw PPC by default (whole-brain changes show). `--zscore`
z-scores each distance bin across the session's electrodes first, leaving only
ROI-vs-ROI differences. `--rmin/--rmax/--bin-w` set the bins,
`--exclude-same-shank` drops same-lead pairs; pair centroids are in MNI.

---

## Key defaults

| parameter | value | note |
|---|---|---|
| `resample_hz` | 500 | Nyquist 250 Hz |
| `min_sample_rate_hz` | 499 | sessions below this are excluded upstream |
| `bands.high_gamma` | 70–150 Hz | Long et al.'s band |
| `bands.low` / `bands.gamma` | 3–8 / 70–110 Hz | PAC phase / amplitude bands |
| `notch_harmonics_up_to_hz` | 150 | site-uniform: notches 100/120/150 for *every* subject |
| `mt_bandwidth` | 2 | NW = 0.6 over a 600 ms window |
| `time_bin_ms` / `mt_window_ms` | 50 / 100 | gamma time course: bin step / sliding window |
| `real_data_buffer_ms` | 500 | real EEG around each window, cropped before the multitaper |
| `computation_metrics` | coh, plv, ppc, ciplv, pli, wpli, pac | `build_roi_synchrony --metrics` |

---

## Layout

```
prepare_sessions.py         session list, data check, events (run first)
build_roi_power.py          power, responsiveness, gamma time course
build_roi_synchrony.py      phase connectivity, per ROI and vs distance
build_power_synchrony.py    power-synchrony correlation (consumes the other two)
plot_recovery.py            simulation recovery curves
roi_subregion_counts.py     ROI/subregion coverage tables (standalone)

fc_comparison_functions.py  connectivity estimators, ROI stats, figures, session dispatch
helper.py                   BIDS EEG / electrode loading, resample + notch, regionalisation
simulate_eeg.py             synthetic EEG for the validity checks
cml_data.py                 OpenNeuro listing / download / cache
data_check.py               per-session validation -> sess_list_df_data_check.json
load_events.py  match_events.py  word_on events (+ recall matching for the cohort rule)
exclusion_log.py            event/session exclusion counts
misc.py  project_paths.py  ptsa_patches.py   pickles, config constants, PTSA numpy patch
config/config.yaml          all tunable parameters
```

`region_translator.csv` maps atlas labels → region; `region_to_burke_lobe.csv`
maps region → one of the 12 ROIs.
