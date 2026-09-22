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
`./bids_data` (override with `CML_BIDS_CACHE`). Files are never re-downloaded.
You are asked once per run before anything is fetched; set
`CML_AUTO_APPROVE=1` for unattended runs.

Not every rhino session is on OpenNeuro yet (as of 2026-09: FR1 149/153 of the
original subjects, catFR1 31/33, pyFR 42/80). The session list is whatever
OpenNeuro lists, so the cohort grows as the release is updated.

## How it runs

```bash
python prepare_sessions.py --n-subjects 3 --workers 4   # session list, data check, events
python build_roi_power.py     --beh word_on --band high_gamma --workers 4
python build_roi_synchrony.py --beh word_on --band high_gamma --workers 4
python build_power_synchrony.py --beh word_on --band high_gamma
```

`prepare_sessions.py` writes the session list and per-session artifacts under
`config.yaml paths.scratch_dir` (default `./scratch`): `sess_list_df*.json`,
`electrode_information/pairs/`, `<beh>/events/`. `--subjects`, `--n-subjects`
and `--n-sessions` restrict it; the whole cohort is ~1300 sessions.

Every build script has two stages:

| stage | function | location |
|---|---|---|
| `--stage compute` | one pickle per session | `SCRATCH_DIR/<beh>/…` (`--workers N` runs N sessions at once in separate processes) |
| `--stage plot` | aggregates pickles → figures + CSVs | `figures/…` |

`--stage both` (the default) does both.

Compute is cached: a session with an existing, complete pickle is skipped.
The cache keys on the file existing and having the expected fields, **not** on
the settings that produced it — so after changing anything upstream (band,
notch, buffers, windows) you must **delete the output directory**.

---

## Figure recipes

Defaults come from `config/config.yaml`.

### 1. ROI power (+ responsiveness, + latency)

```bash
# compute once per (beh, band, fc-mode)
python build_roi_power.py --stage compute --beh word_on --band high_gamma --fc-mode multitaper

# figures
python build_roi_power.py --stage plot --beh word_on --band high_gamma \
    --fc-mode multitaper --fine-labels --responsive-only --tfce-perm 1024
```

Writes to `figures/burke_roi_power/` (Morlet runs go to a `cwt_morlet/`
subfolder):

| file | description |
|---|---|
| `roi_power_<beh>_<band>.png` | per-ROI Cohen's *d*, box plot over subjects |
| `responsiveness_<…>.png` | per-electrode \|t\| distribution + count of responsive electrodes |
| `responsiveness_<…>_fine.png` | same, split by fine `reg_full` label (top 15 by median \|t\|) |
| `power_timecourse_<…>_50ms.png` | Cohen's *d* vs latency, one panel per ROI |
| `power_timecourse_<…>_responsive.png` | same, restricted to task-responsive contacts |

Flags that matter:

- `--fc-mode {multitaper,cwt_morlet}` — multitaper is band-averaged;
  Morlet is constant-Q and gives genuinely independent 50 ms bins. Outputs go to
  separate directories, so both can coexist.
- `--tfce-perm 1024` — TFCE via `mne.stats.permutation_cluster_1samp_test`,
  clustering along time within each ROI, then FDR across ROIs. `0` (default)
  uses Benjamini–Hochberg on the per-bin *t* instead.
- `--responsive-only` — additionally emits the time course using only contacts
  at *p* < `--responsive-alpha` (default 1e-8, matching Long et al.), plus a
  per-ROI electrode-count CSV.
- `--min-electrodes` (default 3) — minimum electrodes per (subject, ROI) cell.

### 2. ROI phase synchrony

```bash
python build_roi_synchrony.py --stage compute --beh word_on --band high_gamma --metrics ppc
python build_roi_synchrony.py --stage plot    --beh word_on --band high_gamma --metric ppc
```

→ `figures/burke_roi_synchrony/`. Pair-level connectivity is binned by
seed–target distance and z-scored **within** each distance bin across the
session's electrodes before being collapsed to one score per electrode
(`fc.collapsed_synchrony`). Without that, an electrode's score partly reports
where it sits rather than what it does, because connectivity falls off with
distance and montages sample distances very unevenly.

`--rmin/--rmax/--bin-w` control the distance bins; `--exclude-same-shank`
drops pairs on the same lead. All pair centroids are in MNI152NLin6ASym (the
one space the BIDS electrode tables carry), so any two are comparable.

### 3. Power–synchrony correlation

**Requires both of the above to have been computed first** — consumes
their pickles.

```bash
python build_power_synchrony.py --beh word_on --band high_gamma --metric ppc
```

→ `figures/power_synchrony/`. Correlates each electrode's power effect against
its synchrony score across electrodes within a session, averages *r* within
subject, then tests across subjects — the structure Rao et al. (2025) use, but
computed per ROI and with the distance-collapsed synchrony score.

---

## Key defaults

| parameter | value | note |
|---|---|---|
| `resample_hz` | 500 | Nyquist 250 Hz |
| `min_sample_rate_hz` | 499 | sessions below this are excluded upstream |
| `bands.high_gamma` | 70–150 Hz | Long et al.'s band; geometric centre 102.5 Hz |
| `bands.gamma` | 70–110 Hz | **filter** band for AEC/PAC — not the analysis band |
| `notch_harmonics_up_to_hz` | 150 | site-uniform: notches 100/120/150 for *every* subject |
| `mt_bandwidth` | 2 | NW = 0.6 over a 600 ms window |
| `mt_window_ms` | 100 | sliding multitaper window for the latency axis |
| `time_bin_ms` | 50 | latency bin width |
| `cwt_fnum` / `cwt_morlet_reps` | 6 / 5 | log-spaced Morlet bank |
| `cwt_buffer_n_sigma` | 4.0 | 45.5 ms at 70 Hz |
| `real_data_buffer_ms` | 50 | real adjacent EEG loaded around each window |

---

## Layout

```
prepare_sessions.py         session list, data check, events (run first)
build_roi_power.py          power, responsiveness, latency  (entry point)
build_roi_synchrony.py      phase connectivity              (entry point)
build_power_synchrony.py    power-synchrony correlation     (entry point, consumes the other two)
burke_roi_connectivity.py   per-seed connectivity vs distance
plot_phase_conn_distance*.py  distance curves, whole-brain and per-ROI
compare_distance_controls.py  which distance correction removes the most geometry
roi_subregion_counts.py     ROI/subregion coverage tables

fc_comparison_functions.py  FC estimators, aggregation, stats, plotting primitives, session dispatch
helper.py                   BIDS EEG / electrode loading, notch, Morlet, regionalisation
cml_data.py                 OpenNeuro listing / download / cache (identical to the COGS4290 copy)
data_check.py               per-session validation -> sess_list_df_data_check.json
load_events.py  match_events.py  per-behavior events (matched succ/unsucc for en/rm)
exclusion_log.py            event/session exclusion counts
cstat.py                    circular statistics (PPC/PLV/ciPLV/PLI)
wavelet.py                  Morlet bank definition (frequencies, widths)
misc.py  matrix_operations.py  figure_io.py   small utilities
project_paths.py            config.yaml -> paths and constants
ptsa_patches.py             PTSA monkey-patch (numpy alias restore)
simulate_eeg.py             synthetic EEG for validation
config/config.yaml          all tunable parameters
```

`region_translator.csv` maps atlas labels → region; `region_to_burke_lobe.csv`
maps region → one of the 12 main ROIs.

---



