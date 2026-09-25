# Changelog

Project-level change log for `phase_visualization`. Newest entries first.

Entries before 2026-08-17 predate git and are not reconstructable; anything
earlier than the last entry below is undocumented.

---

## 2026-09-25 — Scratch and downloads moved out of the repo

- `paths.scratch_dir` → `~/scratch/word_onset_analysis/scratch` (sibling
  `scratch.smokescreen`, `scratch.longetal`, ...); new `paths.bids_cache` →
  `~/scratch/word_onset_analysis/bids_data`, exported as `CML_BIDS_CACHE` by
  project_paths. Existing folders were moved, not recomputed. Figures stay in
  the repo.
- Convention: "juee flags" = the default methods (bipolar, all experiments,
  multitaper, own windows/stats); their results are on hold and never
  overwritten. longetal results are wiped and regenerated on every method
  change. Full-cohort longetal now: 262 sessions / 150 subjects; responsive at
  p<1e-8: 2.7% (per trial), 13.2% (per sample), vs Long 25.3%.

---

## 2026-09-24 — longetal: Long et al. 2020 replication

`snakemake [all|simulations] --config longetal=true` (settings in config
`longetal_params:`; `WOA_LONGETAL` in project_paths; outputs `*.longetal`).
Default runs are unchanged.

- FR1 only; complete sessions only (≥300 WORD events, rule `incomplete_session`).
- Monopolar contacts (one "pair" per contact, distance 0; ECOG/SEEG channels
  only) + common average reference per sample over the kept contacts, applied
  after any simulated EEG so simulations see it too.
- No resample; `--fc-mode hilbert`: band-pass 70–150, Hilbert amplitude at the
  native rate, 500 ms buffer trimmed, envelope → 100 Hz, mean per window (or
  50 ms bins). Pickles store amplitude (`measure: amplitude`); change_dB and
  recovery use 20·log10.
- Windows: blank −750–0 ms vs word 0–1600 ms (not length-equalized).
- Only word_on events required. Responsiveness figure keeps regions with ≥50
  responsive electrodes (CSV keeps all). Power only, no synchrony.
- `exclude_lobes: [limbic, hippocampus, subcortical]` (Long excluded limbic and
  sublobar electrodes): those ROIs map to None under longetal.
- `--t-unit {trial,sample}` (plot stage): Long doesn't say whether the
  responsiveness t-test used trial means or every envelope sample. Hilbert
  pickles now also store `cohens_d_samples` / `n_{lo,hi}_samples`. Smoke:
  6.6% responsive per trial vs 19.3% per sample (Long 25.3%).
- `--combine {d_mean,word_average}` (longetal default word_average): Long
  averaged each word over repeated sessions before testing. Hilbert pickles
  store per-event envelopes (`env_lo`/`env_hi`, float32) and words (`items`);
  `word_averaged_rows` averages per word within subject (labels present in all
  sessions), then d across words or samples. `get_eeg` now labels each epoch
  with its position in the input events (was 0..n-1), so dropped events keep
  their identity. Smoke: 23.1% responsive (word_average, sample) vs Long 25.3%.
- `long_regions.csv`: approximate map from our region names to Long's 16
  subregions (hemispheres pooled); used for longetal `--fine-labels`.
  `exclude_regions: [posterior cingulate gyrus]` (limbic in her atlas).
- `get_eeg` drops events whose clip isn't fully inside the recording before
  epoching (MNE raised instead; truncated OpenNeuro EDFs, e.g. R1156D, R1161E).

---

## 2026-09-24 — Recovery curves

- `sweeps:` in `config/simulation_config.yaml` expand (simulate_eeg loader) to
  one tag per planted value: power gain 1–4 (0–12 dB) and frontal PPC 0–0.99,
  each with and without pink noise. `snakemake simulations` runs them
  compute-only and `plot_recovery.py` draws measured vs planted →
  `<figures>/simulations/recovery.png`.
- Smoke result: noise-free power on the identity line; noisy power below it
  (pink noise shares the band); PPC slightly below identity mid-range (≈0.46 at
  0.5) with or without noise; non-target electrodes 0 throughout.

---

## 2026-09-23 — Snakefile, smokescreen, simulations, no log power

- **Snakefile** runs prepare → ROI power + ROI synchrony → power-synchrony for
  each config `runs:` entry. `--config smokescreen=true` = first 3 subjects,
  outputs under `*.smokescreen` (`WOA_SMOKESCREEN` in project_paths).
  `profiles/default` sets the greedy scheduler (default solver is x86-only).
- **`snakemake simulations`**: `--simulation-tag` on all build scripts; EEG is
  replaced by a `config/simulation_config.yaml` generator, pickles go to
  `<scratch>/sim/<tag>`, figures to `<figures>/simulations/<tag>`. New tags:
  `hg_null`, `hg_power_frontal` (70-150 Hz gain after onset, frontal only),
  `hg_ppc_frontal` (110 Hz phase coupling after onset, frontal only).
  Smoke result: effects recovered in frontal only (power d ≈ +5.8, synchrony
  +1.0), null ≈ 0 everywhere — after the buffer fix below.
- **`real_data_buffer_ms` 50 → 500, and build_roi_power now uses it** (it loaded
  0 ms for multitaper, ~45 ms for Morlet). The 4 Hz-wide 100/120/150 Hz notches
  ring for a few hundred ms; with the windows 0–50 ms from the clip edges, null
  power read d ≈ -0.08 in every ROI. Changes every prepost result.
- All simulation code moved from `helper.py` to `simulate_eeg.py`.
- **Power is no longer log-transformed** (PI): Cohen's d on raw band power.
  Pickle keys `log10_lo/hi` → `pow_lo/hi`, so old pickles recompute.
- `fc.run_sessions` writes `<scratch>/logs/<stage>_errors.log`, one capped line
  per failed session, overwritten each run.

---

## 2026-09-21 — OpenNeuro / BIDS data; no cluster dependency

The pipeline no longer needs rhino. Data comes from CML's public BIDS datasets
on OpenNeuro through `bidsreader` (the BIDS counterpart of cmlreaders) and
`cml_data.py` (S3 listing / download / cache, copied verbatim from
COGS4290_DataMemoryBrainsSolutions `grader-autograder`); parallelism is a local
process pool (`--workers N`) instead of cmldask/SLURM. Verified end to end on
R1111M FR1 (OpenNeuro `ds004789`): prepare_sessions → build_roi_power →
build_roi_synchrony → build_power_synchrony, single- and multi-worker.

### Session key

`(sub, exp, sess)`. BIDS has no localization/montage, so `loc`/`mon` are gone
from `ftag`, `get_dfrow`, `sid` tuples, `sess_list_df*`, `excluded_sessions.csv`,
`unrecoverable_sessions.csv` and the exclusion logs. Old pickles / session
lists are not readable by the new code.

### Data loading (`helper.py`)

- `bids_reader(dfrow)` → `CMLBIDSReader` for the session, fetching its files on
  first use; `prefetch_bids(dfrows)` fetches a list up front (used before any
  multi-worker run so workers never download).
- `get_eeg` epochs the bipolar recording with `mne.Epochs` on the events'
  `eegoffset` (= BIDS `sample`) and returns the same `(event, channel, time)`
  PTSA TimeSeries as before, in microvolts with time in ms, channels in
  `get_pairs` order. Repeated onsets (WORD + its PRE_WORD copy) are epoched
  once and expanded. Events whose clip would run off the recording are dropped
  from both the EEG and the mask (reported), where cmlreaders asserted.
- `load_pairs_table(reader)` builds the pairs table from
  `load_combined_channels(acquisition='bipolar')` in the column vocabulary the
  pipeline reads: `label`, `contact_label_1/2`, `type_1/2` (D/G/S from the BIDS
  `description`), `mni.x/y/z` (pair centroid, MNI152NLin6ASym), `distance`,
  `stein.region` / `wb.region` / `ind.region`, `hemisphere`.
- **A pair takes contact 1's atlas label, or contact 2's when contact 1 has
  none.** cmlreaders' pairs.json looked the atlas up at the pair midpoint;
  BIDS has no pair-level lookup, so pairs straddling two regions can be
  labelled differently from the cmlreaders pipeline. Checked on R1111M FR1
  ses-0 against the cmlreaders test fixtures (2026-09-22): identical 141 pairs
  and per-contact labels (100/100), identical WORD onsets (288/288); Burke ROI
  agrees for 129/141 pairs, fine region for 110/141. The misses are the
  midpoint-vs-contact rule plus main's grid cascade taking `dk` where BIDS
  takes `ind`. Monopolar contacts carry the same labels in both, so a
  monopolar analysis regionalizes identically.
- Regionalization cascades trimmed to the three atlases BIDS carries:
  depths `stein → wb`, grid/strip `stein → ind`; `n/a` is a missing-label
  sentinel. The localization-table merge is gone.
- Removed: the cmlreaders loader + NFS retry, the legacy PTSA/`TalReader`
  path (`/data/eeg`), `get_localization`, `coerce_unhashable_event_fields`,
  `eeg_data_source` plumbing in the simulation hooks.

### Session preparation

- `data_check.py` rewritten: the session list is every (sub, exp, sess)
  OpenNeuro lists for FR1/catFR1/pyFR (`cml_data.session_dataframe`); the
  per-session check loads the pairs table and one WORD-locked clip through the
  production loader, drops flat pairs by scanning the recording, checks the
  phase path, writes `electrode_information/pairs/<ftag>_pairs.json`. Gone:
  the hard-coded 284-row cohort list, the curated `channels_to_drop` table
  (pairs ARE the recording's channels now), the `.mat`/`/protocols` readers,
  the mstime/eegoffset consistency check (BIDS `sample` is derived from onset).
- `load_events.py` and `match_events.py` (the stage-2 event writer this repo
  imported but did not contain) are ported from fc_methods_comparison_cml
  `riley-thesis`. `MatchedEvents.load_events` adapts BIDS events to the CML
  schema (`type`=trial_type, `mstime`=onset·1000, `eegoffset`=sample,
  `rectime`=response_time·1000, `trial`=list, one constant `eegfile`). The
  per-session `R1171M` unix-mstime fix was dropped (BIDS onsets are
  recording-relative); the cmlreaders `correct_retrieval_offsets` /
  `sort_eegfiles` corrections are assumed applied by the BIDS conversion.
- `prepare_sessions.py`: `--workers N`, `--subjects ...`; the new-experiment
  cohort (`--newexps`, ICatFR1/IFR1/RepFR1, `presentation_times.py`,
  `RepFR1_session_presentation_times.csv`, `newexps_overrides`) is removed.

### Dispatch

`fc.run_sessions(fn, items, desc, workers=1, ...)` (process pool) replaces
`run_sessions_local` / `run_sessions_cluster`; `run_compute_stage` and
`prepare_sessions` use it. `dask_client.py`, `cluster.py`, `scripts/env.sh`
and every `--local/--n-workers/--mem/--walltime/--cluster-log-dir/--cluster`
flag are gone; `--workers N` everywhere.

### Geometry

Every contact is in one space, so `pair_xyz_lead(dfrow)` → `(xyz, lead)` from
`mni.*`, and `pair_distance_mask` no longer takes `is_depth` /
`drop_cross_type` (the flag is removed from all scripts). The cmlreaders
contacts-table MNI cache and the duplicate geometry code in
`plot_phase_conn_distance.py` are deleted.

### Config

`config.yaml` / `project_paths.py` reduced to what this repo reads:
`paths.{scratch_dir, exclusion_counts, unrecoverable_sessions_csv}`, bands,
behavior lists, signal-processing constants, `computation_metrics`. Removed:
`python_env_path`, `cluster`, `compute_profiles`, `smokescreen*`,
`newexps_overrides`, `case_study`, the thesis results/intermediate path tree,
`seed_offset`, `behaviors:` per-beh block, and the corresponding constants.
Dependencies are pinned in `requirements.txt` (ptsa and bidsreader from
GitHub).

### Known gaps

- OpenNeuro coverage vs the rhino cohort (2026-09-21): FR1 149/153 subjects,
  catFR1 31/33, pyFR 42/80. The conversion fix is on rhino, not yet released.
- `ri` events fall below the 10-event threshold on many sessions (unchanged
  behaviour; `ri` is not a required behavior).
- `visualize_phase_connectivity.ipynb` still builds `loc`/`mon` dfrows.
- The `aec_envelope` simulation DGP imports `fc_aec_dgp`, which this repo
  never contained (it lives in fc_methods_comparison_cml `sim/`).

---

## 2026-08-17 — High-gamma band + Morlet FC mode

Adds a high-gamma analysis band and an alternate Morlet (time-resolved)
spectral estimator alongside the existing multitaper path.

**Design rule throughout: default behaviour is unchanged.** Every existing
result stays bit-identical and no pickle needs recomputing or migrating. The new
band, the new estimator, and harmonic notching are all opt-in.

---

### 1. Parameters (config/config.yaml)

| key | value | new? | effect when left at default |
|---|---|---|---|
| `bands.high_gamma` | `[70, 150]` | new | none — nothing selects it unless `--band high_gamma` |
| `notch_harmonics_up_to_hz` | `null` | new | none — fundamental-only notch, historical behaviour |
| `fc_mode` | `multitaper` | new | none — existing estimator |
| `cwt_fnum` | `6` | new | Morlet path only |
| `cwt_morlet_reps` | `5` | new | Morlet path only (Rao's value) |
| `cwt_buffer_n_sigma` | `4.0` | new | Morlet path only |

Unchanged: `gamma [70,110]`, `resample_hz 500`, `min_sample_rate_hz 499`,
`mirror_buffer_ms 1000`, `real_data_buffer_ms 50`, `mt_bandwidth 2`.

### Why `high_gamma` is separate from `gamma`

`gamma [70,110]` is a **filter** band — the PAC amplitude band and the AEC/AEC-c
pre-envelope filter — whose 110 Hz upper edge is set by MNE stopband fitting,
not by scientific interest. `high_gamma [70,150]` is an **analysis** band.
Merging them would silently change existing AEC/PAC results.

### Why 70–150

Field-standard high gamma, and the range Long et al. use on this same dataset,
so the two are directly comparable. Geometric centre `sqrt(70*150)` = 102.5 Hz —
i.e. centred at ~100 Hz on the log-frequency axis Morlet wavelets tile, which is
the only sense in which "centred at 100" is meaningful for a wavelet bank.

Nyquist is safe with no new guard: sessions below 499 Hz native are already
excluded upstream (`min_sample_rate_hz` → `data_check` reason `sub_500hz`), so
every retained session has Nyquist ≥ 249.5 Hz and the resample to 500 Hz never
upsamples.

### Why `cwt_buffer_n_sigma = 4.0`

Buffer is derived from the Morlet Gaussian width, σ = `morlet_reps/(2πf)`, so it
scales with frequency instead of being a fixed constant (5σ is 995 ms at 4 Hz
but only 57 ms at 70 Hz — no single number serves both).

At 4σ and fmin = 70 Hz the requirement is **45.5 ms/side**, which the existing
`real_data_buffer_ms: 50` already covers, leaving **9.1 ms of clean separation**
between the buffered `word_on` pre/post windows. So `PREPOST_SPEC` needs no
change: `pre_win (-700,-100)` and `post_win (0,600)` both stay 600 ms,
`equalize_time_length` stays a no-op, and stimulus onset stays at t = 0.

5σ would have needed 113.7 ms against a 100 ms gap, forcing a window change, in
exchange for reducing leakage from 1.1e-7 to 1.4e-11 **in power**. Not worth
23 ms of analysis window.

---

### 2. Methods changed

### `helper.notch_filter` — new `harmonics_up_to_hz` argument

Signature: `notch_filter(eeg, sub, harmonics_up_to_hz=None)`. Backwards
compatible; `None` = fundamental only = historical behaviour.

When set, notches harmonics at ±2 Hz each — the convention already assumed by
`wavelet.Wavelet.MakePlot`, which plots 58–62, 118–122, 178–182.

**Notching is site-uniform.** It removes the union of *both* mains standards'
harmonics for every subject, not just that subject's own fundamental. Per-site
notching would give subjects different effective passbands: a Freiburg (50 Hz)
subject loses 100 and 150 Hz, a US (60 Hz) subject loses 120 Hz. Within
`high_gamma` that is 10% vs 5% of the band, **and the German hole sits at the
band centre**. Group-averaging such subjects means averaging different frequency
content, so an apparent site difference could be a filtering artifact. The union
costs 15% of the band for everyone, but every subject contributes an identical
passband.

For `high_gamma`, set `notch_harmonics_up_to_hz: 150` — that notches exactly
{100, 120, 150}. Larger values only add pointless passes at 180/200 Hz.

> **Not setting this while analysing high gamma is a silent failure mode**: the
> band fills with mains, worst at Freiburg where contamination lands on 100 Hz.

### `helper.get_beh_eeg`

Now passes `NOTCH_HARMONICS_UP_TO_HZ` through to `notch_filter`. No behavioural
change at the default `null`.

### `build_roi_power.run_sess_power` — Morlet estimator

Takes `fc_mode`. `cwt_morlet` selects `band_power_morlet` instead of
`band_power`; both return raw per-event per-channel power `(E, C)`, so they are
drop-in alternatives and the caller's log10 / Cohen's d chain is untouched.

Morlet needs real data either side of the analysis window, so `run_sess_power`
widens each window by `fc.morlet_buffer_ms` before slicing and
`band_power_morlet` trims it back off after the transform — no analysed sample
sits inside the wavelet edge artifact. `n_win_samples` records the *trimmed*
length. For prepost behaviours it also calls `assert_windows_separable`, so a
band or window change that would make the buffered pre/post windows overlap
fails at compute time rather than silently.

`band_power_morlet` uses the same PTSA primitive as `helper.get_power`
(`MorletWaveletFilter(width=5, output='power', complete=True)`) but does not
call it: `get_power` is specialised for Rao's 1000 ms mirror-buffered theta
clips, hardcoding a 1000 ms buffer clip that would erase a 600 ms word_on
window, and folding in log10 and an across-event z-score at the wrong point in
this chain.

Verified on synthetic data with a 100 Hz signal in one channel of six: both
estimators pick the same channel, signal/noise power ratio 26.6 (multitaper)
vs 23.3 (Morlet).

### `fc_comparison_functions.compute_spectral_fc` — cwt grid default

The `cwt_morlet` branch previously defaulted to `np.arange(fmin, fmax+0.5, 1)`.
For 70–150 that is **81 wavelets**, but a 5-rep Morlet at 100 Hz has ~40 Hz
half-power bandwidth, so those estimates are near-duplicates. Now defaults to a
log-spaced bank via `default_cwt_freqs`.

Measured on 16 channels × 40 epochs: **2.83 s → 0.13 s (22× faster)**, grid
`[70, 81.5, 95, 110.6, 128.8, 150]`.

Only affects the `cwt_morlet` branch, which nothing called before this work. The
`multitaper` branch is untouched.

---

### `build_roi_power` — latency bins (`time_bin_ms`, default 50)

`band_power_morlet(..., bin_ms=)` returns `(E, C, n_bins)` instead of `(E, C)`,
binning with Rao's own `helper.timebin_power_timeseries` — the function behind
his 200 ms epochs. `run_sess_power` then computes Cohen's d independently within
each bin, giving **d as a function of latency**, stored as `cohens_d_bins`
`(n_ch, n_bins)` plus `bin_centers_ms`.

50 ms rather than Rao's 200 ms because high gamma affords far more temporal
resolution: a 5-cycle Morlet at 70 Hz has sigma = 11.4 ms, so 50 ms bins are
~4.4 sigma apart and effectively independent. (Rao's 200 ms bins at 3 Hz sit
INSIDE a 265 ms sigma, so his adjacent bins are correlated — his time axis is
oversampled relative to what his wavelets resolve.) 50 ms also separates the
early latencies Long et al. report on this dataset — bitmap 40 ms,
orthographic neighbourhood 80 ms, word frequency 120 ms — which 100 ms bins
would merge. Floor is ~25 ms (~2 sigma at 70 Hz).

Bins apply only to `fc_mode: cwt_morlet` AND prepost behaviours; multitaper has
no time axis, and en/rm contrast two event groups in one window so there is no
pre/post time course to resolve. 600 ms windows divide evenly into 12 bins.

Verified: a 100 Hz burst planted at 200-300 ms peaks in the bins centred at 225
and 275 ms; the noise channel stays flat.

Pickles also gained provenance keys (`fc_mode`, `notch_harmonics_up_to_hz`,
`morlet_reps`, `cwt_fnum`, `buffer_ms`) so a consumer can tell what produced a
file without inferring it from the path.

### `--min-electrodes` unified at 3

`build_roi_power` default 2 -> 3, matching `build_roi_synchrony`, so the power
and synchrony figures admit the same (subject, ROI) cells. Plot-stage only — no
recompute. `build_power_synchrony` keeps `MIN_ELEC_CORR = 5`, which is a
different quantity (electrodes entering a Pearson r across electrodes; an r from
3 points is not meaningful).

### Morlet figures land in `<out-dir>/cwt_morlet/`

All three scripts, mirroring the `<band>__cwt_morlet` split on the compute side.

### Latency stage in `build_roi_power` (Morlet only)

`run_latency_stage` -> `collect_bin_table` -> `subject_roi_bin_table` ->
`bin_stats` -> `plot_time_course`. Reads `cohens_d_bins` from the pickles and
emits a small-multiple figure (one panel per ROI, mean d +/- SEM vs latency,
significant bins marked) plus two CSVs. Runs automatically in the plot stage
when `--fc-mode cwt_morlet`; no-ops for multitaper. Applies the same
`--min-electrodes` floor as the collapsed figure so both describe the same cells.

`bin_stats` is a one-sample t per (ROI, bin) with BH-FDR over the whole
ROI x bin grid. **Interim**: the principled test for a contiguous time grid is
TFCE (Rao's `tfce.py`, the machinery behind his Fig 4B). BH ignores that
neighbouring bins form clusters rather than independent tests, so it will miss
temporally extended-but-weak effects TFCE would find.

Verified on synthetic data: an effect planted at 100-250 ms in one ROI is
recovered at exactly the 125/175/225 ms bins, with no false positives elsewhere.

### Morlet grid no longer lands on a notched line (`nudge_off_notches`)

`default_cwt_freqs(70,150,6)` ended at exactly **150.0 Hz** -- the German 3rd
harmonic, which `notch_harmonics_up_to_hz: 150` removes. One of six wavelets was
centred on a hole. Offending centres now shift to the stopband edge
(150.0 -> 153.5). Any Morlet pickle computed before this is stale.

### Windowed multitaper latency axis (`band_power_windowed`, `mt_window_ms`)

Gives the multitaper path a time course on the SAME bin centres as Morlet, so
the two are comparable bin for bin. Windows are centred on Morlet's centres and
stepped by `time_bin_ms`, not by `mt_window_ms`.

Two constraints surfaced while building it, both consequences of multitaper
being CONSTANT-BANDWIDTH (unlike Morlet, constant-Q — see the buffer note):

1. **50 ms windows are unusable.** Resolution is 1/T and the NW=2 half-bandwidth
   is NW/T: at T=50 ms that is 20 Hz resolution and a 40 Hz half-bandwidth, so
   the whole 80 Hz band collapses into ~one resolution element. 200 ms is the
   floor. Multitaper gains no time resolution by moving up in frequency.
2. **`mt_bandwidth: 2` is illegal below ~500 ms.** MNE's normalized half-
   bandwidth is NW = T*bw/2 and it refuses NW < 0.5; 2 Hz over 200 ms is NW=0.2.
   Fixed by holding the TIME-BANDWIDTH PRODUCT constant instead of the bandwidth
   in Hz: `bw_win = bandwidth * dur/window`, so 600 ms @ 2 Hz and 200 ms @ 6 Hz
   are both NW = 0.6 — identical taper count and estimator shape. The spectral
   smoothing in Hz necessarily differs (2 -> 6); that is the uncertainty
   principle, not a choice.

Bins whose full window falls outside the analysis window return NaN rather than
being truncated (a shorter window has different resolution and is not comparable
to its neighbours). For 600 ms / 50 ms bins / 200 ms windows that is centres 25,
75, 525, 575 — leaving 8 comparable centres at 125-475 ms.

**Effective temporal resolution is `mt_window_ms` (200 ms), not `time_bin_ms`
(50 ms).** The axis is merely SAMPLED at 50 ms; consecutive windows overlap by
150 ms. Morlet's 50 ms bins are ~4.4 sigma apart at 70 Hz and effectively
independent; these are not. This is a cross-check on the Morlet time course, not
an equivalent measurement, and must be labelled as such wherever both are shown.

Verified: a 100 ms burst at 200-300 ms peaks at the 275 ms bin under BOTH
estimators, but Morlet confines it to two bins (SNR 44/51) while multitaper
spreads it over ~250 ms (SNR 4/17/31/31/18/5) and loses peak amplitude to
within-window averaging.

### 3. Additions

All delegate to existing modules — no new Morlet math.

| function | file | delegates to |
|---|---|---|
| `band_dirname(band, fc_mode)` | `fc_comparison_functions.py` | — |
| `default_cwt_freqs(fmin, fmax, fnum, morlet_reps)` | `fc_comparison_functions.py` | `wavelet.Wavelet(...).freqs` |
| `morlet_buffer_ms(fmin, morlet_reps, n_sigma)` | `fc_comparison_functions.py` | `Wavelet.get_morlet_width` |
| `assert_windows_separable(pre_win, post_win, fmin, ...)` | `fc_comparison_functions.py` | `morlet_buffer_ms` |
| `FC_MODES` tuple | `fc_comparison_functions.py` | — |
| `band_power_morlet(seg, sf, fmin, fmax, fnum, morlet_reps, buf_samples)` | `build_roi_power.py` | PTSA `MorletWaveletFilter` |
| `NOTCH_HARMONICS_UP_TO_HZ`, `FC_MODE`, `CWT_FNUM`, `CWT_MORLET_REPS`, `CWT_BUFFER_N_SIGMA` | `project_paths.py` | config |

`assert_windows_separable` raises (not warns) when the buffered pre/post windows
would overlap: overlap means the same samples inform both arms of a contrast,
manufacturing agreement between conditions that must be independent.

### Output-path separation

`band_dirname` keeps the two estimators from colliding:

```
multitaper  ->  <beh>/fc_mats/<cond>/high_gamma/
cwt_morlet  ->  <beh>/fc_mats/<cond>/high_gamma__cwt_morlet/
```

Multitaper returns the bare band name, so **every existing path and pickle stays
where it is**. Without this, a Morlet run would silently overwrite multitaper
results for the same band/behaviour.

---

### 4. CLI

`--fc-mode {multitaper,cwt_morlet}` added to all three scripts, defaulting to
`fc.FC_MODE` (config).

| script | role | status |
|---|---|---|
| `build_roi_power.py` | computes power | **fully live in both modes** |
| `build_roi_synchrony.py` | computes phase FC | flag parsed; `cwt_morlet` **exits with an error** (estimator not threaded) |
| `build_power_synchrony.py` | **consumes only** | flag selects which pickles to read |

`build_roi_synchrony.py` refuses `--fc-mode cwt_morlet` rather than running:
its estimator is still multitaper, so the run would write multitaper numbers
into a directory labelled `cwt_morlet`. Failing loudly beats a mislabelled
pickle.

Note `build_power_synchrony.py` computes nothing — its lines 100/109 are
`SystemExit` messages instructing you to run the other two first. Its flag is a
path *selector*, not a compute option.

---

### 5. Verified

- config loads; `high_gamma (70,150)`, `gamma (70,110)` intact
- `fc_comparison_functions` and `helper` import cleanly
- both FC modes run on synthetic data with a known 100 Hz phase-locked pair:
  multitaper PPC 0.132, Morlet PPC 0.754, both ≈0 for the noise pair
  (Morlet higher because multitaper averages the whole band including
  noise-only frequencies)
- 22× grid speedup measured
- `assert_windows_separable` fires correctly at 5σ, passes at 4σ
- all three `--help` invocations parse

Not run: anything against real EEG (no data access from this session).

---

### 6. Not done yet

1. **`fc_mode` threading for the FC path.** Done for power
   (`run_sess_power`); still needed for `build_roi_synchrony.py`, which goes
   two levels deeper: `run_sess_phase_fc` → `compute_session_fc` →
   `compute_prepost_separate`, plus its five `cond_dir` call sites. Guarded
   until then.
2. **Time-resolved output** — the actual latency payload. `compute_spectral_fc`
   still does `out.mean(axis=-1)`. Planned route reuses existing functions:
   `helper.get_phase` → `helper.timebin_phase_timeseries(bin_width_ms=200)` →
   `fc.event_angles` → `fc.conn_from_z`, i.e. Rao's published method, with the
   200 ms binning supplying the latency axis.
3. **Vendor `tfce.py`** from `RaoEtal25JNeurosci` for time × frequency cluster
   statistics (his Fig. 4B machinery). `tfce_compute` is pure numpy +
   `skimage.measure.label`; the `xarray` / `BehavioralData` imports at the top of
   that module belong to other functions and do not need to come along.
4. **Call `assert_windows_separable` inside `compute_prepost_separate`** so a
   future window or band change cannot silently reintroduce buffer overlap.
