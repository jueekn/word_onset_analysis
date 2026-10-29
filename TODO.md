# On hold / to do

## On hold
- Done (2026-10-21): region labels for depth contacts without a volumetric label now come from
  Riley's imaging-label database (recover_labels.py, copied to <scratch>/imaging_label_recovery;
  white matter / ventricles stay unlabelled); the MNI-neighbour labels only fill pairs the
  database lacks (13%). Before: ~3,600 neighbour-labelled pairs were white matter per the
  database. All region-based figures replotted.
- **Time-locked (evoked) component in synchrony.** PI: ERP subtraction is not the answer;
  review how the literature handles stimulus-locked effects in phase / envelope coupling.
  Currently `subtract_erp: false`. Also covers the early high-gamma AEC-c hubs (0–200 ms,
  within-region pairs), which may be a shared stimulus-locked envelope rise.

## Paper: references to check
- Edwards et al. 2009: confirm it supports "high gamma is broadband, not an oscillation" (Power Methods).
- Azarion et al. 2014: confirm it is the right citation for volumetric (depth) vs surface (grid/strip) labels.
- Lenth_2025 (emmeans): cited as R package v1.10.3 (installed version); update the year/version if re-run.
- Supplementary table (ADD): anatomical label -> 12-region mapping (region_to_burke_lobe.csv).

## Reliability (final analysis of the paper, per PI)
- Done (2026-10-14, plot_reliability.py): test-retest as riley-thesis
  compute_retest_reliability.py (random session halves, >= 2 sessions per half, >= 10 regions,
  Pearson, medians), per anatomical region: power d (alpha 0.63, high gamma 0.79) and synchrony
  node strength (alpha ciPLV 0.16, high-gamma AEC-c 0.13); 91 subjects.
- Identifiability (own half-B profile vs other subjects'): a first try confounded stable
  electrode placement with a subject trait -- needs matched regions.
- Robustness (pick one): subject-subsampling replicability (how many patients to find each
  ROI effect again), or cross-cohort replication (RAM FR1/catFR1 vs pyFR).
- Supplement if asked: split-half (odd/even trials) per measure; power from the stored
  per-trial values, synchrony would need a recompute with trial splits.
- **Individual differences (later)**: subject recall rate vs onset measures (alpha d, high-gamma
  d, latency per lobe). Decide how to handle experiment (FR1 / catFR1 / pyFR recall rates
  differ) and electrode coverage before testing.

## Latency
- Done (2026-10-19): pairwise lobe tests, mixed model with fine regions as random effects
  (mixed_models.R latency). Next: show them on the latency figure (brackets or letters).

## Spectral tilt
- Frontal: low up / beta down; occipital: alpha down / gamma up. Test directly with specparam
  (aperiodic exponent change, word on vs off, per ROI) + periodic peaks (alpha) separately;
  small extra figure. Needs compute.py to store per-electrode word-off / word-on spectra
  (trial-averaged, 1 Hz) -> next recompute.

## Recall decoding (plot_decoding.py)
- Current: encoding contrast, high gamma per lobe x 100 ms window, >= 5 electrodes per lobe,
  stars AUC > 0.5 (permutation chance 0.493 reported only).
- **Hyperparameter tuning**: nested CV over the L2 strength C per subject (now fixed at the
  lab's 7.2e-4, ram_utils / Ezzyat et al. 2017).
- Check whether the small above-chance recall information at word onset (frontal / parietal,
  vs the permutation level) is serial-position / encoding-state drift (add serial position).

## Open
- **Spectrum 64 Hz bin**: the 60 Hz notch ringing came from the 50 ms buffer; recheck the
  64 Hz bin with the 500 ms buffer before dropping this item.
- **Alpha in 200 ms epochs**: one taper, ±3 Hz smoothing; PI caveat on timing at low
  frequencies. Option: Hilbert phase binned per epoch (Rao / Solomon).
- **Frontal alpha power increase**: check evoked contribution and fine-region breakdown.

## Methods notes / deviations to report
- Region-pair floor 100 subjects (MIN_SUBJECTS_PAIR, as riley-thesis).
- Synchrony includes within-region pairs (contact-sharing pairs excluded).
- Localization caveats (Riley): grids/strips before ~R1350 not brain-shift corrected;
  FreeSurfer 5.3; provenance of the "mni" coordinates.
- 9 sessions excluded for non-mains 100 Hz peaks; R1171M_FR1_2 fails event extraction.

## Cleanup
- `match_events.py`: encoding/retrieval matching code now unused.
- ~30 GB raw EEG from the null simulations under `bids_data/` (delete by hand).
