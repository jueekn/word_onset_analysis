# On hold / to do

## On hold
- **Region labels for depth contacts (151 subjects).** The OpenNeuro BIDS release has no
  whole-brain volumetric label (`wb` / rhino `atlases.whole_brain` / `mni.region`) for the
  depth contacts of 151 FR1/catFR1 subjects (129 with no depth label at all, 22 with only
  stein/das MTL labels). Ask Riley whether the rhino localization files have them.
  *Interim (2026-10-03):* `fc.mni_neighbour_labels` labels these contacts from expert-labelled
  depth contacts of other subjects within 5 mm in MNI space (>= 3, majority vote);
  leave-one-subject-out agreement 89% (12 ROIs) / 79% (fine regions), 70% coverage.
- **Time-locked (evoked) component in synchrony.** PI: ERP subtraction is not the answer;
  review how the literature handles stimulus-locked effects in phase / envelope coupling.
  Currently `subtract_erp: false`. Also covers the early high-gamma AEC-c hubs (0–200 ms,
  within-region pairs), which may be a shared stimulus-locked envelope rise.

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
