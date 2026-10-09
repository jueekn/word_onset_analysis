# On hold / to do

## Done in the 2026-10-06 recompute
- Single-pass compute: load each session once, write power + synchrony together.
- Notch the 120 Hz mains harmonic (`notch_harmonics_up_to_hz: 150`; inside the 70-150 Hz
  power band). data_check then stops excluding on notched harmonics: the 16 sessions
  excluded for 120 Hz come back; the 9 with non-mains 100 Hz peaks stay excluded.

## Modular scripts (done 2026-10-07)
- compute.py + plot_boxplots / plot_spectrum / plot_case_studies / plot_timecourse /
  plot_epoch_network / plot_latency; build_roi_synchrony.py removed; build_roi_power.py is
  Long-only legacy (delete with the Long code when you decide).
- High-gamma envelopes from the 2026-10-06 run are float16 (one electrode overflowed;
  plot_latency skips non-finite); compute.py now stores float32 (next recompute).

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
  Currently `subtract_erp: false`. Also covers the early high-gamma AEC-c network (0–200 ms),
  which may be a shared stimulus-locked envelope rise, and power vs synchrony consistency.

## Next: reliability analysis (final analysis of the paper, per PI)
- Main: test-retest across sessions (subjects with 2+ sessions; per-electrode effect in
  session 1 vs 2 for alpha / high-gamma power d, alpha ciPLV, high-gamma AEC-c, latency) +
  within- vs between-subject identifiability (own session-2 map vs other subjects', on
  matched regions). No recompute needed: per-session results are stored.
- Robustness (pick one): subject-subsampling replicability (how many patients to find each
  ROI effect again), or cross-cohort replication (RAM FR1/catFR1 vs pyFR).
- Supplement if asked: split-half (odd/even trials) per measure; power from the stored
  per-trial values, synchrony would need a recompute with trial splits.
- First step: count subjects with multiple sessions and overlapping electrodes.
- PAC: include in the reliability checks. Trial-shuffle control (12 subj, 2026-10-08):
  shuffled PAC rises as much at word onset as real PAC, so the increase is stimulus-locked
  (evoked); real > shuffled within electrodes by ~0.004 in both windows (steady coupling,
  not a word-onset change). Decide: drop PAC, or report shuffle-corrected.

## Spectrum heatmap: better significance + spectral tilt
- Present significance as contiguous frequency ranges, not per-bin stars: finer spectrum
  (1-2 Hz steps, ~2-150 Hz) + TFCE with sign-flip permutations across subjects (per ROI along
  frequency, corrected over ROIs), drawn as outlines on the heatmap.
- Spectral tilt (frontal: low up / beta down; occipital: alpha down / gamma up): test directly
  with specparam (aperiodic exponent change, word on vs off, per ROI) + periodic peaks
  (alpha) separately; small extra figure.
- Needs: compute.py to store per-electrode word-off / word-on spectra (trial-averaged, 1 Hz)
  and per-electrode d at that resolution -> next recompute.

## Latency figure refinement
- plot_latency.py: two panels (lobe-coloured bars, mean +/- SEM | magma heatmap); y labels
  repeated across panels -> share them; consider pairwise tests of the order (within
  subjects with both regions) before claiming a specific sequence.

## Open
- **Spectrum 64 Hz bin.** 60 Hz notch rings at the clip edges with the 50 ms buffer (null
  sim: d -0.75 at 60 Hz, ~56–64 Hz). Options: notch the continuous recording before cutting
  clips; 500 ms buffer for power; blank the bin. Alpha / high-gamma bands are unaffected.
- **Alpha in 200 ms epochs**: one taper, ±3 Hz smoothing; PI caveat on timing at low
  frequencies. Option: Hilbert phase binned per epoch (Rao / Solomon).
- **Frontal alpha power increase**: check evoked contribution and fine-region breakdown.
- **Synchrony locality**: split each electrode's partners into within- vs between-ROI (after
  the no-ERP synchrony run).
- **Synchrony simulations (running 2026-10-05)**: alpha_lag_frontal (ciPLV), hg_env_frontal
  (AEC-c), leak_frontal (zero-lag; both should stay 0), hg_null; 30 subjects ->
  figures/simulations/<tag>/. hg_ppc bursts sit at the clip centre (~800 ms in the 0-1600 ms
  post load), outside the 0-600 ms window -- fix or retire hg_ppc / the PPC recovery sweeps.

## Methods notes / deviations to report
- Region-pair floor 30 subjects (riley-thesis 100: only 3 ROI pairs reach it here).
- Localization caveats (Riley): grids/strips before ~R1350 not brain-shift corrected;
  FreeSurfer 5.3; provenance of the "mni" coordinates.
- 16 sessions excluded for mains harmonics > 5x; R1171M_FR1_2 fails event extraction.

## Cleanup
- `match_events.py`: encoding/retrieval matching code now unused.
- ~30 GB raw EEG from the null simulations under `bids_data/` (delete by hand).
- Nothing committed since 8aa205c.
