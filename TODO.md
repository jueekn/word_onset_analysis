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
  Currently `subtract_erp: false`. Also covers the early high-gamma AEC-c network (0–200 ms),
  which may be a shared stimulus-locked envelope rise, and power vs synchrony consistency.

## Open
- **Spectrum 64 Hz bin.** 60 Hz notch rings at the clip edges with the 50 ms buffer (null
  sim: d -0.75 at 60 Hz, ~56–64 Hz). Options: notch the continuous recording before cutting
  clips; 500 ms buffer for power; blank the bin. Alpha / high-gamma bands are unaffected.
- **Alpha in 200 ms epochs**: one taper, ±3 Hz smoothing; PI caveat on timing at low
  frequencies. Option: Hilbert phase binned per epoch (Rao / Solomon).
- **Frontal alpha power increase**: check theta (6–8 Hz) vs alpha, evoked contribution, and
  fine-region breakdown.
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
