"""
build_roi_power.py

Local (single-electrode) spectral power in the same 12 Burke ROIs used by the
phase-connectivity analyses -- the power counterpart to build_roi_synchrony.py.

Same events, same windows, same band, same multitaper front end as the phase
metrics; the only change is that each electrode is treated on its own instead
of as one endpoint of a pair:

    per event, per electrode:  multitaper PSD over the window, averaged in band
    per electrode:             log10, then Cohen's d of hi vs lo ACROSS events
    per subject, per ROI:      mean of d over that ROI's electrodes
    across subjects:           box plot (one box per ROI)

Two stages:

  compute  one pickle per session (dask/SLURM by default)
             <save_root>/<beh>/power/<band>/<ftag>_power.pkl
             {"sid", "labels", "reg_full", "log10_lo", "log10_hi",
              "cohens_d", "n_events"}
  plot     aggregate those pickles -> per-subject-per-ROI table -> box plots

Relationship to Rao et al. 2025 (J Neurosci)
--------------------------------------------
The per-electrode summary is Cohen's d of log10 power, region-averaged, matching
that paper's power pipeline (helper.get_power -> comp_elpomx ->
regionalize_electrode_powers). Specifically:

  adopted    log10 BEFORE the contrast; `helper.cohens_d` verbatim (pooled-SD,
             the same function vendored in this repo); the effect size -- not raw
             power -- as the per-electrode quantity; mean of d over the
             electrodes in a region as the region value.
  not adopted  the Morlet (width=5, discrete 3-8 Hz) front end. This analysis
             keeps the multitaper estimator of the phase metrics it accompanies,
             so power and connectivity describe the same band with the same
             spectral estimator.
  no-op here  get_power's z-score across events. It is one (mean, std) per
             channel/frequency applied to both samples, and Cohen's d is
             invariant to an affine transform -- so with a single band-averaged
             value per event it changes nothing. (In Rao's pipeline it does
             matter, because it reweights the frequencies before the band
             average; here the band average happens inside the multitaper.)

Note that Rao's contrast (recalled vs not-recalled) is between INDEPENDENT event
groups, while pre vs post word onset are the same trials at two windows. The
pooled-SD formula is kept for consistency; it ignores that pairing and is
therefore somewhat conservative. A paired d_z would be mean(diff)/sd(diff).

Measures plotted. Raw log-power is not comparable across subjects (amplifier
gain, referencing, and coverage differ); the two window-contrast measures are:

  cohens_d       effect size of hi vs lo log10 power, across events, per
                 electrode. Unitless and gain-invariant -- the panel to read for
                 a task effect, and the Rao-consistent one.
  z_lo / z_hi    log10 power in the lo / hi condition, z-scored ACROSS that
                 subject's electrodes (descriptive, NOT Rao's across-events
                 z-score). Removes the per-subject gain offset, so the box plot
                 reads "which ROIs carry relatively more power in this band".
  change_dB      10 * (log10 P_hi - log10 P_lo) -- the same contrast as
                 cohens_d but in dB rather than SD units. Written to the CSVs;
                 plot it with `--measures change_dB`.

Every panel gets a per-ROI one-sample t of the subject values vs 0, FDR-corrected
across the 12 ROIs (fc.roi_stats). What that test IS depends on the measure:
on the contrast measures (cohens_d, change_dB) it is algebraically the PAIRED t;
on the level measures (z_lo, z_hi) it asks whether the ROI departs from that
subject's whole-brain mean, which is where 0 sits after the z-score.

Electrode -> ROI uses the canonical `regionalize_electrodes_by_type` label
(volumetric cascade for depths, surface for grid/strip) mapped through
region_to_burke_lobe.csv + hemisphere, identical to the connectivity scripts.

Usage:
    python build_roi_power.py                         # compute (cluster) + plot
    python build_roi_power.py --local --n-sessions 2  # local smoke test
    python build_roi_power.py --stage plot            # replot from pickles
    python build_roi_power.py --beh voc               # vocalization contrast
"""
from __future__ import annotations

from typing import Any

import argparse
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc

# lo/hi are this behavior's two conditions: word_on/voc contrast two TIME
# WINDOWS of the same events, en/rm contrast two EVENT GROUPS in one window.
MEASURES = {
    "z_lo": "{lo}\n{band} power (z across electrodes)",
    "z_hi": "{hi}\n{band} power (z across electrodes)",
    "cohens_d": "{hi} vs {lo}\n{band} power (Cohen's d)",
    "change_dB": "{hi} - {lo}\n{band} power (dB)",
}

# change_dB is the same contrast as cohens_d in dB rather than SD units; it stays
# out of the default figure (it would be a redundant panel) but is written to the
# CSVs and can be plotted with --measures.
DEFAULT_MEASURES = ("z_lo", "z_hi", "cohens_d")


def panel_labels(beh, band):
    c = fc.contrast(beh)
    return {k: v.format(lo=c["lo_label"], hi=c["hi_label"], band=band)
            for k, v in MEASURES.items()}


def power_dir(root, beh, band, fc_mode="multitaper"):
    return Path(root) / beh / "power" / fc.band_dirname(band, fc_mode)


# ------------------------------ compute stage --------------------------------
def _word_locked_eeg(dfrow: pd.Series, beh: str, win: tuple[float, float],
                     buffer_ms: float = 0.0) -> Any:
    """EEG for the POST-type events of `beh`, loaded over one window spanning
    both the pre and post analysis windows.

    Same load as `fc.session_eeg` but with `beh` as a parameter. For the PREPOST
    behaviors the PRE_* events are row-for-row copies of the POST events
    (fc.compute_prepost_separate asserts this), so slicing one clip in time is
    equivalent to the two separate loads the FC pipeline does, and guarantees
    pre/post come from the same trials.
    """
    import fc_comparison_functions as fc

    ev = fc.load_events(dfrow, beh)
    if ev is None:
        return None
    post_mask = np.asarray(ev.attrs["mask"], bool)
    if not post_mask.any():
        return None
    words = ev[post_mask].reset_index(drop=True)
    words.attrs = dict(ev.attrs)
    # buffer_ms widens the LOADED clip beyond `win` so a downstream Morlet
    # buffer is taken from REAL ADJACENT data rather than eaten out of the
    # analysis window. get_beh_eeg defaults real_data_buffer_ms to 0, so
    # omitting this silently shortened every Morlet window by 2*buffer_ms.
    eeg, _ = fc.get_beh_eeg(dfrow, words, save=False, window=win,
                            real_data_buffer_ms=buffer_ms)
    return (np.asarray(eeg.data), np.asarray(eeg.time, float),
            float(eeg.samplerate))


def window_slice(data, t, win):
    """(n_events, n_channels, n_win) slice of `data` covering `win`, inclusive.

    Both endpoints are inclusive, so a window whose upper edge lands exactly on
    a sample keeps it. The pre and post windows can therefore differ by one
    sample (the clip's last sample is one step short of the requested end) --
    `fc.equalize_time_length` reconciles them before the spectra are taken.
    """
    m = (t >= win[0]) & (t <= win[1])
    if m.sum() < 2:
        raise ValueError(f"window {win} covers {int(m.sum())} samples")
    return data[:, :, m]


def band_power(seg, sf, fmin, fmax, bandwidth):
    """Per-event, per-channel multitaper power averaged over [fmin, fmax].

    Mirrors the multitaper front end of the phase metrics (same bandwidth,
    adaptive=False, low_bias=False) -- see fc.mt_event_angles -- but keeps the
    auto-spectrum instead of the cross-spectrum. Returns (n_events, n_channels).
    """
    from mne.time_frequency import psd_array_multitaper

    psds, _ = psd_array_multitaper(
        seg, sf, fmin=fmin, fmax=fmax, bandwidth=bandwidth,
        adaptive=False, low_bias=False, normalization="full", verbose=False)
    return np.asarray(psds).mean(-1)                       # (E, C)


def band_power_morlet(seg, sf, fmin, fmax, fnum, morlet_reps, buf_samples,
                      bin_ms=None):
    """Per-event, per-channel MORLET power averaged over the band. (E, C).

    Same contract as `band_power` (raw power, not log10 -- the caller logs), so
    the two are drop-in alternatives selected by --fc-mode.

    Uses the same PTSA primitive as `helper.get_power`
    (`MorletWaveletFilter(width=5, output='power', complete=True)`) over the
    log-spaced bank from `fc.default_cwt_freqs`. It does NOT call get_power
    itself, because that function is specialised for Rao's 1000 ms mirror-
    buffered theta clips: it hardcodes a 1000 ms buffer clip (which would erase
    a 600 ms word_on window outright), and folds in log10 and an across-event
    z-score at a point in the chain where this caller needs neither.

    `seg` must already include `buf_samples` of REAL adjacent data on each side;
    the wavelet is convolved over the buffered segment and the buffer is then
    trimmed, so no analysed sample sits inside the Morlet edge artifact.

    `bin_ms` (None = collapse time, as multitaper does) instead returns
    (E, C, n_bins) -- the latency axis. Binning uses Rao's own
    `helper.timebin_power_timeseries`, the function behind his 200 ms epochs.
    """
    from ptsa.data.timeseries import TimeSeries
    from ptsa.data.filters import MorletWaveletFilter
    import fc_comparison_functions as fc

    freqs = fc.default_cwt_freqs(fmin, fmax, fnum=fnum, morlet_reps=morlet_reps)
    ts = TimeSeries(
        np.asarray(seg, float),
        dims=('event', 'channel', 'time'),
        coords={'event': np.arange(seg.shape[0]),
                'channel': np.arange(seg.shape[1]),
                'time': np.arange(seg.shape[2]) / sf,
                'samplerate': sf},
    )
    power = MorletWaveletFilter(
        freqs=freqs, width=morlet_reps, output='power', complete=True,
    ).filter(timeseries=ts).transpose('event', 'channel', 'frequency', 'time')

    p = np.asarray(power)                                  # (E, C, F, T)
    if buf_samples:
        p = p[..., buf_samples:p.shape[-1] - buf_samples]
    if bin_ms:
        # Latency axis. Average over frequency first -> (E, C, T), then bin along
        # time with Rao's own binner (helper.timebin_power_timeseries, the same
        # function behind his 200 ms epochs) -> (E, C, n_bins).
        import helper
        return helper.timebin_power_timeseries(
            p.mean(axis=2), sf, bin_width_ms=int(bin_ms))
    return p.mean(axis=(2, 3))                             # (E, C)


def band_power_windowed(seg, sf, fmin, fmax, bandwidth, bin_ms, window_ms):
    """Sliding-window multitaper power at the SAME bin centres as the Morlet path.

    Returns (E, C, n_bins) so it is a drop-in peer of band_power_morlet's binned
    output, with an IDENTICAL time axis -- the windows are centred on Morlet's
    bin centres and stepped by bin_ms, not by window_ms. That is what makes the
    two estimators directly comparable bin for bin.

    Why the window is 200 ms and not 50: multitaper is CONSTANT-BANDWIDTH, so its
    frequency resolution is 1/T and the NW=2 half-bandwidth is NW/T regardless of
    centre frequency. At T = 50 ms that is 20 Hz resolution and a 40 Hz
    half-bandwidth -- the whole 80 Hz band becomes ~one resolution element. Unlike
    Morlet (constant-Q, sigma proportional to 1/f), multitaper gains no time
    resolution by moving up in frequency.

    CONSEQUENCE: consecutive windows overlap by (window_ms - bin_ms), so the
    EFFECTIVE temporal resolution is window_ms. The axis is merely SAMPLED at
    bin_ms. Morlet's 50 ms bins are ~4.4 sigma apart at 70 Hz and effectively
    independent; these are not. Read this as a cross-check on the Morlet time
    course, not as an equivalent measurement.

    Bins whose full window would fall outside the analysis window are returned as
    NaN rather than truncated -- a shorter window has different spectral
    resolution and would not be comparable to its neighbours. With a 600 ms
    window, 50 ms bins and a 200 ms window that is the first two and last two
    bins (centres 25, 75, 525, 575), leaving 8 comparable centres at 125-475 ms.
    """
    n_t = seg.shape[-1]
    dur_ms = 1000.0 * n_t / sf
    n_bins = int(round(dur_ms / float(bin_ms)))

    # Hold the TIME-BANDWIDTH PRODUCT constant, not the bandwidth in Hz.
    # MNE's `bandwidth` is the full bandwidth in Hz; the normalized half-
    # bandwidth is NW = T * bandwidth / 2, and MNE refuses NW < 0.5. The
    # configured mt_bandwidth=2 over the full 600 ms window is NW = 0.6; the
    # same 2 Hz over a 200 ms window would be NW = 0.2 and simply errors.
    #
    # Scaling bandwidth by (dur/window) keeps NW -- and therefore the taper
    # count and the estimator's shape -- identical to the collapsed run:
    #     600 ms @ 2 Hz  ->  NW = 0.6
    #     200 ms @ 6 Hz  ->  NW = 0.6
    # What cannot be preserved is the spectral smoothing in Hz (2 -> 6 Hz).
    # That is the uncertainty principle, not a choice: a shorter window buys
    # time resolution with frequency resolution. State it wherever these are
    # compared to the collapsed numbers.
    bw_win = float(bandwidth) * (dur_ms / float(window_ms))
    centers = (np.arange(n_bins) + 0.5) * float(bin_ms)
    half = float(window_ms) / 2.0
    w_n = int(round(float(window_ms) * sf / 1000.0))

    out = np.full(seg.shape[:2] + (n_bins,), np.nan)
    for k, c in enumerate(centers):
        if c - half < 0 or c + half > dur_ms:
            continue
        i0 = int(round((c - half) * sf / 1000.0))
        sl = seg[..., i0:i0 + w_n]
        if sl.shape[-1] != w_n:
            continue
        out[..., k] = band_power(sl, sf, fmin, fmax, bw_win)
    return out


def run_sess_power(
    dfrow: pd.Series, save_root: str, beh: str, band: str, root_dir: str,
    fc_mode: str = "multitaper", time_bin_ms: int | None = None,
    mt_window_ms: int | None = None,
) -> str:
    """Compute per-electrode band power for one session; write one pickle.

    Returns a short status string for the dispatcher's progress bar.
    """
    import helper
    import fc_comparison_functions as fc

    fc.root_dir = root_dir
    helper.root_dir = root_dir

    sid = f"{dfrow['sub']}_{dfrow['exp']}_{dfrow['sess']}"
    out_dir = str(power_dir(save_root, beh, band, fc_mode))
    out_path = join(out_dir, f"{fc.ftag(dfrow)}_power.pkl")

    if os.path.exists(out_path):
        try:
            cached = fc.load_pickle(out_path)
            # n_win_samples marks a pickle written after the pre/post window
            # equalization fix. Pickles without it carry the unequal-window bias
            # (pre estimated over one more sample than post), so treat them as
            # stale and recompute rather than silently reporting them cached.
            if all(k in cached for k in ("log10_lo", "log10_hi", "cohens_d",
                                         "reg_full", "n_win_samples")):
                return f"{sid}: cached"
        except Exception:
            pass

    fmin, fmax = fc.bands[band]
    prepost = beh in fc.PREPOST_SPEC

    # Decided before the EEG load: the Morlet edge buffer must be REAL data from
    # outside the analysis window, so the clip has to be widened at load time.
    morlet = fc_mode == "cwt_morlet"
    buf_ms = (fc.morlet_buffer_ms(fmin, morlet_reps=fc.CWT_MORLET_REPS,
                                  n_sigma=fc.CWT_BUFFER_N_SIGMA)
              if morlet else 0.0)

    if prepost:
        spec = fc.PREPOST_SPEC[beh]
        pre_win, post_win = spec["pre_win"], spec["post_win"]
        loaded = _word_locked_eeg(dfrow, beh, (pre_win[0], post_win[1]),
                                  buffer_ms=buf_ms)
        if loaded is None:
            return f"{sid}: no events ({beh})"
        data, t, sf = loaded
        ev_mask = None
    else:
        # en / rm: one window, two event groups (mask True = the `hi` arm),
        # exactly the split compute_session_fc makes for these behaviors.
        ev = fc.load_events(dfrow, beh)
        if ev is None:
            return f"{sid}: no events ({beh})"
        eeg, ev_mask = fc.get_beh_eeg(dfrow, ev, save=False)
        data = np.asarray(eeg.data)
        t = np.asarray(eeg.time, float)
        sf = float(eeg.samplerate)
        ev_mask = np.asarray(ev_mask, bool)
        pre_win = post_win = tuple(helper.beh_to_event_windows[beh])

    pairs = helper.get_pairs(dfrow)
    n_ch = data.shape[1]
    if pairs is None or len(pairs) != n_ch:
        raise ValueError(
            f"{sid}: len(pairs)={None if pairs is None else len(pairs)} but eeg "
            f"n_ch={n_ch}; channel order misaligned for region labels.")

    localization = helper.get_localization(dfrow)
    reg_full = np.asarray(
        helper.regionalize_electrodes_by_type(pairs, localization), dtype=object)
    labels = pairs["label"].astype(str).to_numpy()

    # Same multitaper half-bandwidth the phase metrics use (config.yaml
    # mt_bandwidth = 2). project_paths types it as optional (null => MNE's own
    # ~8*sfreq/n_times default); refuse to run in that case rather than pick a
    # bandwidth here, which would silently diverge from the connectivity front end.
    bandwidth = fc.MT_BANDWIDTH
    if bandwidth is None:
        raise ValueError(
            "config.yaml mt_bandwidth is unset; power and the phase metrics must "
            "share one multitaper bandwidth (default mt_bandwidth: 2)")

    # Morlet needs REAL data either side of the analysis window, because the
    # wavelet's support reaches outside it; multitaper does not. The buffer is
    # frequency dependent (fc.morlet_buffer_ms, from Wavelet.get_morlet_width),
    # so widen each window by it here and trim it back off after the transform.
    # The clip loaded by _word_locked_eeg already extends real_data_buffer_ms
    # beyond the union span, which covers this at the outer edges; the inner
    # edges sit inside the pre/post gap (fc.assert_windows_separable enforces
    # that the two buffered windows cannot meet).
    if morlet:
        if prepost:
            fc.assert_windows_separable(
                pre_win, post_win, fmin,
                morlet_reps=fc.CWT_MORLET_REPS, n_sigma=fc.CWT_BUFFER_N_SIGMA)
        nbuf = int(buf_ms * sf / 1000.0)
        widen = lambda w: (w[0] - buf_ms, w[1] + buf_ms)
    else:
        nbuf = 0
        widen = lambda w: w

    mt_win = int(mt_window_ms or fc.MT_WINDOW_MS)

    def _power(seg, bin_ms=None):
        if morlet:
            return band_power_morlet(seg, sf, fmin, fmax, fc.CWT_FNUM,
                                     fc.CWT_MORLET_REPS, nbuf, bin_ms=bin_ms)
        if bin_ms:
            # Windowed multitaper: same bin centres as Morlet, but each estimate
            # spans mt_win ms, so its effective resolution is mt_win, not bin_ms.
            return band_power_windowed(seg, sf, fmin, fmax, bandwidth,
                                       bin_ms, mt_win)
        return band_power(seg, sf, fmin, fmax, bandwidth)

    # Latency axis. Available to BOTH estimators now, but only for the prepost
    # behaviours -- en/rm contrast two EVENT GROUPS in one window, so there is no
    # pre/post time course to resolve.
    binned = bool(time_bin_ms and prepost)

    if prepost:
        # Inclusive time masks give the pre window one more sample than the post
        # window (the clip's last sample is one step short of the requested end),
        # so equalize them exactly as the FC path does before estimating spectra
        # -- otherwise the two are estimated over slightly different window
        # lengths, which shifts the frequency-bin centres and biases the contrast.
        lo_seg = window_slice(data, t, widen(pre_win))
        hi_seg = window_slice(data, t, widen(post_win))
        lo_seg, hi_seg = fc.equalize_time_length(lo_seg, hi_seg)
        n_win = lo_seg.shape[-1] - 2 * nbuf
        with np.errstate(divide="ignore", invalid="ignore"):
            lp_lo = np.log10(_power(lo_seg))
            lp_hi = np.log10(_power(hi_seg))
    else:
        # both arms share the loaded window, so nothing to equalize
        n_win = data.shape[-1] - 2 * nbuf
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.log10(_power(data))
        lp_lo, lp_hi = lp[~ev_mask], lp[ev_mask]

    lp_lo[~np.isfinite(lp_lo)] = np.nan
    lp_hi[~np.isfinite(lp_hi)] = np.nan

    # Per-bin contrast, (n_ch, n_bins). Same Cohen's d as the collapsed measure,
    # computed independently within each time bin -> d as a function of latency.
    d_bins = bin_centers = None
    if binned:
        # BASELINE = the WHOLE pre window, not the matching pre bin.
        #
        # Every post bin is contrasted against one baseline value per event, the
        # mean over the entire pre window. Bin-to-bin matching (post bin k vs pre
        # bin k) would be wrong three ways: the pairing is arbitrary (nothing
        # makes -700..-650 ms the right reference for 0..50 ms); it imports any
        # structure in the baseline -- anticipation late in the pre window, the
        # previous item's offset response early in it -- into the response time
        # course with opposite sign, manufacturing a trend; and it contrasts each
        # post bin against 1/B of the available baseline, so it is ~B times
        # noisier than it needs to be. A single baseline also matches what the
        # COLLAPSED figure in this same script does, and what Long et al. do
        # (z-score to the whole blank-screen period).
        #
        # The baseline is built from the SAME binned estimator as the response,
        # then averaged over bins -- not from the whole-window call. For the
        # windowed multitaper those differ (600 ms @ 2 Hz vs 100 ms @ 12 Hz, both
        # NW=0.6), and mixing them would put a constant offset on the whole time
        # course. nanmean also drops the multitaper edge bins that cannot fit a
        # full window.
        #
        # LOG FIRST, THEN AVERAGE OVER BINS. Averaging power and then logging is
        # NOT equivalent and is a real bug: the response arm is log(power of ONE
        # bin) while the baseline would be log(MEAN power over B bins), and since
        # the arithmetic mean of fluctuating positive power exceeds its geometric
        # mean, the baseline sits systematically high. Measured on null data
        # (pre and post statistically identical) that put d = -0.21 in EVERY bin
        # -- a constant downward shift of the whole time course. Logging first
        # puts both arms in the same units (log power of one bin) and returns
        # d = +0.01 under the null.
        with np.errstate(divide="ignore", invalid="ignore"):
            l_lo = np.log10(_power(lo_seg, bin_ms=time_bin_ms))   # (E, C, B) log
            b_hi = np.log10(_power(hi_seg, bin_ms=time_bin_ms))   # (E, C, B) log
            l_lo[~np.isfinite(l_lo)] = np.nan
            base = np.nanmean(l_lo, axis=-1)                      # (E, C)    log
        b_hi[~np.isfinite(b_hi)] = np.nan
        base[~np.isfinite(base)] = np.nan
        nb = b_hi.shape[-1]
        d_bins = np.full((n_ch, nb), np.nan)
        for c in range(n_ch):
            bb = base[np.isfinite(base[:, c]), c]     # one value per event
            if bb.size <= 1:
                continue
            for k in range(nb):
                aa = b_hi[np.isfinite(b_hi[:, c, k]), c, k]
                if aa.size > 1:
                    d_bins[c, k] = helper.cohens_d(aa, bb)
        # Centres are ms into the POST window; the baseline is a single scalar
        # per event, so the axis now refers unambiguously to post-onset time.
        bin_centers = (np.arange(nb) + 0.5) * float(time_bin_ms)

    # Per-electrode effect size, Rao et al. 2025's per-electrode quantity:
    # `helper.cohens_d` (pooled SD) on log10 power, hi vs lo, across events.
    # Done channel by channel so a degenerate event drops only its own channel's
    # contribution instead of the whole electrode (helper.cohens_d is not
    # nan-aware); in practice every event is finite and this is one call's worth
    # of arithmetic per channel.
    d = np.full(n_ch, np.nan)
    for c in range(n_ch):
        a = lp_hi[np.isfinite(lp_hi[:, c]), c]
        b = lp_lo[np.isfinite(lp_lo[:, c]), c]
        if a.size >= 2 and b.size >= 2:
            d[c] = float(helper.cohens_d(a, b))

    out: dict[str, Any] = {
        "sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"]),
                int(dfrow["loc"]), int(dfrow["mon"])),
        "labels": labels,
        "reg_full": reg_full,
        "log10_lo": np.nanmean(lp_lo, axis=0),
        "log10_hi": np.nanmean(lp_hi, axis=0),
        "cohens_d": d,
        "n_events": int(data.shape[0]),
        "n_lo": int(lp_lo.shape[0]), "n_hi": int(lp_hi.shape[0]),
        "n_win_samples": int(n_win),
        "beh": beh, "band": band, "fmin": fmin, "fmax": fmax,
        "lo_win": pre_win, "hi_win": post_win,
        "mt_bandwidth": bandwidth,
        # Provenance: what actually produced these numbers. Lets a consumer (or
        # a future cache check) tell a Morlet pickle from a multitaper one, and
        # a notched run from an un-notched one, without guessing from the path.
        "fc_mode": fc_mode,
        "notch_harmonics_up_to_hz": fc.NOTCH_HARMONICS_UP_TO_HZ,
        "morlet_reps": fc.CWT_MORLET_REPS if morlet else None,
        "cwt_fnum": fc.CWT_FNUM if morlet else None,
        "buffer_ms": (buf_ms if morlet else fc.REAL_DATA_BUFFER_MS),
        # Latency axis (Morlet + prepost only; None otherwise).
        "cohens_d_bins": d_bins,          # (n_ch, n_bins)
        "bin_centers_ms": bin_centers,    # ms into each window
        "time_bin_ms": int(time_bin_ms) if binned else None,
        "mt_window_ms": (mt_win if (binned and not morlet) else None),
    }
    os.makedirs(out_dir, exist_ok=True)
    fc.save_pickle(out_path, out)
    return f"{sid}: {n_ch} elec, {out['n_lo']}+{out['n_hi']} events"


# ------------------------------- plot stage ----------------------------------
def collect_electrode_table(save_root, beh, band, lobe_of, n_sessions=None,
                            fc_mode="multitaper"):
    """Walk the per-session pickles -> tidy per-(subject, electrode) table.

    Sessions of the same subject are averaged per electrode LABEL (in log10
    units, i.e. a geometric mean over sessions) before anything else, so a
    subject with 4 sessions does not outweigh one with 1.
    """
    d = power_dir(save_root, beh, band, fc_mode)
    files = sorted(d.glob("*_power.pkl"))
    if n_sessions is not None:
        files = files[:n_sessions]
    if not files:
        raise SystemExit(
            f"no pickles in {d}\nrun the compute stage first: "
            f"python build_roi_power.py --stage compute --beh {beh} --band {band} "
            f"--fc-mode {args.fc_mode}")

    rows = []
    for f in tqdm(files, desc="load sessions"):
        try:
            p = fc.load_pickle(str(f))
        except Exception as e:
            print(f"[skip] {f.name}: {e!r}")
            continue
        if "log10_lo" not in p:
            print(f"[skip] {f.name}: written by an older version; "
                  "delete it and rerun the compute stage")
            continue
        sub = str(p["sid"][0])
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        for lab, r, lo, hi, dd in zip(p["labels"], roi, p["log10_lo"],
                                      p["log10_hi"], p["cohens_d"]):
            if r is None or not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            rows.append((sub, str(lab), r, float(lo), float(hi), float(dd)))

    if not rows:
        raise SystemExit("no electrodes with a Burke ROI and finite power")
    df = pd.DataFrame(rows,
                      columns=["sub", "label", "roi", "lo", "hi", "cohens_d"])
    # one row per (subject, electrode): mean across that subject's sessions
    # (log power geometrically, d as a plain mean of effect sizes)
    return (df.groupby(["sub", "label", "roi"], as_index=False)
              [["lo", "hi", "cohens_d"]].mean())


def subject_roi_table(elec_df, min_electrodes=1):
    """Per-electrode measures -> per-(subject, ROI) means.

    z_lo / z_hi are z-scored across ALL of that subject's ROI-assigned
    electrodes (not within ROI), which is what removes the subject-level gain
    offset while preserving between-ROI differences. cohens_d and change_dB need
    no normalization -- the offset cancels in the window contrast.
    """
    parts = []
    for _sub, g in elec_df.groupby("sub", sort=True):
        g = g.copy()
        g["change_dB"] = 10.0 * (g["hi"] - g["lo"])
        for src, dst in (("lo", "z_lo"), ("hi", "z_hi")):
            g[dst] = fc.zscore_across(g[src].to_numpy(float))
        parts.append(g)
    per_elec = pd.concat(parts, ignore_index=True)
    return per_elec, fc.subject_roi_means(per_elec, list(MEASURES),
                                          min_electrodes=min_electrodes)


def run_latency_stage(save_root, beh, band, args, lobe_of):
    """Time course of the contrast, for EITHER estimator.

    Gated on the pickles actually carrying per-bin data, not on fc_mode -- the
    windowed multitaper produces a latency axis too (on the same bin centres),
    which is the whole point of being able to compare the two.
    """
    if not getattr(args, "time_bin_ms", None):
        return
    for responsive_only in ([False, True] if args.responsive_only else [False]):
        keep = None
        if responsive_only:
            rdf = collect_responsiveness(save_root, beh, band, lobe_of,
                                         args.n_sessions, fc_mode=args.fc_mode)
            rdf = rdf[rdf["p"] < args.responsive_alpha]
            keep = set(zip(rdf["sub"], rdf["label"]))
            print(f"[latency/responsive] restricting to {len(keep)} contacts "
                  f"at p<{args.responsive_alpha:g}")
        _latency_one(save_root, beh, band, args, lobe_of, keep, responsive_only)


def _latency_one(save_root, beh, band, args, lobe_of, keep, responsive_only):
    bin_df = collect_bin_table(save_root, beh, band, lobe_of,
                               args.n_sessions, fc_mode=args.fc_mode,
                               keep_keys=keep)
    if bin_df is None or len(bin_df) == 0:
        print("[latency] these pickles carry no per-bin data; skipping. "
              "Recompute with --time-bin-ms to get a time course.")
        return
    tbl = subject_roi_bin_table(bin_df, min_electrodes=args.min_electrodes)
    stats = bin_stats(tbl, roi_order=fc.ROI_ORDER,
                      n_perm=args.tfce_perm, seed=0)

    # Electrodes actually contributing to each ROI, after the min-electrodes cut.
    cnt = (bin_df[bin_df["roi"].isin(set(tbl["roi"]))]
           .groupby("roi").agg(n_elec=("label", "nunique"),
                               n_sub=("sub", "nunique")).reset_index())
    lab = "responsive only" if responsive_only else "all electrodes"
    print(f"\n[latency] electrodes per ROI ({lab}):")
    print(cnt.sort_values("n_elec", ascending=False).to_string(index=False))

    tag = (f"{beh}_{band}_{args.fc_mode}_{args.time_bin_ms}ms"
           + ("_responsive" if responsive_only else ""))
    os.makedirs(args.out_dir, exist_ok=True)
    cnt.to_csv(join(args.out_dir, f"power_timecourse_{tag}_counts.csv"),
               index=False)
    stats.to_csv(join(args.out_dir, f"power_timecourse_{tag}_stats.csv"),
                 index=False)
    tbl.to_csv(join(args.out_dir, f"power_timecourse_{tag}_per_subject.csv"),
               index=False)
    path = plot_time_course(stats, args.out_dir, beh, band, tag)
    sig = stats[stats["q"] < 0.05]
    print(f"[latency] {tbl['sub'].nunique()} subjects, "
          f"{stats['bin_ms'].nunique()} bins x {stats['roi'].nunique()} ROIs; "
          f"{len(sig)} cells at q<0.05")
    if len(sig):
        # Peak per ROI, not "earliest significant cell": the latter sorts by time
        # regardless of sign or size and so surfaces tiny suppressions (e.g.
        # L-frontal @ 75 ms, d=-0.02) as if they were response onsets.
        pk = (sig.loc[sig.groupby("roi")["mean_d"].idxmax()]
                 .sort_values("mean_d", ascending=False).head(4))
        print("[latency] strongest ROIs (peak bin):")
        for _, r in pk.iterrows():
            print(f"          {r['roi']:<14} peak @ {r['bin_ms']:>4.0f} ms  "
                  f"d={r['mean_d']:+.3f}  q={r['q']:.1e}")
    print(f"[latency] wrote {path}")


def run_responsiveness_stage(save_root, beh, band, args, lobe_of):
    """Long-style electrode-selection summary. Plot-stage only, no recompute."""
    for fine in ([False, True] if args.fine_labels else [False]):
        df = collect_responsiveness(save_root, beh, band, lobe_of, args.n_sessions,
                                    fc_mode=args.fc_mode, by_fine_label=fine)
        st = responsiveness_stats(df, alpha=args.responsive_alpha)
        tag = (f"{beh}_{band}_{args.fc_mode}" + ("_fine" if fine else ""))
        os.makedirs(args.out_dir, exist_ok=True)
        # CSV is ALWAYS unfiltered -- the cut below is display only.
        st.to_csv(join(args.out_dir, f"responsiveness_{tag}.csv"), index=False)
        # Fine labels are ~40 groups, most with no response at all. Plot the
        # top N by median |t| (stats is already sorted that way), after
        # dropping labels too small for that median to be stable. The CSV keeps
        # every label regardless.
        st_plot = st
        if fine:
            big = st[st["n_elec"] >= _FINE_MIN_ELECTRODES]
            tiny = st[st["n_elec"] < _FINE_MIN_ELECTRODES]
            st_plot = big.head(args.fine_top_n)
            if len(tiny):
                print(f"[responsive/fine] excluded {len(tiny)} label(s) with "
                      f"<{_FINE_MIN_ELECTRODES} electrodes: "
                      f"{', '.join(tiny['grp'].head(8))}"
                      f"{' ...' if len(tiny) > 8 else ''}")
        df_plot = df[df["grp"].isin(set(st_plot["grp"]))]
        path = plot_responsiveness(df_plot, st_plot, args.out_dir, beh, band, tag,
                                   alpha=args.responsive_alpha)
        if fine and len(st) > len(st_plot):
            print(f"[responsive/fine] plotting top {len(st_plot)} of {len(st)} "
                  f"labels by median |t|; full table in the CSV")
        overall = 100 * (df["p"] < args.responsive_alpha).mean()
        print(f"[responsive{'/fine' if fine else ''}] {len(df)} electrodes, "
              f"{overall:.1f}% responsive at p<{args.responsive_alpha:g} "
              f"(Long: 25.3%)")
        print(st.head(6).to_string(index=False))
        print(f"[responsive] wrote {path}")


def run_plot_stage(save_root, beh, band, args):
    lobe_of = fc.load_burke_maps()
    run_responsiveness_stage(save_root, beh, band, args, lobe_of)
    run_latency_stage(save_root, beh, band, args, lobe_of)
    elec_df = collect_electrode_table(save_root, beh, band, lobe_of,
                                      args.n_sessions,
                                      fc_mode=args.fc_mode)
    per_elec, tbl = subject_roi_table(elec_df,
                                      min_electrodes=args.min_electrodes)
    print(f"[collect] {elec_df['sub'].nunique()} subjects, "
          f"{len(elec_df)} ROI-assigned electrodes")

    # every measure is tested and written to the stats CSV, whether or not it
    # got a panel in this figure
    stats = {m: fc.roi_stats(tbl, m) for m in MEASURES}
    c = fc.contrast(beh)
    fc.print_roi_stats(
        stats["cohens_d"],
        f"{c['hi_label']} vs {c['lo_label']} {band} power (Cohen's d, mean over "
        f"the ROI's electrodes), per ROI:")

    labels = panel_labels(beh, band)
    rng = np.random.default_rng(0)
    fc.roi_figure(
        [(labels[m],
          lambda ax, m=m: fc.roi_panel(ax, tbl, m, rng, stats=stats[m],
                                       style=args.style))
         for m in args.measures],
        args.out_dir, f"roi_power_{beh}_{band}")

    fc.write_roi_csvs(args.out_dir, f"roi_power_{beh}_{band}", tbl, per_elec,
                      stats)
    return tbl


# ---------------------------------- CLI --------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="both", choices=("compute", "plot", "both"))
    p.add_argument("--fc-mode", default=fc.FC_MODE,
                   choices=list(fc.FC_MODES), dest="fc_mode",
                   help="spectral estimator for the phase metrics. multitaper (default): one band-averaged estimate, no time axis. cwt_morlet: Morlet wavelets, time-resolved, enables latency analyses. Outputs land in a separate <band>__cwt_morlet directory so the two estimators never overwrite each other.")
    fc.add_common_args(p)
    p.add_argument("--out-dir", default=join("figures", "burke_roi_power"))
    p.add_argument("--measures", nargs="+", default=list(DEFAULT_MEASURES),
                   choices=list(MEASURES),
                   help="panels to draw; all measures are tested and written to "
                        f"the stats CSV regardless (default: {list(DEFAULT_MEASURES)})")
    p.add_argument("--style", default="box", choices=("box", "ci"),
                   help="box: median/IQR + whiskers + mean diamond (default). "
                        "ci: mean + 95%% CI, clearer on small subsets")
    p.add_argument("--time-bin-ms", type=int, default=fc.TIME_BIN_MS,
                   dest="time_bin_ms",
                   help="width (ms) of the latency bins the Morlet path averages "
                        "within before the across-event contrast. Ignored under "
                        "--fc-mode multitaper (no time axis). Rao uses 200; 50 is "
                        "the default here because a 5-cycle Morlet at 70 Hz has "
                        "sigma=11.4 ms, so 50 ms bins are ~4.4 sigma apart and "
                        "effectively independent. Do not go below ~25 ms.")
    p.add_argument("--mt-window-ms", type=int, default=fc.MT_WINDOW_MS,
                   dest="mt_window_ms",
                   help="sliding-window length (ms) for the WINDOWED multitaper "
                        "latency axis, centred on the same bin centres as the "
                        "Morlet path. 100 (default) matches Morlet's 12-bin axis "
                        "while sharing only 50%% of the data between neighbours. "
                        "The floor is ~100: at 50 ms the smoothing half-bandwidth "
                        "is 12 Hz, so the estimate reaches 58 Hz and mains leaks "
                        "in from the 58-62 notch. Note the "
                        "effective temporal resolution is this window, NOT "
                        "--time-bin-ms; the axis is only SAMPLED at that step.")
    p.add_argument("--responsive-alpha", type=float, default=1e-8,
                   dest="responsive_alpha",
                   help="p threshold for calling an electrode task-responsive. "
                        "1e-8 matches Long et al., who retained 25.3%% of "
                        "electrodes at that level.")
    p.add_argument("--tfce-perm", type=int, default=0, dest="tfce_perm",
                   help="permutations for the TFCE correction on the time "
                        "course (0 = off, use BH instead; 1000 is typical). "
                        "TFCE clusters along TIME within each ROI, never across "
                        "ROIs, and builds its null by sign-flipping whole "
                        "subjects -- so it makes no independence assumption "
                        "about neighbouring bins, unlike BH. Writes q_tfce, "
                        "which the figure then uses for its stars.")
    p.add_argument("--responsive-only", action="store_true",
                   dest="responsive_only",
                   help="ALSO produce the time-course figure using only "
                        "task-responsive contacts (p < --responsive-alpha), the "
                        "electrode set Long et al. compute their latencies on. "
                        "Writes a second figure tagged _responsive plus a "
                        "per-ROI electrode-count CSV.")
    p.add_argument("--fine-labels", action="store_true", dest="fine_labels",
                   help="ALSO group responsiveness by the raw reg_full label "
                        "instead of the 12 Burke ROIs. Splits occipital into "
                        "calcarine / cuneus / fusiform / lateral occipital / "
                        "lingual -- the five regions Long reports separately and "
                        "that this pipeline otherwise pools into one bar.")
    p.add_argument("--fine-top-n", type=int, default=15, dest="fine_top_n",
                   help="with --fine-labels, plot only the top N reg_full labels "
                        "by median |t| (default 15). Display only -- the CSV "
                        "always lists every label.")
    p.add_argument("--min-electrodes", type=int, default=3,
                   help="min electrodes a subject must have IN AN ROI for that "
                        "(subject, ROI) cell to enter the group test "
                        "(default: 3, matching build_roi_synchrony.py so the "
                        "power and synchrony figures admit the same subjects)")
    args = p.parse_args()

    # Morlet figures go to their own subfolder so the two estimators' figures
    # never overwrite each other, mirroring the <band>__cwt_morlet split on the
    # compute side. An explicit --out-dir is still honoured as the parent.
    if args.fc_mode != "multitaper":
        args.out_dir = join(args.out_dir, args.fc_mode)
    if args.n_sessions is not None and args.n_subjects is not None:
        raise ValueError("pass only one of --n-sessions / --n-subjects")
    return args


# --------------------------- latency (time-bin) stage -------------------------
# Only populated by the Morlet path: `cohens_d_bins` (n_ch, n_bins) is Cohen's d
# computed independently inside each time bin, so it is d as a function of
# latency. Multitaper pickles have None here and these functions no-op.

def collect_bin_table(save_root, beh, band, lobe_of, n_sessions=None,
                      fc_mode="cwt_morlet", keep_keys=None):
    """Per-(subject, electrode, time-bin) Cohen's d -> tidy frame.

    Mirrors collect_electrode_table's aggregation: sessions of one subject are
    averaged per electrode LABEL first, so a 4-session subject does not outweigh
    a 1-session subject.
    """
    d = power_dir(save_root, beh, band, fc_mode)
    files = sorted(d.glob("*_power.pkl"))
    if n_sessions is not None:
        files = files[:n_sessions]

    rows, centers = [], None
    for f in tqdm(files, desc="load bins"):
        try:
            p = fc.load_pickle(str(f))
        except Exception:
            continue
        if p.get("cohens_d_bins") is None:
            continue
        if centers is None:
            centers = np.asarray(p["bin_centers_ms"], float)
        sub = str(p["sid"][0])
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        db = np.asarray(p["cohens_d_bins"], float)
        for lab, r, drow in zip(p["labels"], roi, db):
            if r is None:
                continue
            # keep_keys restricts to a (subject, electrode) whitelist -- used to
            # rerun the time course on TASK-RESPONSIVE contacts only, which is
            # how Long et al. compute their latencies (they keep the 25% of
            # electrodes clearing p<1e-8 and discard the rest).
            if keep_keys is not None and (sub, str(lab)) not in keep_keys:
                continue
            for k, val in enumerate(drow):
                if np.isfinite(val):
                    rows.append((sub, str(lab), r, float(centers[k]), float(val)))

    if not rows:
        return pd.DataFrame(columns=["sub", "label", "roi", "bin_ms", "d"])
    df = pd.DataFrame(rows, columns=["sub", "label", "roi", "bin_ms", "d"])
    return (df.groupby(["sub", "label", "roi", "bin_ms"], as_index=False)["d"]
              .mean())


def subject_roi_bin_table(bin_df, min_electrodes=3):
    """(subject, ROI, bin) mean d, applying the same electrode floor as the
    collapsed figure so the two describe the same cells."""
    g = (bin_df.groupby(["sub", "roi", "bin_ms"])
               .agg(d=("d", "mean"), n_elec=("label", "nunique"))
               .reset_index())
    return g[g["n_elec"] >= min_electrodes].reset_index(drop=True)


def bin_stats(tbl, roi_order=None, n_perm=0, seed=0):
    """One-sample t vs 0 per (ROI, bin), BH-FDR over ALL roi x bin cells.

    FDR is taken over the whole grid rather than per ROI, because the family
    being tested is the whole time-by-region map.

    NOTE: this is an interim correction. The principled test for a contiguous
    time grid is TFCE (Rao's tfce.py, the machinery behind his Fig 4B), which
    exploits the fact that neighbouring bins are not independent tests but a
    cluster. BH here is conservative in the wrong way: it ignores that structure
    and so will miss temporally extended-but-weak effects that TFCE would find.
    """
    from scipy.stats import ttest_1samp
    rois = list(roi_order) if roi_order is not None else sorted(tbl["roi"].unique())
    rows = []
    for roi in rois:
        sub = tbl[tbl["roi"] == roi]
        for b in sorted(sub["bin_ms"].unique()):
            v = sub.loc[sub["bin_ms"] == b, "d"].dropna().to_numpy(float)
            if v.size >= fc.MIN_SUBJECTS_ROI and np.ptp(v) > 0:
                t, p = ttest_1samp(v, 0.0)
            else:
                t = p = np.nan
            rows.append({"roi": roi, "bin_ms": b, "n": v.size,
                         "mean_d": float(np.nanmean(v)) if v.size else np.nan,
                         "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size))
                                if v.size > 1 else np.nan,
                         # t-based 95% CI half-width, via the same fc.mean_ci
                         # the ROI box plots use -- so the two figures in this
                         # script mean the same thing by a shaded interval.
                         "ci95": (fc.mean_ci(v)[1] if v.size > 1 else np.nan),
                         "t": float(t), "p": float(p)})
    out = pd.DataFrame(rows)
    m = out["p"].notna()
    q = np.full(len(out), np.nan)
    if m.any():                                    # Benjamini-Hochberg
        pv = out.loc[m, "p"].to_numpy(float)
        order = np.argsort(pv)
        n = pv.size
        adj = np.minimum.accumulate((pv[order] * n / np.arange(1, n + 1))[::-1])[::-1]
        qv = np.empty(n); qv[order] = np.clip(adj, 0, 1)
        q[np.where(m)[0]] = qv
    out["q"] = q

    if n_perm:
        out = _add_tfce(out, tbl, rois, n_perm=n_perm, seed=seed)
    return out


def _add_tfce(out, tbl, rois, n_perm=1000, seed=0, min_subjects=10):
    """Attach TFCE-corrected p-values (`q_tfce`) to the per-bin stats table.

    Uses `mne.stats.permutation_cluster_1samp_test`. The `threshold=dict(...)`
    form is what selects TFCE, and `adjacency=None` makes clustering 1-D along
    TIME -- which is what we want, because our map is ROI x time and only time is
    continuous. L-frontal and L-parietal are not neighbours in any metric sense,
    and the row order is an arbitrary sort, so 2-D clustering would merge
    unrelated regions (measured: a 1.41x inflation of TFCE when two ROI rows hold
    identical clusters).

    RUN PER ROI, FOR TWO REASONS.
      1. MNE needs complete data and cannot omit NaN. Across the full ROI x time
         map 54% of cells are missing and exactly ONE subject of 360 has every
         ROI -- but WITHIN a region the data is complete (110-249 subjects), so
         the per-ROI split costs no data at all.
      2. A joint null would take the max statistic over cells whose n ranges
         93-249. Low-n cells give noisier t, dominate the max, and inflate the
         null for well-powered regions. Per-ROI nulls never compare across n.

    THE CORRECTION IS THEREFORE TWO-LEVEL:
      within a region -- TFCE + sign-flip permutation, family-wise across time;
      across regions  -- Benjamini-Hochberg over every (ROI, bin) p.
    MNE's p-values alone are corrected across TIME ONLY; without the second level
    the 12 regions searched would go uncounted.

    Why bother when BH-on-t is already there: that treats each cell as if it
    stood alone, so a weak effect spread over consecutive bins fails at every
    single bin and vanishes. TFCE scores a cell by the size of the contiguous run
    it belongs to as well as its height, and its null comes from flipping whole
    subjects -- carrying the real temporal autocorrelation into the null instead
    of assuming it away.

    TFCE is NOT automatically kinder. Height enters squared and extent only as a
    square root, so a strong brief effect still outranks a weak sustained one; on
    this dataset it is the STRICTER test (41 significant cells vs 61 for BH).
    """
    from mne.stats import permutation_cluster_1samp_test

    rows = []
    for roi in rois:
        piv = (tbl[tbl["roi"] == roi]
               .pivot_table(index="sub", columns="bin_ms", values="d")
               .dropna())                       # complete cases within this ROI
        if piv.shape[0] < min_subjects:
            continue
        _, _, p_roi, _ = permutation_cluster_1samp_test(
            piv.to_numpy(float),
            threshold=dict(start=0, step=0.05),   # dict form => TFCE
            n_permutations=n_perm,
            adjacency=None,                       # 1-D: cluster along time only
            tail=0,                               # two-sided
            seed=seed,
            out_type="mask",
            verbose=False,
        )
        rows += [(roi, b, float(pv)) for b, pv in zip(piv.columns, p_roi)]

    out = out.copy()
    if not rows:
        out["q_tfce"] = np.nan
        return out

    p_df = pd.DataFrame(rows, columns=["roi", "bin_ms", "p_tfce"])
    # Second level: BH across every (ROI, bin) so the 12 regions are accounted for.
    from statsmodels.stats.multitest import multipletests
    p_df["q_tfce"] = multipletests(p_df["p_tfce"].to_numpy(float),
                                   method="fdr_bh")[1]
    key = {(r, b): q for r, b, q in
           zip(p_df["roi"], p_df["bin_ms"], p_df["q_tfce"])}
    out["q_tfce"] = [key.get((r, b), np.nan)
                     for r, b in zip(out["roi"], out["bin_ms"])]
    return out


def pretty_roi(name):
    """Display form of an ROI or reg_full label: capitalise the region word.

    Keeps a single-letter hemisphere prefix intact, so "L-occipital" becomes
    "L-Occipital". Only the first letter is raised -- title-casing every word
    would give "Lateral Occipital Cortex", which reads oddly for anatomy.
    """
    t = str(name)
    if "-" in t and len(t.split("-", 1)[0]) == 1:
        hemi, rest = t.split("-", 1)
        return f"{hemi.upper()}-{rest[:1].upper()}{rest[1:]}"
    return t[:1].upper() + t[1:]


def plot_time_course(stats, out_dir, beh, band, tag, alpha=0.05):
    """One small-multiple panel per ROI: mean d +/- SEM vs latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rois = [r for r in fc.ROI_ORDER if r in set(stats["roi"])]
    ncol = 4
    nrow = int(np.ceil(len(rois) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    ylim = np.nanmax(np.abs(stats["mean_d"].to_numpy(float))) * 1.35 or 1.0

    for ax, roi in zip(axes, rois):
        s = stats[stats["roi"] == roi].sort_values("bin_ms")
        x = s["bin_ms"].to_numpy(float)
        y = s["mean_d"].to_numpy(float)
        # 95% CI, not SEM: a +/-1 SEM band clears zero at roughly p=0.32, which
        # reads as significant next to the stars. Matches --style ci on the box
        # plots (fc.mean_ci).
        e = s["ci95"].to_numpy(float)
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.fill_between(x, y - e, y + e, alpha=0.25, lw=0, zorder=2)
        ax.plot(x, y, lw=1.6, zorder=3)
        # Significance as stars (fc.stars: *** / ** / *), the same convention
        # the rest of the project's figures use.
        qcol = "q_tfce" if "q_tfce" in s.columns else "q"
        q = s[qcol].to_numpy(float)
        for xi, qi in zip(x, q):
            lab = fc.stars(qi)
            if lab:
                ax.text(xi, ylim * 0.80, lab, ha="center", va="center",
                        fontsize=9, color="k", zorder=4)
        ax.set_title(f"{pretty_roi(roi)}  (n={int(np.nanmax(s['n']))})",
                     fontsize=9)
        ax.set_ylim(-ylim, ylim)
    for ax in axes[len(rois):]:
        ax.axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel("Time after word onset (ms)")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("Cohen's d")

    #fig.suptitle(f"{fc.contrast(beh)['hi_label']} vs "
    #             f"{fc.contrast(beh)['lo_label']} — {band} power over latency\n"
    #             f"band: 95% CI across subjects   |   "
    #             f"* q<0.05  ** q<0.01  *** q<0.001  (BH over all ROI x bin cells)",
    #             fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(out_dir, exist_ok=True)
    path = join(out_dir, f"power_timecourse_{tag}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path



# --------------------------- responsiveness stage -----------------------------
# Long et al.'s electrode-selection analysis, recoverable from the stored
# per-electrode fields with no recompute: t = d * sqrt(n_lo*n_hi/(n_lo+n_hi)).
#
# Their criterion is a t-test of HG during word presentation vs blank screen,
# |t|, p < 1e-8, which retained 2775/10949 electrodes (25.3%). Ours contrasts
# Word On against Word Off rather than a blank screen, but it asks the same
# question of each electrode: does this contact respond to the word at all.
#
# This is the analysis that explains a flat ROI mean. If occipital has ~70%
# responsive contacts and frontal ~5%, the frontal ROI average is diluted more
# than an order of magnitude by contacts that never respond -- so "flat" is a
# statement about the POPULATION, not about whether frontal cortex responds.

# Label values meaning "no anatomical assignment" rather than a region. Dropped
# BEFORE grouping: they are not a brain area, and unlabelled contacts are
# numerous enough to outrank real regions on any count.
_UNLABELLED = {"nan", "none", "", "unknown", "n/a", "na", "white matter",
               "unlabeled", "unlabelled"}

# Minimum electrodes for a reg_full label to be RANKED/PLOTTED in the fine-label
# figure. Not a CLI flag: it is a floor for the ranking statistic to mean
# anything, not an analysis choice. Mean |t| over <10 contacts is dominated by
# whichever one happens to be extreme. The CSV is unaffected -- every label,
# however small, is still written out.
_FINE_MIN_ELECTRODES = 10


def collect_responsiveness(save_root, beh, band, lobe_of, n_sessions=None,
                           fc_mode="multitaper", by_fine_label=False):
    """Per-(subject, electrode) t and p for the word contrast.
    """
    from scipy.stats import t as tdist

    d = power_dir(save_root, beh, band, fc_mode)
    files = sorted(d.glob("*_power.pkl"))
    if n_sessions is not None:
        files = files[:n_sessions]
    rows = []
    for f in tqdm(files, desc="load responsiveness"):
        try:
            p = fc.load_pickle(str(f))
        except Exception:
            continue
        if "cohens_d" not in p or "n_lo" not in p:
            continue
        sub = str(p["sid"][0])
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        fine = np.asarray(p["reg_full"], dtype=object)
        nlo, nhi = float(p["n_lo"]), float(p["n_hi"])
        for lab, r, fl, dd in zip(p["labels"], roi, fine, p["cohens_d"]):
            # Check the RAW value BEFORE str(). reg_full is np.nan for contacts
            # with no anatomical label (white matter, outside brain, unmapped),
            # and str(np.nan) == "nan" - which silently became its own group and
            # topped the chart, because unlabelled contacts are common.
            raw = fl if by_fine_label else r
            if raw is None:
                continue
            if isinstance(raw, float) and not np.isfinite(raw):
                continue
            grp = str(raw).strip()
            if grp.lower() in _UNLABELLED:
                continue
            if not np.isfinite(dd):
                continue
            rows.append((sub, str(lab), str(grp), str(r), float(dd), nlo, nhi))
    if not rows:
        raise SystemExit(f"no usable pickles in {d}")
    df = pd.DataFrame(rows,
                      columns=["sub", "label", "grp", "roi", "d", "n_lo", "n_hi"])
    # one row per (subject, electrode): average d over that subject's sessions.
    # `roi` rides along so a fine-label figure can still be coloured by lobe.
    df = (df.groupby(["sub", "label", "grp", "roi"], as_index=False)
            .agg(d=("d", "mean"), n_lo=("n_lo", "mean"), n_hi=("n_hi", "mean")))
    # d -> t -> two-sided p (pooled-SD d, independent-samples t)
    df["t"] = df["d"] * np.sqrt(df["n_lo"] * df["n_hi"] / (df["n_lo"] + df["n_hi"]))
    df["abs_t"] = df["t"].abs()
    df["p"] = 2 * tdist.sf(df["abs_t"], df["n_lo"] + df["n_hi"] - 2)
    return df


def responsiveness_stats(df, alpha=1e-8):
    """Per group: n electrodes, % responsive, mean |t|, % of responders enhanced."""
    df = df.assign(responsive=df["p"] < alpha, enhanced=df["t"] > 0)
    g = df.groupby("grp")
    out = pd.DataFrame({
        "n_elec": g.size(),
        "n_responsive": g["responsive"].sum(),
        "pct_responsive": 100 * g["responsive"].mean(),
        "mean_abs_t": g["abs_t"].mean(),
        "median_abs_t": g["abs_t"].median(),
    })
    resp = df[df["responsive"]]
    out["pct_enhanced_of_responders"] = (
        100 * resp.groupby("grp")["enhanced"].mean()).reindex(out.index)
    out["roi"] = g["roi"].agg(lambda v: v.mode().iat[0] if len(v) else "")
    # Sorted by MEDIAN |t| -- the statistic the box plot actually draws, so the
    # panel reads top-to-bottom in the order it is sorted. It is also robust:
    # unlike the mean (Long's "average magnitude t-value") a couple of extreme
    # contacts cannot carry a label, which is how a 5-contact caudate nucleus
    # ranked above lingual. _FINE_MIN_ELECTRODES is still applied, since a
    # median over a handful of contacts is unstable even though it is not
    # outlier-driven.
    #
    # mean_abs_t stays in the CSV for comparison with Long, who ranks on it.
    # The two disagree for genuinely bimodal regions -- fusiform is the case to
    # watch: a responsive subset inside a mostly-unresponsive population gives a
    # high mean and a low median.
    return out.sort_values("median_abs_t", ascending=False).reset_index()


def plot_responsiveness(df, stats, out_dir, beh, band, tag, alpha=1e-8):
    """Two panels, one row per group, sharing the y order.

    LEFT  -- box plot of the per-electrode |t| in each group. One observation per
             (subject, electrode): that contact's Cohen's d for Word On vs Word
             Off across events, converted to t. Box = IQR, line = median,
             whiskers = 1.5 IQR, outliers hidden. It shows the SPREAD of response
             strength across contacts, which is the point -- a region can have a
             low median and still contain very strong individual electrodes.
    RIGHT -- how many of those contacts clear the responsiveness threshold.

    Coloured by Burke lobe via fc.LOBE_COLORS (Okabe-Ito, colourblind-safe), the
    palette the ROI figures already use. Colour is redundant with the y labels,
    never the sole carrier of identity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(stats["grp"])

    def _col(roi):
        lobe = str(roi).split("-", 1)[-1]          # "L-occipital" -> "occipital"
        return fc.LOBE_COLORS.get(lobe, "0.5")
    colors = [_col(r) for r in stats["roi"]]

    fig, axes = plt.subplots(
        1, 2, figsize=(13, max(4.0, 0.32 * len(order) + 1.6)), sharey=True)

    data = [df.loc[df["grp"] == g, "abs_t"].to_numpy(float) for g in order]
    bp = axes[0].boxplot(data, vert=False, showfliers=False, widths=0.65,
                         patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("0.25")
    for med in bp["medians"]:
        med.set_color("0.15"); med.set_linewidth(1.4)
    axes[0].set_yticks(np.arange(1, len(order) + 1))
    axes[0].set_yticklabels([pretty_roi(g) for g in order], fontsize=8)
    axes[0].set_xlabel("|t| per electrode")
    axes[0].axvline(0, color="0.7", lw=0.8)
    axes[0].invert_yaxis()

    y = np.arange(1, len(order) + 1)
    axes[1].barh(y, stats["n_responsive"], color=colors, alpha=0.85,
                 edgecolor="0.25")
    axes[1].set_xlabel("electrodes")
    axes[1].set_title(f"Number of responsive electrodes (p < {alpha:g})",
                      fontsize=10)
    axes[1].tick_params(labelleft=False)
    xmax = max(1, int(stats["n_responsive"].max()))
    for yi, nr, ne, pc in zip(y, stats["n_responsive"], stats["n_elec"],
                              stats["pct_responsive"]):
        axes[1].text(nr + 0.01 * xmax, yi, f"{nr:.0f} / {ne}  ({pc:.0f}%)",
                     va="center", fontsize=7)
    axes[1].set_xlim(0, xmax * 1.28)

    axes[0].set_title(f"High gamma power, "
                      f"{fc.contrast(beh)['hi_label']} vs "
                      f"{fc.contrast(beh)['lo_label']}", fontsize=10)

    # Lobe legend, outside on the right; only the lobes actually drawn.
    from matplotlib.patches import Patch
    seen, handles = set(), []
    for r in stats["roi"]:
        lobe = str(r).split("-", 1)[-1]
        if lobe not in seen:
            seen.add(lobe)
            handles.append(Patch(facecolor=fc.LOBE_COLORS.get(lobe, "0.5"),
                                 edgecolor="0.25", alpha=0.85,
                                 label=pretty_roi(lobe)))
    # tight_layout FIRST, so the axes are already shrunk into rect before the
    # legend is anchored against the reserved strip. Anchoring at x=1.0 (the
    # figure's right EDGE) put the legend outside the canvas, where savefig
    # silently clipped it -- the legend was being drawn all along, just off-page.
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    if handles:
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.885, 0.5),
                   frameon=False, fontsize=8, title="ROI",
                   title_fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    path = join(out_dir, f"responsiveness_{tag}.png")
    # bbox_inches="tight" is belt-and-braces: anything that still lands slightly
    # outside the canvas gets included rather than cropped away.
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


def main() -> None:
    import helper

    args = parse_args()
    root_dir, save_root = fc.resolve_roots(args)
    c = fc.contrast(args.beh)                   

    if args.beh in fc.PREPOST_SPEC:
        sp = fc.PREPOST_SPEC[args.beh]
        win = f"two windows: lo={sp['pre_win']}  hi={sp['post_win']}"
    else:
        win = (f"one window {tuple(helper.beh_to_event_windows[args.beh])}, "
               f"two event groups")
    print(f"[setup] beh={args.beh}  band={args.band} {fc.bands[args.band]} Hz")
    print(f"[setup] {c['lo_label']} (lo) vs {c['hi_label']} (hi) -- {win}")

    if args.stage in ("compute", "both"):
        fc.run_compute_stage(
            run_sess_power, desc="ROI power", job_name="roi_power",
            root_dir_=root_dir, local=args.local, n_sessions=args.n_sessions,
            n_subjects=args.n_subjects, n_workers=args.n_workers, mem=args.mem,
            walltime=args.walltime, cluster_log_dir=args.cluster_log_dir,
            save_root=save_root, beh=args.beh, band=args.band, root_dir=root_dir,
            fc_mode=args.fc_mode, time_bin_ms=args.time_bin_ms,
            mt_window_ms=args.mt_window_ms)

    if args.stage in ("plot", "both"):
        run_plot_stage(save_root, args.beh, args.band, args)


if __name__ == "__main__":
    main()
