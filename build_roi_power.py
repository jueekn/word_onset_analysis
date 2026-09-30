"""
build_roi_power.py

Local (single-electrode) spectral power in the same 12 Burke ROIs used by the
phase-connectivity analyses -- the power counterpart to build_roi_synchrony.py.

Same events, same windows, same band, same multitaper front end as the phase
metrics; the only change is that each electrode is treated on its own instead
of as one endpoint of a pair:

    per event, per electrode:  multitaper PSD over the window, averaged in band
    per electrode:             Cohen's d of hi vs lo power ACROSS events
    per subject, per ROI:      mean of d over that ROI's electrodes
    across subjects:           box plot (one box per ROI)

Two stages:

  compute  one pickle per session (--workers N to parallelise)
             <save_root>/<beh>/power/<band>/<ftag>_power.pkl
             {"sid", "labels", "reg_full", "pow_lo", "pow_hi",
              "cohens_d", "n_events"}
  plot     aggregate those pickles -> per-subject-per-ROI table -> box plots

Relationship to Rao et al. 2025 (J Neurosci)
--------------------------------------------
The per-electrode summary is Cohen's d of band power (no log transform), region-averaged, following
that paper's power pipeline (helper.get_power -> comp_elpomx ->
regionalize_electrode_powers). Specifically:

  adopted    `helper.cohens_d` verbatim (pooled-SD,
             the same function vendored in this repo); the effect size -- not raw
             power -- as the per-electrode quantity; mean of d over the
             electrodes in a region as the region value.
  not adopted  Rao's Morlet front end: power uses the same multitaper estimator
             as the phase metrics.
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

Measures. Raw power is not comparable across subjects (amplifier gain,
referencing, and coverage differ), so only window contrasts are used:

  cohens_d       effect size of hi vs lo power, across events, per electrode.
                 Unitless and gain-invariant; the plotted measure.
  change_dB      10 * log10(P_hi / P_lo) of the session-mean powers -- the same
                 contrast in dB. Written to the CSVs only.

Per ROI, a one-sample t of the subject values vs 0 (= paired t), FDR across the
12 ROIs (fc.roi_stats). The figure has one row per band already plotted into
--out-dir (fc.band_contrast_figure).

Timing (power time course) is computed for gamma bands only (fmin >= TIMING_MIN_HZ):
latency is not meaningful for slow oscillations. Long et al. settings (Hilbert,
responsiveness, fine labels) come from config `longetal_params`, never from flags.

Electrode -> ROI uses the canonical `regionalize_electrodes_by_type` label
(volumetric cascade for depths, surface for grid/strip) mapped through
region_to_burke_lobe.csv + hemisphere, identical to the connectivity scripts.

Usage:
    python build_roi_power.py                         # compute + plot
    python build_roi_power.py --n-sessions 2          # smoke test
    python build_roi_power.py --workers 4             # 4 sessions at a time
    python build_roi_power.py --stage plot            # replot from pickles
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

MEASURES = ("cohens_d", "change_dB")
TIMING_MIN_HZ = 30          # time course only for gamma (PI: latency meaningless at low freqs)
RESPONSIVE_ALPHA = 1e-8     # Long et al.'s responsiveness threshold
LONG = fc.LONGETAL or {}


def power_dir(root, beh, band):
    return Path(root) / beh / "power" / fc.band_dirname(band, fc.POWER_MODE)


# ------------------------------ compute stage --------------------------------
def _word_locked_eeg(dfrow: pd.Series, beh: str, win: tuple[float, float],
                     buffer_ms: float = 0.0, simulation_tag: str | None = None) -> Any:
    """EEG for the POST-type events of `beh`, loaded over one window spanning
    both the pre and post analysis windows.

    Same load as `fc.session_eeg` but with `beh` as a parameter. For the PREPOST
    behaviors the PRE_* events are row-for-row copies of the POST events
    (fc.compute_prepost_separate asserts this), so slicing one clip in time is
    equivalent to the two separate loads the FC pipeline does, and guarantees
    pre/post come from the same trials.
    """
    import fc_comparison_functions as fc
    import helper

    ev = fc.load_events(dfrow, beh)
    if ev is None:
        return None
    post_mask = np.asarray(ev.attrs["mask"], bool)
    if not post_mask.any():
        return None
    words = ev[post_mask].reset_index(drop=True)
    words.attrs = dict(ev.attrs)
    # buffer_ms widens the LOADED clip beyond `win` with real adjacent data, so
    # resample/notch (and Hilbert) edge effects stay outside the analysis windows.
    eeg, _ = fc.get_beh_eeg(dfrow, words, win, buffer_ms, simulation_tag)
    kept = np.asarray(eeg.event, int)
    items = words["item_name"].to_numpy()[kept]   # word of each kept event
    if (fc.LONGETAL or {}).get("baseline") != "full_isi":
        return (np.asarray(eeg.data), np.asarray(eeg.time, float),
                float(eeg.samplerate), items, None)
    # blank screen before each word: previous word's offset -> onset (ms); 750 when
    # there is no previous word in the list (first word follows the countdown)
    bev = helper.bids_reader(dfrow, eeg=False).load_events()
    bw = bev[bev["trial_type"] == "WORD"].sort_values("onset")
    blank = ((bw["onset"] - (bw["onset"] + bw["duration"]).groupby(bw["list"]).shift(1)) * 1000).fillna(750.0)
    blank = pd.Series(blank.to_numpy(), index=bw["sample"].astype(int).to_numpy())
    blank_ms = blank.reindex(words["eegoffset"].astype(int).to_numpy()[kept]).fillna(750.0).to_numpy()
    return (np.asarray(eeg.data), np.asarray(eeg.time, float),
            float(eeg.samplerate), items, blank_ms)


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


def hilbert_envelope(seg, sf, fmin, fmax, buf_samples, env_hz, n_bands=1):
    """Long et al. 2020 high gamma: Hilbert AMPLITUDE envelope at the native rate,
    trim the buffer, downsample to `env_hz`. (E, C, T). n_bands=1: one band-pass
    (same recipe as fc.compute_aec_buffered). n_bands>1: mean amplitude over
    log-spaced Gaussian bands, Hilbert done in the frequency domain (the
    Mesgarani-lab filterbank convention)."""
    import mne
    import scipy.fft as sfft
    from scipy.signal import hilbert
    from mne.filter import next_fast_len
    n = np.shape(seg)[-1]
    if n_bands > 1:
        N = next_fast_len(n)
        f = sfft.fftfreq(N, 1 / sf)
        X = sfft.fft(np.asarray(seg, np.float32), N, axis=-1)   # complex64 keeps memory down
        edges = np.geomspace(fmin, fmax, n_bands + 1)
        env = np.zeros(np.shape(seg), np.float32)
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = 2 * np.exp(-0.5 * ((f - np.sqrt(lo * hi)) / ((hi - lo) / 2)) ** 2) * (f > 0)
            env += np.abs(sfft.ifft(X * g.astype(np.float32), axis=-1)[..., :n]) / n_bands
    else:
        x = mne.filter.filter_data(np.asarray(seg, float), sf, fmin, fmax, verbose=False)
        env = np.abs(hilbert(x, N=next_fast_len(n), axis=-1))[..., :n]
    if buf_samples:
        env = env[..., buf_samples:n - buf_samples]
    return mne.filter.resample(np.asarray(env, float), down=sf / env_hz, verbose=False)


def band_power_windowed(seg, sf, fmin, fmax, bandwidth, bin_ms, window_ms):
    """Sliding-window multitaper power, (E, C, n_bins): windows of window_ms
    centred every bin_ms. Consecutive windows overlap, so the EFFECTIVE temporal
    resolution is window_ms; the axis is only SAMPLED at bin_ms. Bins whose full
    window falls outside the segment are NaN (a shorter window would have a
    different spectral resolution)."""
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
    simulation_tag: str | None = None,
) -> str:
    """Compute per-electrode band power for one session; write one pickle.

    Returns a short status string for the dispatcher's progress bar.
    """
    import helper
    import fc_comparison_functions as fc

    fc.root_dir = root_dir
    helper.root_dir = root_dir

    sid = f"{dfrow['sub']}_{dfrow['exp']}_{dfrow['sess']}"
    fc_mode = fc.POWER_MODE
    out_dir = str(power_dir(save_root, beh, band))
    out_path = join(out_dir, f"{fc.ftag(dfrow)}_power.pkl")

    if os.path.exists(out_path):
        try:
            cached = fc.load_pickle(out_path)
            # n_win_samples marks a pickle written after the pre/post window
            # equalization fix. Pickles without it carry the unequal-window bias
            # (pre estimated over one more sample than post), so treat them as
            # stale and recompute rather than silently reporting them cached.
            if all(k in cached for k in ("pow_lo", "pow_hi", "cohens_d",
                                         "reg_full", "n_win_samples")) and (
                    fc_mode != "hilbert" or cached.get("env_lo") is not None):
                return f"{sid}: cached"
        except Exception:
            pass

    fmin, fmax = fc.bands[band]
    hilb = fc_mode == "hilbert"   # filter edge lives in the real-data buffer
    buf_ms = fc.REAL_DATA_BUFFER_MS if hilb else 0.0
    spec = fc.PREPOST_SPEC[beh]
    pre_win, post_win = spec["pre_win"], spec["post_win"]
    loaded = _word_locked_eeg(dfrow, beh, (pre_win[0], post_win[1]),
                              buffer_ms=fc.REAL_DATA_BUFFER_MS,   # keeps notch ringing out of both windows
                              simulation_tag=simulation_tag)
    if loaded is None:
        return f"{sid}: no events ({beh})"
    data, t, sf, items, blank_ms = loaded

    pairs = helper.get_pairs(dfrow)
    n_ch = data.shape[1]
    if pairs is None or len(pairs) != n_ch:
        raise ValueError(
            f"{sid}: len(pairs)={None if pairs is None else len(pairs)} but eeg "
            f"n_ch={n_ch}; channel order misaligned for region labels.")

    reg_full = np.asarray(helper.regionalize_electrodes_by_type(pairs), dtype=object)
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

    nbuf = int(buf_ms * sf / 1000.0)
    widen = lambda w: (w[0] - buf_ms, w[1] + buf_ms)
    mt_win = fc.MT_WINDOW_MS
    time_bin_ms = fc.TIME_BIN_MS if fmin >= TIMING_MIN_HZ else None

    env_hz = (fc.LONGETAL or {}).get("envelope_hz", 100)
    env_cache = {}
    def _env(seg):   # Hilbert envelope, computed once per segment
        if id(seg) not in env_cache:
            env_cache[id(seg)] = hilbert_envelope(seg, sf, fmin, fmax, nbuf, env_hz,
                                                  (fc.LONGETAL or {}).get("hg_bands", 1))
        return env_cache[id(seg)]

    def _power(seg, bin_ms=None):
        if hilb:
            e = _env(seg)
            return helper.timebin_power_timeseries(e, env_hz, bin_width_ms=int(bin_ms)) if bin_ms else np.nanmean(e, -1)
        if bin_ms:
            return band_power_windowed(seg, sf, fmin, fmax, bandwidth,
                                       bin_ms, mt_win)
        return band_power(seg, sf, fmin, fmax, bandwidth)

    # Inclusive time masks give the pre window one more sample than the post
    # window, so equalize them exactly as the FC path does before estimating
    # spectra -- otherwise the frequency-bin centres shift between the arms.
    lo_seg = window_slice(data, t, widen(pre_win))
    hi_seg = window_slice(data, t, widen(post_win))
    if not hilb:   # Long's windows differ in length (750 vs 1600 ms); a mean amplitude doesn't need them equal
        lo_seg, hi_seg = fc.equalize_time_length(lo_seg, hi_seg)
    n_win = lo_seg.shape[-1] - 2 * nbuf
    if hilb and LONG.get("baseline") == "full_isi":
        # each trial's whole blank screen: blank the pre samples before the previous word's offset
        e = _env(lo_seg)
        t_env = pre_win[0] + np.arange(e.shape[-1]) * 1000.0 / env_hz
        e[np.broadcast_to(t_env[None, None, :] < -blank_ms[:, None, None], e.shape)] = np.nan
    p_lo = _power(lo_seg)
    p_hi = _power(hi_seg)

    p_lo[~np.isfinite(p_lo)] = np.nan
    p_hi[~np.isfinite(p_hi)] = np.nan

    # Per-bin contrast, (n_ch, n_bins). Same Cohen's d as the collapsed measure,
    # computed independently within each time bin -> d as a function of latency.
    d_bins = bin_centers = None
    if time_bin_ms:
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
        # then averaged over bins -- not from the whole-window call (600 ms @ 2 Hz
        # vs 100 ms @ 12 Hz would put a constant offset on the time course).
        # nanmean also drops the edge bins that cannot fit a full window.
        #
        # Raw power: the mean over B baseline bins and a single response bin
        # estimate the same expected power, so the null stays unbiased.
        l_lo = _power(lo_seg, bin_ms=time_bin_ms)             # (E, C, B)
        b_hi = _power(hi_seg, bin_ms=time_bin_ms)             # (E, C, B)
        l_lo[~np.isfinite(l_lo)] = np.nan
        base = np.nanmean(l_lo, axis=-1)                      # (E, C)
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
    # `helper.cohens_d` (pooled SD) on power, hi vs lo, across events.
    # Done channel by channel so a degenerate event drops only its own channel's
    # contribution instead of the whole electrode (helper.cohens_d is not
    # nan-aware); in practice every event is finite and this is one call's worth
    # of arithmetic per channel.
    d = np.full(n_ch, np.nan)
    for c in range(n_ch):
        a = p_hi[np.isfinite(p_hi[:, c]), c]
        b = p_lo[np.isfinite(p_lo[:, c]), c]
        if a.size >= 2 and b.size >= 2:
            d[c] = float(helper.cohens_d(a, b))

    # Sample-level contrast (hilbert): every envelope sample is an observation,
    # as Long et al. may have done (--t-unit sample at the plot stage).
    d_samp = n_samp_lo = n_samp_hi = None
    if hilb:
        flat = lambda e: e.transpose(0, 2, 1).reshape(-1, e.shape[1])   # (E*T, C)
        e_lo, e_hi = flat(_env(lo_seg)), flat(_env(hi_seg))
        e_lo = e_lo[np.isfinite(e_lo).all(1)]   # full_isi: drop blanked samples
        d_samp, n_samp_lo, n_samp_hi = helper.cohens_d(e_hi, e_lo), len(e_lo), len(e_hi)

    out: dict[str, Any] = {
        "sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"])),
        "labels": labels,
        "reg_full": reg_full,
        "pow_lo": np.nanmean(p_lo, axis=0),
        "pow_hi": np.nanmean(p_hi, axis=0),
        "cohens_d": d,
        "n_events": int(data.shape[0]),
        "n_lo": int(p_lo.shape[0]), "n_hi": int(p_hi.shape[0]),
        "n_win_samples": int(n_win),
        "beh": beh, "band": band, "fmin": fmin, "fmax": fmax,
        "lo_win": pre_win, "hi_win": post_win,
        "mt_bandwidth": bandwidth,
        "fc_mode": fc_mode,
        "measure": "amplitude" if hilb else "power",   # pow_* hold amplitude under hilbert
        "cohens_d_samples": d_samp, "n_lo_samples": n_samp_lo, "n_hi_samples": n_samp_hi,
        # per-event envelopes + words, for averaging each word over sessions (--combine word_average)
        "env_lo": _env(lo_seg).astype(np.float32) if hilb else None,
        "env_hi": _env(hi_seg).astype(np.float32) if hilb else None,
        "items": items if hilb else None,
        "simulation_tag": simulation_tag,
        "notch_harmonics_up_to_hz": fc.NOTCH_HARMONICS_UP_TO_HZ,
        "buffer_ms": fc.REAL_DATA_BUFFER_MS,
        # Latency axis (gamma only; None otherwise).
        "cohens_d_bins": d_bins,          # (n_ch, n_bins)
        "bin_centers_ms": bin_centers,    # ms into the post window
        "time_bin_ms": time_bin_ms,
        "mt_window_ms": mt_win if (time_bin_ms and not hilb) else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    fc.save_pickle(out_path, out)
    return f"{sid}: {n_ch} elec, {out['n_lo']}+{out['n_hi']} events"


# ------------------------------- plot stage ----------------------------------
def collect_electrode_table(save_root, beh, band, lobe_of, n_sessions=None):
    """Walk the per-session pickles -> tidy per-(subject, electrode) table.

    Sessions of the same subject are averaged per electrode LABEL before
    anything else, so a
    subject with 4 sessions does not outweigh one with 1.
    """
    d = power_dir(save_root, beh, band)
    files = sorted(d.glob("*_power.pkl"))[:n_sessions]
    if not files:
        raise SystemExit(
            f"no pickles in {d}\nrun the compute stage first: "
            f"python build_roi_power.py --stage compute --band {band}")

    rows = []
    for f in tqdm(files, desc="load sessions"):
        try:
            p = fc.load_pickle(str(f))
        except Exception as e:
            print(f"[skip] {f.name}: {e!r}")
            continue
        if "pow_lo" not in p:
            print(f"[skip] {f.name}: written by an older version; "
                  "delete it and rerun the compute stage")
            continue
        sub = str(p["sid"][0])
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        for lab, r, lo, hi, dd in zip(p["labels"], roi, p["pow_lo"],
                                      p["pow_hi"], p["cohens_d"]):
            if r is None or not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            rows.append((sub, str(lab), r, float(lo), float(hi), float(dd)))

    if not rows:
        raise SystemExit("no electrodes with a Burke ROI and finite power")
    df = pd.DataFrame(rows,
                      columns=["sub", "label", "roi", "lo", "hi", "cohens_d"])
    # one row per (subject, electrode): mean across that subject's sessions
    # (power and d both as plain means)
    return (df.groupby(["sub", "label", "roi"], as_index=False)
              [["lo", "hi", "cohens_d"]].mean())


def subject_roi_table(elec_df, min_electrodes=1, db=10.0):
    """Per-electrode measures -> per-(subject, ROI) means."""
    per_elec = elec_df.assign(change_dB=db * np.log10(elec_df["hi"] / elec_df["lo"]))  # db=20 for amplitude (hilbert)
    return per_elec, fc.subject_roi_means(per_elec, list(MEASURES),
                                          min_electrodes=min_electrodes)


def run_latency_stage(save_root, beh, band, args, lobe_of):
    """Time course of the contrast (gamma only). Long et al.: also on responsive
    contacts only, the electrode set they compute latencies on."""
    if fc.bands[band][0] < TIMING_MIN_HZ:
        return
    for responsive_only in ([False, True] if LONG else [False]):
        keep = None
        if responsive_only:
            rdf = collect_responsiveness(save_root, beh, band, lobe_of, args.n_sessions)
            rdf = rdf[rdf["p"] < RESPONSIVE_ALPHA]
            keep = set(zip(rdf["sub"], rdf["label"]))
            print(f"[latency/responsive] restricting to {len(keep)} contacts "
                  f"at p<{RESPONSIVE_ALPHA:g}")
        _latency_one(save_root, beh, band, args, lobe_of, keep, responsive_only)


def _latency_one(save_root, beh, band, args, lobe_of, keep, responsive_only):
    bin_df = collect_bin_table(save_root, beh, band, lobe_of,
                               args.n_sessions, keep_keys=keep)
    if len(bin_df) == 0:
        print("[latency] these pickles carry no per-bin data; skipping.")
        return
    tbl = subject_roi_bin_table(bin_df, min_electrodes=args.min_electrodes)
    stats = fc.bin_stats(tbl, "bin_ms", "d")

    # Electrodes actually contributing to each ROI, after the min-electrodes cut.
    cnt = (bin_df[bin_df["roi"].isin(set(tbl["roi"]))]
           .groupby("roi").agg(n_elec=("label", "nunique"),
                               n_sub=("sub", "nunique")).reset_index())
    lab = "responsive only" if responsive_only else "all electrodes"
    print(f"\n[latency] electrodes per ROI ({lab}):")
    print(cnt.sort_values("n_elec", ascending=False).to_string(index=False))

    tag = (f"{beh}_{band}_{fc.POWER_MODE}_{fc.TIME_BIN_MS}ms"
           + ("_responsive" if responsive_only else ""))
    os.makedirs(args.out_dir, exist_ok=True)
    cnt.to_csv(join(args.out_dir, f"power_timecourse_{tag}_counts.csv"),
               index=False)
    stats.to_csv(join(args.out_dir, f"power_timecourse_{tag}_stats.csv"),
                 index=False)
    tbl.to_csv(join(args.out_dir, f"power_timecourse_{tag}_per_subject.csv"),
               index=False)
    path = fc.roi_curve_figure(stats, "bin_ms", join(args.out_dir, f"power_timecourse_{tag}"),
                               "Time after word onset (ms)", "Cohen's d")
    sig = stats[stats["q"] < 0.05]
    print(f"[latency] {tbl['sub'].nunique()} subjects, "
          f"{stats['bin_ms'].nunique()} bins x {stats['roi'].nunique()} ROIs; "
          f"{len(sig)} cells at q<0.05")
    if len(sig):
        # Peak per ROI, not "earliest significant cell": the latter sorts by time
        # regardless of sign or size and so surfaces tiny suppressions (e.g.
        # L-frontal @ 75 ms, d=-0.02) as if they were response onsets.
        pk = (sig.loc[sig.groupby("roi")["mean"].idxmax()]
                 .sort_values("mean", ascending=False).head(4))
        print("[latency] strongest ROIs (peak bin):")
        for _, r in pk.iterrows():
            print(f"          {r['roi']:<14} peak @ {r['bin_ms']:>4.0f} ms  "
                  f"d={r['mean']:+.3f}  q={r['q']:.1e}")
    print(f"[latency] wrote {path}")


def run_responsiveness_stage(save_root, beh, band, args, lobe_of, fine_top_n=15):
    """Long-style electrode-selection summary. Plot-stage only, no recompute.
    fine_labels (or longetal) adds a figure by raw label (top `fine_top_n`)."""
    t_unit, combine = LONG.get("t_unit", "trial"), LONG.get("combine", "d_mean")
    min_resp = LONG.get("min_responsive_per_region", 0)
    for fine in ([False, True] if (args.fine_labels or LONG) else [False]):
        df = collect_responsiveness(save_root, beh, band, lobe_of, args.n_sessions,
                                    by_fine_label=fine)
        st = responsiveness_stats(df, alpha=RESPONSIVE_ALPHA)
        overall = 100 * (df["p"] < RESPONSIVE_ALPHA).mean()
        print(f"[responsive{'/fine' if fine else ''}] {len(df)} electrodes, "
              f"{overall:.1f}% responsive (t per {t_unit}) at p<{RESPONSIVE_ALPHA:g} "
              f"(Long: 25.3%)")
        tag = (f"{beh}_{band}_{fc.POWER_MODE}" + ("_fine" if fine else "")
               + (("_pertimepoint" if t_unit == "sample" else "_pertrial") if LONG else "")
               + ("_wordavg" if combine == "word_average" else ""))
        os.makedirs(args.out_dir, exist_ok=True)
        if fine and combine == "word_average":
            plot_long_peaks(df, args.out_dir, tag, RESPONSIVE_ALPHA,   # per trial: every region with a responder
                            min_resp if t_unit == "sample" else 1)
        # CSV is ALWAYS unfiltered -- the cut below is display only.
        st.to_csv(join(args.out_dir, f"responsiveness_{tag}.csv"), index=False)
        # Fine labels are ~40 groups, most with no response at all: plot the top
        # N by median |t| among labels with enough electrodes for a stable median.
        st_plot = st[st["n_elec"] >= _FINE_MIN_ELECTRODES].head(fine_top_n) if fine else st
        st_plot = st_plot[st_plot["n_responsive"] >= min_resp]
        if st_plot.empty:
            print(f"[responsive] no region has >= {min_resp} "
                  f"responsive electrodes; figure skipped (see the CSV)")
            continue
        df_plot = df[df["grp"].isin(set(st_plot["grp"]))]
        path = plot_responsiveness(df_plot, st_plot, args.out_dir, beh, band, tag,
                                   alpha=RESPONSIVE_ALPHA)
        print(st.head(6).to_string(index=False))
        print(f"[responsive] wrote {path}")


def run_plot_stage(save_root, beh, band, args):
    lobe_of = fc.load_burke_maps()
    run_responsiveness_stage(save_root, beh, band, args, lobe_of)
    run_latency_stage(save_root, beh, band, args, lobe_of)
    elec_df = collect_electrode_table(save_root, beh, band, lobe_of, args.n_sessions)
    per_elec, tbl = subject_roi_table(elec_df, db=20.0 if fc.POWER_MODE == "hilbert" else 10.0,
                                      min_electrodes=args.min_electrodes)
    print(f"[collect] {elec_df['sub'].nunique()} subjects, "
          f"{len(elec_df)} ROI-assigned electrodes")

    stats = {m: fc.roi_stats(tbl, m) for m in MEASURES}
    c = fc.contrast(beh)
    fc.print_roi_stats(
        stats["cohens_d"],
        f"{c['hi_label']} vs {c['lo_label']} {band} power (Cohen's d, mean over "
        f"the ROI's electrodes), per ROI:")
    stem = f"roi_power_{beh}_{{band}}"
    fc.write_roi_csvs(args.out_dir, stem.format(band=band), tbl, per_elec, stats)
    fc.band_contrast_figure(args.out_dir, stem, "cohens_d",
                            f"power {c['hi_label']} vs {c['lo_label']} (Cohen's d)")
    return tbl


# ---------------------------------- CLI --------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="both", choices=("compute", "plot", "both"))
    fc.add_common_args(p)
    p.add_argument("--out-dir", default=join("figures", "burke_roi_power"))
    p.add_argument("--fine-labels", action="store_true", dest="fine_labels",
                   help="also plot responsiveness by raw anatomical label")
    p.add_argument("--min-electrodes", type=int, default=3,
                   help="min electrodes a subject must have in an ROI to enter the group test")
    args = p.parse_args()
    args = p.parse_args()

    if fc.POWER_MODE != "multitaper":   # longetal: hilbert figures in their own subfolder
        args.out_dir = join(args.out_dir, fc.POWER_MODE)
    if args.n_sessions is not None and args.n_subjects is not None:
        raise ValueError("pass only one of --n-sessions / --n-subjects")
    return args


# --------------------------- latency (time-bin) stage -------------------------
# `cohens_d_bins` (n_ch, n_bins): Cohen's d within each time bin (gamma only).

def collect_bin_table(save_root, beh, band, lobe_of, n_sessions=None, keep_keys=None):
    """Per-(subject, electrode, time-bin) Cohen's d -> tidy frame.

    Mirrors collect_electrode_table's aggregation: sessions of one subject are
    averaged per electrode LABEL first, so a 4-session subject does not outweigh
    a 1-session subject.
    """
    files = sorted(power_dir(save_root, beh, band).glob("*_power.pkl"))[:n_sessions]
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


def _long_region(fine_label):
    """'L lateral occipital cortex' -> Long et al.'s subregion (hemispheres pooled)."""
    if not hasattr(_long_region, "map"):
        _long_region.map = pd.read_csv(join(os.path.dirname(os.path.abspath(__file__)),
                                            "long_regions.csv")).set_index("region")["long_region"]
    return _long_region.map.get(str(fine_label).split(" ", 1)[-1])


def word_averaged_rows(files, lobe_of, t_unit, by_fine_label):
    """Long et al.: average each word's envelope over a subject's sessions, then
    one d per electrode across words (trial) or across all envelope samples
    (sample). Electrodes = labels present in every session of the subject."""
    import helper
    by_sub = {}   # files grouped by subject; one subject's envelopes in memory at a time
    for f in files:
        by_sub.setdefault(Path(f).name.split("_")[0], []).append(f)
    rows = []
    for sub, fs in tqdm(by_sub.items(), desc="word-average"):
        ps = [p for p in (fc.load_pickle(str(f)) for f in fs) if p.get("env_lo") is not None]
        if not ps:
            continue
        labels = [l for l in ps[0]["labels"] if all(l in set(q["labels"]) for q in ps)]
        col = lambda q: [list(q["labels"]).index(l) for l in labels]
        acc = {}   # word -> [sum_lo, n_lo, sum_hi, n_hi], NaN-aware (full_isi blanks pre samples)
        for q in ps:
            c = col(q)
            for i, w in enumerate(q["items"]):
                lo, hi = q["env_lo"][i][c], q["env_hi"][i][c]
                s = acc.setdefault(str(w).upper(), [0.0, 0, 0.0, 0])
                s[0], s[1] = s[0] + np.nan_to_num(lo), s[1] + np.isfinite(lo)
                s[2], s[3] = s[2] + hi, s[3] + 1
        with np.errstate(invalid="ignore", divide="ignore"):
            lo = np.stack([s[0] / s[1] for s in acc.values()])   # (W, C, T_lo), NaN where never observed
        hi = np.stack([s[2] / s[3] for s in acc.values()])
        if t_unit == "sample":
            flat = lambda e: e.transpose(0, 2, 1).reshape(-1, e.shape[1])
            a, b = flat(hi), flat(lo)
            b = b[np.isfinite(b).all(1)]
        else:
            a, b = hi.mean(-1), np.nanmean(lo, -1)
        d = helper.cohens_d(a, b)
        # Long: average word response z-scored to the blank; latency = sample of its
        # absolute max after onset, enhanced if that peak is positive
        blank = lo.transpose(0, 2, 1).reshape(-1, lo.shape[1])
        z = (hi.mean(0) - np.nanmean(blank, 0)[:, None]) / np.nanstd(blank, 0, ddof=1)[:, None]
        k = np.abs(z).argmax(1)
        lat, enh = k * 1000.0 / (fc.LONGETAL or {}).get("envelope_hz", 100), z[np.arange(len(k)), k] > 0
        reg = dict(zip(ps[0]["labels"], ps[0]["reg_full"]))
        roi = fc.roi_of_reg_full([reg[l] for l in labels], lobe_of)
        for l, r, dd, lt, en in zip(labels, roi, d, lat, enh):
            if r is None or not np.isfinite(dd):
                continue
            grp = _long_region(reg[l]) if by_fine_label else r
            if grp is None:
                continue
            rows.append((sub, str(l), str(grp), str(r), float(dd), float(len(b)), float(len(a)), float(lt), float(en)))
    return rows


def collect_responsiveness(save_root, beh, band, lobe_of, n_sessions=None,
                           by_fine_label=False):
    """Per-(subject, electrode) t and p for the word contrast. t_unit "sample"
    (hilbert only) treats every envelope sample, not every trial, as an observation.
    """
    t_unit, combine = LONG.get("t_unit", "trial"), LONG.get("combine", "d_mean")
    dk, nk = ("cohens_d_samples", "_samples") if t_unit == "sample" else ("cohens_d", "")
    from scipy.stats import t as tdist

    d = power_dir(save_root, beh, band)
    files = sorted(d.glob("*_power.pkl"))[:n_sessions]
    rows = word_averaged_rows(files, lobe_of, t_unit, by_fine_label) if combine == "word_average" else []
    for f in (tqdm(files, desc="load responsiveness") if combine != "word_average" else []):
        try:
            p = fc.load_pickle(str(f))
        except Exception:
            continue
        if p.get(dk) is None or "n_lo" not in p:
            continue
        sub = str(p["sid"][0])
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        fine = np.asarray(p["reg_full"], dtype=object)
        nlo, nhi = float(p["n_lo" + nk]), float(p["n_hi" + nk])
        for lab, r, fl, dd in zip(p["labels"], roi, fine, p[dk]):
            # Check the RAW value BEFORE str(). reg_full is np.nan for contacts
            # with no anatomical label (white matter, outside brain, unmapped),
            # and str(np.nan) == "nan" - which silently became its own group and
            # topped the chart, because unlabelled contacts are common.
            raw = fl if by_fine_label else r
            if raw is None or (fc.LONGETAL and r is None):   # longetal: excluded lobes stay out of fine labels too
                continue
            if isinstance(raw, float) and not np.isfinite(raw):
                continue
            grp = str(raw).strip()
            if fc.LONGETAL and by_fine_label:   # Long et al.'s subregions, hemispheres pooled
                grp = _long_region(grp)
                if grp is None:
                    continue
            if grp.lower() in _UNLABELLED:
                continue
            if not np.isfinite(dd):
                continue
            rows.append((sub, str(lab), str(grp), str(r), float(dd), nlo, nhi, np.nan, np.nan))
    if not rows:
        raise SystemExit(f"no usable pickles in {d}")
    df = pd.DataFrame(rows,
                      columns=["sub", "label", "grp", "roi", "d", "n_lo", "n_hi", "latency", "peak_enh"])
    # one row per (subject, electrode): average d over that subject's sessions
    # (word_average rows are already one per electrode; the mean is a no-op).
    # `roi` rides along so a fine-label figure can still be coloured by lobe.
    df = (df.groupby(["sub", "label", "grp", "roi"], as_index=False)
            .agg(d=("d", "mean"), n_lo=("n_lo", "mean"), n_hi=("n_hi", "mean"),
                 latency=("latency", "mean"), peak_enh=("peak_enh", "mean")))
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


# Long et al. 2020 (paper text where given, else read off thesis Figs 2.6A/C): latency ms, % enhanced; (mean, sem)
LONG_PEAKS = {   # sem None = eyeballed mean, no error bar drawn
    "Middle Occipital Gyrus": ((243.3, 17.3), (92.9, 2.6)), "Cuneus": ((304.7, 26.6), (79, None)),
    "Fusiform Gyrus": ((352.2, 21.0), (92.4, 2.2)), "Lingual Gyrus": ((433.5, 51.7), (94.2, 3.3)),
    "Superior Parietal Lobule": ((535, None), (68, None)), "Inferior Temporal Gyrus": ((600, None), (61, None)),
    "Precentral Gyrus": ((610, None), (74, None)), "Precuneus": ((620, None), (66, None)),
    "Inferior Parietal Lobule": ((635, None), (53, None)), "Middle Temporal Gyrus": ((645, None), (61.5, None)),
    "Inferior Frontal Gyrus": ((648, None), (53.5, None)), "Middle Frontal Gyrus": ((648, None), (52, None)),
    "Postcentral Gyrus": ((662, None), (57, None)), "Superior Temporal Gyrus": ((665, None), (59, None)),
    "Superior Frontal Gyrus": ((690, None), (35, None)), "Medial Frontal Gyrus": ((740, None), (53, None)),
}


def plot_long_peaks(df, out_dir, tag, alpha, min_resp):
    """Long's Fig 2.6A/C: mean latency and % enhanced of the RESPONSIVE electrodes
    per subregion (mean +- SEM over electrodes), Juee as bars, Long's as red
    points. Regions with >= min_resp responsive electrodes."""
    import matplotlib.pyplot as plt
    r = df[df["p"] < alpha].dropna(subset=["latency"])
    g = r.groupby("grp")
    st = pd.DataFrame({"n": g.size(), "lat": g["latency"].mean(), "lat_sem": g["latency"].sem(),
                       "enh": 100 * g["peak_enh"].mean()})
    st["enh_sem"] = np.sqrt(st["enh"] * (100 - st["enh"]) / st["n"])
    st = st[st["n"] >= min_resp]
    if st.empty:
        return
    st["roi"] = g["roi"].agg(lambda v: v.mode().iat[0]).reindex(st.index)
    st.to_csv(join(out_dir, f"long_peaks_{tag}.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.35 * len(st) + 1.8))
    for ax, (col, j, lab) in zip(axes, (("lat", 0, "latency (ms)"), ("enh", 1, "% enhanced (peak > 0)"))):
        s = st.sort_values(col)
        y = np.arange(len(s))
        colors = [fc.LOBE_COLORS.get(str(x).split("-")[-1], "0.6") for x in s["roi"]]
        ax.barh(y, s[col], xerr=s[col + "_sem"], color=colors, alpha=0.8, label="Juee")
        ref = [LONG_PEAKS.get(gname, (None, None))[j] for gname in s.index]
        yy = [i for i, v in zip(y, ref) if v]
        ax.errorbar([v[0] for v in ref if v], yy, xerr=[v[1] or 0 for v in ref if v], fmt="D", color="crimson",
                    ms=4, label="Long et al.")
        ax.set_yticks(y, [f"{gname} ({n})" for gname, n in zip(s.index, s["n"])])
        ax.set_xlabel(lab)
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Responsive electrodes")
    fig.tight_layout()
    path = join(out_dir, f"long_peaks_{tag}.png")
    fig.savefig(path, dpi=160); plt.close(fig)
    print(f"[long peaks] wrote {path}")


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
    axes[0].set_yticklabels([fc.pretty_roi(g) for g in order], fontsize=8)
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

    axes[0].set_title(f"{band} power, "
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
                                 label=fc.pretty_roi(lobe)))
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
    args = parse_args()
    root_dir, save_root = fc.resolve_roots(args)
    c = fc.contrast(args.beh)
    sp = fc.PREPOST_SPEC[args.beh]
    print(f"[setup] beh={args.beh}  band={args.band} {fc.bands[args.band]} Hz  ({fc.POWER_MODE})")
    print(f"[setup] {c['lo_label']} (lo) {sp['pre_win']} vs {c['hi_label']} (hi) {sp['post_win']}")

    if args.stage in ("compute", "both"):
        fc.run_compute_stage(
            run_sess_power, desc="ROI power", root_dir_=root_dir,
            n_sessions=args.n_sessions, n_subjects=args.n_subjects,
            workers=args.workers, save_root=save_root, beh=args.beh, band=args.band, root_dir=root_dir,
            simulation_tag=args.simulation_tag)

    if args.stage in ("plot", "both"):
        run_plot_stage(save_root, args.beh, args.band, args)


if __name__ == "__main__":
    main()
