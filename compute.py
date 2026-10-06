"""compute.py -- single-pass per-session compute: load each session's EEG once,
write every power and synchrony pickle the plot stages read.

Per session the word-off (PRE_WORD, -700..-100 ms) and word-on (WORD, 0..1600 ms)
clips are loaded once, each widened by `real_data_buffer_ms` of real data, and
shared by:

  synchrony  alpha ciPLV (multitaper) and high-gamma AEC-c (70-110 Hz Hilbert),
             main window + 200 ms epochs  -> <save_root>/word_on/fc_mats/diff/<band>/
             (fc.compute_prepost_separate, as build_roi_synchrony)
  power      alpha: multitaper over 8-13 Hz           -> word_on/power/alpha/
             spectrum: multitaper, 10 log bins 5-100   -> word_on/power/spectrum/
             high gamma: Hilbert in 10 Hz sub-bands 70-150 Hz, each normalized by
             its word-off mean, then averaged (broadband HFA with every frequency
             weighted equally)                          -> word_on/power/high_gamma/
             with a 50 ms-bin time course and per-trial envelopes at 10 ms for
             latency analyses

Every power pickle also keeps per-trial power and each word's preceding blank
(ms since the previous word's offset; NaN for a list's first word), for the
baseline (ISI-split) checks.

    python compute.py [--n-subjects K] [--workers N] [--simulation-tag TAG]
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

import fc_comparison_functions as fc
import helper
from build_roi_power import FREQ_EDGES, band_power, band_power_bins, power_dir
from build_roi_synchrony import cond_dir

BEH = "word_on"
SYNC = {"alpha": "ciplv", "high_gamma": "aec_c"}
HG_SUBBANDS = [(f, f + 10) for f in range(70, 150, 10)]
ENV_STEP_MS = 10          # stored per-trial high-gamma envelope resolution
VERSION = "single_pass_1"


def _crop(eeg: Any, win: tuple[float, float]) -> np.ndarray:
    t = np.asarray(eeg.time, float)
    return np.asarray(eeg.data, float)[..., (t >= win[0]) & (t < win[1])]


def _blank_before(events: pd.DataFrame) -> np.ndarray:
    """ms from the previous word's offset to each word's onset (same list), NaN for the first."""
    e = events[["trial", "onset", "duration"]].astype(float)
    prev_off = (e["onset"] + e["duration"]).groupby(e["trial"]).shift(1)
    return ((e["onset"] - prev_off) * 1000).to_numpy()


def _cohens_d(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Per-channel Cohen's d (hi vs lo trials) over the trial axis; (E, C) -> (C,)."""
    return np.asarray(helper.cohens_d(hi, lo), float).reshape(hi.shape[1:])


def hg_envelope(x: np.ndarray, sf: float, n_buf: int) -> np.ndarray:
    """(E, C, T) sub-band Hilbert power, buffer cropped, NOT yet normalized:
    list of per-sub-band arrays."""
    out = []
    for lo, hi in HG_SUBBANDS:
        a = fc.band_analytic(x, sf, lo, hi)
        out.append((np.abs(a) ** 2)[..., n_buf:a.shape[-1] - n_buf])
    return out


def run_session(dfrow: pd.Series, save_root: str, root_dir: str,
                simulation_tag: str | None = None) -> str:
    fc.root_dir = helper.root_dir = root_dir
    sid = fc.ftag(dfrow)
    outs = {b: cond_dir(save_root, BEH, "diff", b) / f"{sid}_fc_mats.pkl" for b in SYNC}
    pows = {b: power_dir(save_root, BEH, b) / f"{sid}_power.pkl" for b in ("alpha", "high_gamma", "spectrum")}
    if all(p.exists() for p in [*outs.values(), *pows.values()]) and \
            fc.load_pickle(str(outs["alpha"])).get("version") == VERSION:   # small file; written first
        return f"{sid}: cached"

    ev = fc.load_events(dfrow, BEH)
    if ev is None:
        return f"{sid}: no events"
    pairs = helper.get_pairs(dfrow)
    om = fc.make_overlap_mask(pairs, n_ch=len(pairs))
    reg_full = np.asarray(helper.regionalize_electrodes_by_type(pairs), dtype=object)
    labels = pairs["label"].astype(str).to_numpy()
    meta = {"sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"])), "reg_full": reg_full,
            "version": VERSION, "buffer_ms": fc.REAL_DATA_BUFFER_MS, "subtract_erp": fc.SUBTRACT_ERP,
            "notch_harmonics_up_to_hz": fc.NOTCH_HARMONICS_UP_TO_HZ, "simulation_tag": simulation_tag}

    clips: dict[Any, Any] = {}   # (window) -> (eeg, mask): each clip loaded once
    def load(d, events, window, buf, sim):
        k = tuple(window)
        if k not in clips:
            clips[k] = fc.get_beh_eeg(d, events, window, buf, sim)
        return clips[k]

    # --- synchrony (both bands share the two loaded clips) --------------------
    for band, m in SYNC.items():
        mat = fc.compute_prepost_separate(dfrow, BEH, ev, [m], *fc.bands[band], om,
                                          fc.REAL_DATA_BUFFER_MS, simulation_tag=simulation_tag,
                                          load_fn=load)
        out = {**meta, m: np.squeeze(mat[m]["diff"]), f"{m}_epochs": mat[m]["epochs"]}
        os.makedirs(outs[band].parent, exist_ok=True)
        fc.save_pickle(str(outs[band]), out)

    # --- power from the same clips -------------------------------------------
    spec = fc.PREPOST_SPEC[BEH]
    pre_win, post_win = tuple(spec["pre_win"]), tuple(spec["post_win"])
    (pre_eeg, _), (post_eeg, _) = [v for k, v in clips.items() if k == pre_win][0], \
        [v for k, v in clips.items() if k != pre_win][0]
    sf = float(post_eeg.samplerate)
    n_buf = int(round(fc.REAL_DATA_BUFFER_MS * sf / 1000.0))
    word_ev = ev[np.asarray(ev.attrs["mask"], bool)].reset_index(drop=True)
    blank = _blank_before(word_ev)
    blank_lo = blank[np.asarray(pre_eeg.event, int)]
    blank_hi = blank[np.asarray(post_eeg.event, int)]
    lo, hi = fc.equalize_time_length(_crop(pre_eeg, pre_win), _crop(post_eeg, post_win))
    trial = {"labels": labels, "n_lo": len(lo), "n_hi": len(hi), "n_win_samples": lo.shape[-1],
             "lo_win": pre_win, "hi_win": post_win, "blank_ms_lo": blank_lo, "blank_ms_hi": blank_hi}

    def save(band, p_lo, p_hi, extra):
        out = {**meta, **trial, "band": band, "fmin": fc.bands[band][0], "fmax": fc.bands[band][1],
               "pow_lo": np.nanmean(p_lo, 0), "pow_hi": np.nanmean(p_hi, 0),
               "cohens_d": _cohens_d(p_hi, p_lo),
               "trial_pow_lo": p_lo.astype(np.float32), "trial_pow_hi": p_hi.astype(np.float32), **extra}
        os.makedirs(pows[band].parent, exist_ok=True)
        fc.save_pickle(str(pows[band]), out)

    bw = fc.MT_BANDWIDTH
    save("alpha", band_power(lo, sf, *fc.bands["alpha"], bw), band_power(hi, sf, *fc.bands["alpha"], bw),
         {"measure": "power", "fc_mode": "multitaper", "mt_bandwidth": bw})
    b_lo, b_hi = band_power_bins(lo, sf, bw), band_power_bins(hi, sf, bw)   # (E, C, n_bins)
    save("spectrum", b_lo.mean(-1), b_hi.mean(-1),
         {"measure": "power", "fc_mode": "multitaper", "mt_bandwidth": bw, "freq_edges": FREQ_EDGES,
          "cohens_d_freq": np.stack([_cohens_d(b_hi[..., k], b_lo[..., k]) for k in range(b_lo.shape[-1])], -1)})

    # high gamma: sub-band Hilbert power, each sub-band / its word-off mean (per channel), averaged
    t_pre = np.asarray(pre_eeg.time, float)[n_buf:len(pre_eeg.time) - n_buf]
    t_post = np.asarray(post_eeg.time, float)[n_buf:len(post_eeg.time) - n_buf]
    in_pre = (t_pre >= pre_win[0]) & (t_pre < pre_win[1])
    e_pre, e_post = hg_envelope(np.asarray(pre_eeg.data, float), sf, n_buf), \
        hg_envelope(np.asarray(post_eeg.data, float), sf, n_buf)
    base = [np.nanmean(p[..., in_pre], axis=(0, 2), keepdims=True) for p in e_pre]
    env_pre = np.mean([p / b for p, b in zip(e_pre, base)], axis=0)[..., in_pre]
    env_post = np.mean([p / b for p, b in zip(e_post, base)], axis=0)
    in_post = (t_post >= post_win[0]) & (t_post < post_win[1])
    p_lo, p_hi = env_pre.mean(-1), env_post[..., in_post].mean(-1)
    tb = fc.TIME_BIN_MS
    edges = np.arange(post_win[0], post_win[1] + 1e-9, tb)
    d_bins = np.stack([_cohens_d(env_post[..., (t_post >= a) & (t_post < b)].mean(-1), p_lo)
                       for a, b in zip(edges[:-1], edges[1:])], -1)
    step = int(round(ENV_STEP_MS * sf / 1000.0))
    ds = lambda e: e[..., :e.shape[-1] // step * step].reshape(*e.shape[:2], -1, step).mean(-1).astype(np.float16)
    save("high_gamma", p_lo, p_hi,
         {"measure": "power", "fc_mode": "hilbert_subbands", "subbands": HG_SUBBANDS,
          "cohens_d_bins": d_bins, "bin_centers_ms": (edges[:-1] + edges[1:]) / 2 - post_win[0],
          "time_bin_ms": tb, "env_step_ms": ENV_STEP_MS,
          "env_lo": ds(env_pre), "env_lo_t0_ms": float(t_pre[in_pre][0]),
          "env_hi": ds(env_post), "env_hi_t0_ms": float(t_post[0])})
    return f"{sid}: {len(labels)} elec, {len(lo)}+{len(hi)} events"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    fc.add_common_args(p)
    args = p.parse_args()
    root_dir, save_root = fc.resolve_roots(args)
    print(f"[setup] buffer {fc.REAL_DATA_BUFFER_MS} ms, notch harmonics up to "
          f"{fc.NOTCH_HARMONICS_UP_TO_HZ} Hz, version {VERSION}")
    fc.run_compute_stage(run_session, desc="compute", root_dir_=root_dir,
                         n_sessions=args.n_sessions, n_subjects=args.n_subjects, workers=args.workers,
                         save_root=save_root, root_dir=root_dir, simulation_tag=args.simulation_tag)


if __name__ == "__main__":
    main()
