"""simulate_eeg.py -- synthetic EEG for the pipeline validity checks.

helper.get_eeg calls replace_w_simulated_EEG when --simulation-tag is set: the
session's real montage, regions and events are kept (so the whole real pipeline
is exercised) and only the signal is replaced. Data-generating processes,
parameterised in config/simulation_config.yaml (each on top of pink noise):

  hg_power     band-limited noise on every channel; on target-lobe channels the
               band part is scaled by hg_gain from word onset on.
  osc_lag      sustained oscillation, phases drawn per event from a wrapped normal
               (target pairs at target_ppc0 pre / target_ppc1 post, all other pairs
               at global_ppc), every other target channel lagged by phase_lag
               -> ciPLV (alpha) should rise.
  hg_envelope  independent band-limited carriers; on post events target
               channels share a slow amplitude envelope -> AEC-c should rise.
  leak         zero-lag common source on target channels, post events only
               (volume conduction) -> ciPLV and AEC-c should stay at 0.

A list of processes (e.g. [osc_lag, hg_envelope]) adds them up. hg_envelope option
`equal_conditions: true` shares the envelope on pre events too (equal coupling, null).

Optional `active_ms: [a, b]` restricts the post-onset effect (osc_lag, hg_envelope,
hg_power) to a <= t < b ms, to test the epoch network's / decoder's timing.

hg_power options: `onset_ms: {lobe: ms}` starts each target lobe's gain at its own time
(latency order); `recalled_only: true` applies the gain only to words later recalled
(the recall decoder).

Optional `line_amplitude` adds a 60 Hz mains sinusoid (random phase per event,
shared by all channels), to test the notch filter on line noise.

The random seed is a CRC of the real clip, so reruns reproduce the same signal.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml  # pyright: ignore[reportMissingTypeStubs]
from ptsa.data.timeseries import TimeSeries  # pyright: ignore[reportMissingTypeStubs]

from helper import get_pairs, regionalize_electrodes_by_type

NDArrayAny = np.ndarray
_SIM_CONFIG_PATH = Path(__file__).parent / "config" / "simulation_config.yaml"


def _load_simulation_parameters() -> dict[str, dict[str, Any]]:
    """Named parameter sets; each `sweeps` value becomes its own tag `<sweep>_<value>`."""
    with open(_SIM_CONFIG_PATH) as f:
        params = yaml.safe_load(f)
    for name, sw in params.pop("sweeps", {}).items():
        for v in sw["values"]:
            params[f"{name}_{v}"] = {**sw["params"], **{k: v for k in sw["keys"]},
                                     "sweep": name, "planted": v}
    return params


simulation_parameters = _load_simulation_parameters()
# {sweep name: [its tags in value order]}, for the Snakefile and plot_recovery.py
SWEEPS: dict[str, list[str]] = {}
for _tag, _p in simulation_parameters.items():
    if "sweep" in _p:
        SWEEPS.setdefault(_p["sweep"], []).append(_tag)


def generate_pink_noise(n: int, amplitude: float, exponent: float, n_ref: int = 1000) -> NDArrayAny:
    """1/f^exponent noise of length n (global np.random state). Scaled so its spectrum is the
    same for any n (amplitude = total std at n_ref samples): normalising each clip to a fixed
    total variance instead moved power out of the analysed band in longer clips (the word-on
    clip is longer than the word-off clip: a broadband word-on decrease in the null)."""
    m = n + (n % 2)
    scales = np.linspace(0, 0.5, m // 2 + 1)[1:] ** (-exponent / 2)
    f = np.concatenate([[0], np.random.normal(scale=scales) * np.exp(2j * np.pi * np.random.random(m // 2))])
    ref = np.linspace(0, 0.5, n_ref // 2 + 1)[1:] ** (-exponent / 2)
    sigma = np.sqrt(2 * np.sum(ref ** 2)) / n_ref * np.sqrt(n_ref / m)
    return (amplitude * np.fft.irfft(f) / sigma)[:n]


def _pink(shape: tuple[int, ...], p: dict[str, Any]) -> NDArrayAny:
    """Independent pink noise per (event, channel) row."""
    if not p["pinknoise_amplitude"]:
        return np.zeros(shape)
    rows = [generate_pink_noise(shape[-1], p["pinknoise_amplitude"], p["pinknoise_exponent"])
            for _ in range(int(np.prod(shape[:-1])))]
    return np.reshape(rows, shape)


def _channel_lobes(regionalizations: Sequence[Any]) -> NDArrayAny:
    """Burke lobe of each channel's region ('L frontal pole' -> 'frontal'), None if unmapped."""
    lobe_of = pd.read_csv(Path(__file__).resolve().parent / 'region_to_burke_lobe.csv').set_index('region')['burke_lobe']
    return np.array([lobe_of.get(r.split(' ', 1)[1]) if isinstance(r, str) else None for r in regionalizations], dtype=object)


def _phase_cov(target: NDArrayAny, target_ppc: float, global_ppc: float) -> NDArrayAny:
    """Wrapped-normal covariance whose pairwise PPC (= exp(-sigma^2)) is target_ppc
    between target channels and global_ppc otherwise."""
    ppc = np.where(np.outer(target, target), target_ppc, global_ppc)
    np.fill_diagonal(ppc, 1.0)
    var = np.triu(-np.log(ppc), 1)                    # pair phase-difference variance
    cov = np.full(ppc.shape, var.max()) - (var + var.T) / 2
    np.linalg.cholesky(cov)                           # raises if not positive definite
    return cov


def replace_w_simulated_EEG(eeg: TimeSeries, dfrow: pd.Series, condition_mask: NDArrayAny,
                            simulation_tag: str, events: pd.DataFrame | None = None) -> TimeSeries:
    """Same shape/coords as `eeg` (time in ms), signal from `simulation_tag`.
    condition_mask True = post-onset events (osc_lag: target_ppc1), False = pre; `events`
    (indexed by eeg.event) gives correct_recall for `recalled_only`."""
    p = simulation_parameters[simulation_tag]
    np.random.seed(zlib.crc32(np.ascontiguousarray(eeg.data).tobytes()))
    lobes = _channel_lobes(regionalize_electrodes_by_type(get_pairs(dfrow)))
    target = np.isin(lobes, p["target_lobes"])
    t_ms = np.asarray(eeg.time, float)
    data = _pink(eeg.shape, p)
    win = ((t_ms >= p["active_ms"][0]) & (t_ms < p["active_ms"][1])) if "active_ms" in p \
        else np.ones(t_ms.size, bool)   # where post-onset coupling is on
    for dgp in np.atleast_1d(p["data_generating_process"]):   # several processes add up
        if dgp == "hg_power":
            import mne
            hg = p["hg_amplitude"] * mne.filter.filter_data(np.random.standard_normal(eeg.shape),
                                                            float(eeg.samplerate), *p["band"], verbose=False)
            onset = np.array([p.get("onset_ms", {}).get(lb, 0) for lb in lobes])[target]
            gain = np.where((t_ms[None, :] >= onset[:, None]) & win[None, :], p["hg_gain"], 1.0)   # (target ch, time)
            rows = (events.loc[np.asarray(eeg.event), "correct_recall"].fillna(0).to_numpy(bool)
                    if p.get("recalled_only") else np.ones(eeg.shape[0], bool))
            hg[np.ix_(rows, target)] *= gain
            data += hg
        elif dgp == "osc_lag":
            # sustained oscillation; every other target channel shifted by phase_lag
            # (ciPLV sees only the lagged pairs); outside `win` post events are uncoupled
            mask = np.asarray(condition_mask, bool)
            lag = np.where(target, np.cumsum(target) % 2 * p["phase_lag"], 0.0)
            def draw(key, n):   # (n, n_ch, 1) per-event phases
                cov = _phase_cov(target, p[key], p["global_ppc"])
                return (np.random.multivariate_normal(np.zeros(len(cov)), cov, size=n) + lag)[:, :, None]
            for m, key in ((~mask, "target_ppc0"), (mask, "target_ppc1")):
                if m.any():
                    n = int(m.sum())
                    ph = np.where(win, draw(key, n), draw("target_ppc0", n))
                    data[m] += p["osc_amplitude"] * np.cos(2 * np.pi * p["oscillation_frequency"]
                                                           * t_ms[None, None, :] / 1000.0 + ph)
        elif dgp == "hg_envelope":
            # independent band-limited carriers; on post events the target channels
            # share one slow (< env_hz) amplitude modulation, otherwise independent
            import mne
            sf = float(eeg.samplerate)
            carrier = mne.filter.filter_data(np.random.standard_normal(eeg.shape), sf, *p["band"], verbose=False)
            # slow envelope filtered on a 2 s-padded segment, then cropped: the low-pass
            # is longer than the shorter clips, so its edges must not reach the clip
            pad = int(2 * sf)
            env = mne.filter.filter_data(np.random.standard_normal((*eeg.shape[:-1], eeg.shape[-1] + 2 * pad)),
                                         sf, None, p["env_hz"], verbose=False)[..., pad:-pad]
            # equal_conditions: shared on pre events too (equal coupling, zero contrast)
            mask = np.ones(eeg.shape[0], bool) if p.get("equal_conditions") else np.asarray(condition_mask, bool)
            own = env[np.ix_(mask, target)]
            env[np.ix_(mask, target)] = np.where(win, own[:, :1], own)   # first target channel's envelope, shared
            env /= env.std(axis=-1, keepdims=True)
            data += p["hg_amplitude"] * carrier * np.exp(p["env_depth"] * env)
        elif dgp == "leak":
            # zero-lag common source added to target channels on post events only:
            # volume conduction; ciPLV and AEC-c should not respond
            mask = np.asarray(condition_mask, bool)
            src = _pink((int(mask.sum()), 1, eeg.shape[-1]), p)
            data[np.ix_(mask, target)] += p["leak_gain"] * src
        else:
            raise ValueError(f"unknown data_generating_process {dgp!r}")
    if p.get("line_amplitude"):   # mains: one random phase per event, common to all channels
        ph = np.random.uniform(0, 2 * np.pi, (eeg.shape[0], 1, 1))
        data += p["line_amplitude"] * np.sin(2 * np.pi * 60.0 * t_ms[None, None, :] / 1000.0 + ph)
    return eeg.copy(data=data)
