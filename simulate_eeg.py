"""simulate_eeg.py -- synthetic EEG for the pipeline validity checks.

helper.get_eeg calls replace_w_simulated_EEG when --simulation-tag is set: the
session's real montage, regions and events are kept (so the whole real pipeline
is exercised) and only the signal is replaced. Two data-generating processes,
parameterised in config/simulation_config.yaml:

  hg_power  pink noise + band-limited noise on every channel; on target-lobe
            channels the band part is scaled by hg_gain from word onset on.
  hg_ppc    Morlet bursts at oscillation_frequency, phases drawn per event from
            a wrapped normal: target-lobe pairs at target_ppc0 (pre) / target_ppc1
            (post), all other pairs at global_ppc; plus pink noise.

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


def generate_pink_noise(n: int, amplitude: float, exponent: float) -> NDArrayAny:
    """1/f^exponent noise of length n (global np.random state)."""
    m = n + (n % 2)
    scales = np.linspace(0, 0.5, m // 2 + 1)[1:] ** (-exponent / 2)
    f = np.concatenate([[0], np.random.normal(scale=scales) * np.exp(2j * np.pi * np.random.random(m // 2))])
    sigma = np.sqrt(2 * np.sum(scales ** 2)) / m
    return (amplitude * np.fft.irfft(f) / sigma)[:n]


def _pink(shape: tuple[int, ...], p: dict[str, Any]) -> NDArrayAny:
    """Independent pink noise per (event, channel) row."""
    if not p["pinknoise_amplitude"]:
        return np.zeros(shape)
    rows = [generate_pink_noise(shape[-1], p["pinknoise_amplitude"], p["pinknoise_exponent"])
            for _ in range(int(np.prod(shape[:-1])))]
    return np.reshape(rows, shape)


def _target_lobe_mask(regionalizations: Sequence[Any], lobes: Sequence[str]) -> NDArrayAny:
    """True for channels whose region ('L frontal pole' ...) is in one of `lobes` (Burke lobes)."""
    lobe_of = pd.read_csv(Path(__file__).resolve().parent / 'region_to_burke_lobe.csv').set_index('region')['burke_lobe']
    return np.array([isinstance(r, str) and lobe_of.get(r.split(' ', 1)[1]) in lobes for r in regionalizations])


def _phase_cov(target: NDArrayAny, target_ppc: float, global_ppc: float) -> NDArrayAny:
    """Wrapped-normal covariance whose pairwise PPC (= exp(-sigma^2)) is target_ppc
    between target channels and global_ppc otherwise."""
    ppc = np.where(np.outer(target, target), target_ppc, global_ppc)
    np.fill_diagonal(ppc, 1.0)
    var = np.triu(-np.log(ppc), 1)                    # pair phase-difference variance
    cov = np.full(ppc.shape, var.max()) - (var + var.T) / 2
    np.linalg.cholesky(cov)                           # raises if not positive definite
    return cov


def _bursts(n_ev: int, cov: NDArrayAny, t_s: NDArrayAny, p: dict[str, Any]) -> NDArrayAny:
    """(n_ev, n_ch, n_t) real Morlet bursts centred in the clip, per-event phases
    from a wrapped normal with covariance `cov`."""
    f = p["oscillation_frequency"]
    sig = p["morlet_reps"] / (2 * np.pi * f)
    amp = p["wavelet_amplitude"] * np.sqrt(2) / np.sqrt(sig * np.sqrt(np.pi))   # sqrt(2): real part keeps unit power
    phases = np.random.multivariate_normal(np.full(len(cov), np.pi), cov, size=n_ev) % (2 * np.pi) - np.pi
    tt = t_s[None, None, :] - (phases / (2 * np.pi * f))[:, :, None]
    return amp * np.exp(-tt ** 2 / (2 * sig ** 2)) * np.cos(2 * np.pi * f * tt)


def replace_w_simulated_EEG(eeg: TimeSeries, dfrow: pd.Series, condition_mask: NDArrayAny,
                            simulation_tag: str) -> TimeSeries:
    """Same shape/coords as `eeg` (time in ms), signal from `simulation_tag`.
    condition_mask True = post-onset events (hg_ppc: target_ppc1), False = pre."""
    p = simulation_parameters[simulation_tag]
    np.random.seed(zlib.crc32(np.ascontiguousarray(eeg.data).tobytes()))
    target = _target_lobe_mask(regionalize_electrodes_by_type(get_pairs(dfrow)), p["target_lobes"])
    t_ms = np.asarray(eeg.time, float)
    data = _pink(eeg.shape, p)
    if p["data_generating_process"] == "hg_power":
        import mne
        hg = p["hg_amplitude"] * mne.filter.filter_data(np.random.standard_normal(eeg.shape),
                                                        float(eeg.samplerate), *p["band"], verbose=False)
        hg[:, target] *= np.where(t_ms >= 0, p["hg_gain"], 1.0)
        data += hg
    else:   # hg_ppc
        t_s = (t_ms - (t_ms[0] + t_ms[-1]) / 2) / 1000.0
        mask = np.asarray(condition_mask, bool)
        for m, key in ((~mask, "target_ppc0"), (mask, "target_ppc1")):
            if m.any():
                data[m] += _bursts(int(m.sum()), _phase_cov(target, p[key], p["global_ppc"]), t_s, p)
    return eeg.copy(data=data)
