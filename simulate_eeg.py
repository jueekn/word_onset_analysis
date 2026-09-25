"""simulate_eeg.py — synthetic EEG generators for FC validation simulations.

`simulation_parameters` registers named parameter sets used by the standalone
sim runners (sim/run_*_simulation.py) and the production pipeline path
(build_subj_mat.py --simulation-tag). Lower-level generators build covariance
matrices, sample wrapped-multivariate-normal phases, and assemble per-event
EEG as Morlet-wavelet bursts + optional pink noise.
"""
from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr  # pyright: ignore[reportMissingTypeStubs]
from time import time
from ptsa.data.timeseries import TimeSeries  # pyright: ignore[reportMissingTypeStubs]
import matplotlib.pyplot as plt

from cstat import ppc_to_wrapped_normal_sigma, sample_wrapped_multivariate_normal, ppc
from matrix_operations import apply_function_expand_dims, symmetrize, is_square, is_symmetric, is_positive_definite, strict_triu, sort_array_across_order

NDArrayAny = npt.NDArray[Any]
# helper imports this module lazily (inside get_eeg), so importing from helper here is safe.
from misc import duration_to_samples, get_time_offset
from wavelet import Wavelet
from helper import get_pairs, regionalize_electrodes_by_type
from pathlib import Path
import yaml  # pyright: ignore[reportMissingTypeStubs]


_SIM_CONFIG_PATH = Path(__file__).parent / "config" / "simulation_config.yaml"


def _load_simulation_parameters() -> dict[Any, Any]:
    """Load named sim configs from config/simulation_config.yaml.

    The file is the canonical store; adding/editing a sim is a YAML edit,
    not a Python edit. The runner reads simulation_parameters[<tag>] which
    is populated by this loader at module import time.

    Adds the legacy sentinels ('', None) back as aliases for 'standard' so
    callers that previously passed the empty string / None still resolve
    to a no-sim entry.
    """
    if not _SIM_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"simulation_config not found at {_SIM_CONFIG_PATH}; "
            "did you delete the file?"
        )
    with open(_SIM_CONFIG_PATH) as f:
        params = yaml.safe_load(f) or {}
    if not isinstance(params, dict):
        raise TypeError(
            f"{_SIM_CONFIG_PATH} must be a YAML mapping, got {type(params)}"
        )
    for name, sw in params.pop("sweeps", {}).items():   # one tag per swept value
        for v in sw["values"]:
            params[f"{name}_{v}"] = {**sw["params"], **{k: v for k in sw["keys"]},
                                     "sweep": name, "planted": v}
    # Legacy sentinels: empty-string and None keys both meant
    # "no simulation, use empirical EEG". Re-add them so any code path
    # that still references those keys keeps working.
    if "standard" in params:
        params[""] = params["standard"]
        params[None] = params["standard"]
    return params


simulation_parameters = _load_simulation_parameters()
# {sweep name: [its tags in value order]}, for the Snakefile and plot_recovery.py
SWEEPS = {}
for _tag, _p in simulation_parameters.items():
    if isinstance(_p, dict) and "sweep" in _p:
        SWEEPS.setdefault(_p["sweep"], []).append(_tag)

AVAILABLE_SIMULATIONS = list(simulation_parameters.keys())
NULL_SIMULATION_TAGS = ['null_connectivity', 'equal_oscillation_null']


# AVAILABLE_SIMULATIONS = [
#             'standard', '', None,  # use empirical EEG (no simulation)
#             'null_connectivity',  # no functional connectivity (FC) with EEG simulated with pink noise
#             'strong_oscillation_only',  # idealized FC with noiseless wavelet oscillations giving non-physiologically strong FC
#             'oscillation_only',  # idealized FC with noiseless wavelet oscillations having realistic FC
#             'noisy_oscillation',  # wavelet oscillations plus additive pink noise
# ]


def generate_pink_noise(
    N_samples: int,
    pinknoise_amplitude: float,
    pinknoise_exponent: float,
    rng: None | int | np.random.Generator = None,
) -> NDArrayAny:
    """Generate 1/f^exponent pink noise.

    Parameters
    ----------
    rng : None, int, or np.random.Generator
        If None, draws from the legacy module-level ``np.random`` state
        (backward-compatible). Otherwise, an int seed or pre-built Generator;
        ``np.random.default_rng(rng)`` is used to normalize.
    """
    out_n = N_samples
    n = N_samples
    if (n % 2) == 1:
        n += 1
    scales = np.linspace(0, 0.5, n//2+1)[1:]
    scales = scales**(-pinknoise_exponent/2)
    if rng is None:
        normal_draws = np.random.normal(scale=scales)
        uniform_draws = np.random.random(n//2)
    else:
        rng = np.random.default_rng(rng)
        normal_draws = rng.normal(scale=scales)
        uniform_draws = rng.random(n//2)
    pinkf = normal_draws * np.exp(2j*np.pi*uniform_draws)
    fdata = np.concatenate([[0], pinkf])
    sigma = np.sqrt(2*float(np.sum(scales**2))) / n  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    data = pinknoise_amplitude * np.fft.irfft(fdata)/sigma
    return data[:out_n]


def get_random_state_by_time() -> int:
    """Time-based 32-bit pseudo-random state. Used as a legacy seed fallback."""
    return int(time() * 1e6) % (2**32 - 1)


def sample_phases(
    mean: NDArrayAny,
    cov: NDArrayAny,
    samples: int = 1,
    random_state: int | None = None,
    rng: None | int | np.random.Generator = None,
) -> NDArrayAny:
    """Sample phases from a wrapped multivariate normal.

    Parameters
    ----------
    random_state : int or None
        Legacy seed argument. If given, it is forwarded to ``np.random.seed``
        (mutates global state — kept for backward compatibility). Prefer
        ``rng``.
    rng : None, int, or np.random.Generator
        Preferred seeding interface. When provided, draws are routed through a
        local ``np.random.Generator`` and the global state is left untouched.
    """
    # kappa = plv_to_kappa(population_ppc)
    # phases = vonmises.rvs(kappa=kappa, loc=phase_offset, size=samples,
    #                       random_state=random_state if random_state is not None else get_random_state_by_time())  # random_state=None is deterministic

    if rng is not None:
        rng = np.random.default_rng(rng)
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        draws = rng.multivariate_normal(mean=mean + np.pi, cov=cov, size=samples)
        phases = draws % (2 * np.pi) - np.pi
        return phases

    if random_state is not None:
        np.random.seed(random_state)
    phases = sample_wrapped_multivariate_normal(mean=mean, cov=cov, size=samples)
    return phases


def generate_event_wavelets(
    phases: NDArrayAny,
    duration_s: float = 1,
    frequency: float = 5,
    phase_offset: float = 0,
    morlet_reps: int = 5,
    sampling_rate: float = 1000,
    amplitude: float | None = None,
    wavelet_type: str = 'real',
    normalize: bool = True,
) -> NDArrayAny:
    """Per-channel Morlet bursts whose phase offsets come from `phases`.

    Args:
        phases: per-channel target phase offsets (radians).
        duration_s: epoch duration in seconds.
        frequency: center frequency of the Morlet (Hz).
        phase_offset: unused (kept for API compatibility).
        morlet_reps, sampling_rate, amplitude: forwarded to Wavelet().
        wavelet_type: 'real' / 'imaginary' / 'full'.
        normalize: if True and wavelet_type ∈ {real, imaginary}, scale by
            sqrt(2) so unity power is preserved.
    """
    # generate wavelets centered at middle of epoch
    wavelet = Wavelet(fmin=2,
                      fmax=200,
                      fnum=1,
                      sampling_rate=sampling_rate,
                      morlet_reps=morlet_reps,
                      amplitude=amplitude,
                      tmin=-duration_s / 2,
                      tmax=duration_s / 2)

    offsets = get_time_offset(phases, frequency)
    def get_offset_wavelet(offset: float) -> NDArrayAny:
        w = wavelet.Morlet(wavelet.tvals - offset, wavelet.morlet_reps, frequency)
        if wavelet_type == 'real':
            w = w.real
        elif wavelet_type == 'imaginary':
            w = w.imag
        elif wavelet_type == 'full':
            pass
        else:
            raise ValueError
        if normalize and (wavelet_type in ['real', 'imaginary']):
            # normalize to unity power (dropping real or imaginary component halves power)
            w *= np.sqrt(2)
        return w
    wavelets = apply_function_expand_dims(offsets, get_offset_wavelet)  # pyright: ignore[reportArgumentType]
    return wavelets

def sample_eeg(
    n_events: int,
    n_channels: int,
    phase_mean: NDArrayAny,
    phase_covariance: NDArrayAny,
    sample_rate_Hz: float,
    start_time_ms: float,
    duration_ms: float,
    wavelet_amplitude: float = 1,
    connectivity_frequency_Hz: float = 5,
    morlet_reps: int = 5,
    pinknoise_amplitude: float = 1,
    pinknoise_exponent: float = 1,
    rng: None | int | np.random.Generator = None,
) -> TimeSeries:
    """Generate one subject's synthetic EEG as a PTSA TimeSeries.

    For each event, draw a phase vector from N(phase_mean, phase_covariance)
    wrapped to (-pi, pi], build per-channel Morlet bursts at those phases,
    add optional pink noise per channel. Returns a TimeSeries with dims
    ('event', 'channel', 'time').
    """
    # Code-issues #54: optional rng seed / generator threads into both
    # stochastic branches (sample_phases for the wavelet phase draws,
    # generate_pink_noise for the noise). Same Generator passed through
    # means deterministic replay of the entire EEG when rng is fixed,
    # without touching numpy's global state.
    rng_obj = np.random.default_rng(rng) if rng is not None else None
    duration_s = duration_ms / 1000
    if wavelet_amplitude > 0:
        assert n_channels == len(phase_mean)
        assert phase_covariance.shape == (n_channels, n_channels)
    N_samples_per_epoch = duration_to_samples(duration_s, sample_rate_Hz)
    end_time_ms = start_time_ms + duration_ms
    times = np.linspace(start_time_ms, end_time_ms, N_samples_per_epoch)
    coords = {'event': ['event_' + str(i) for i in range(n_events)],
              'channel': ['CH' + str(i) for i in range(n_channels)],
              'time': times
    }
    data = np.full(shape=(n_events, n_channels, len(times)), fill_value=np.nan)

    # Single top-level guard makes the wavelets-bound invariant structural so
    # pyright can prove `wavelets[channel]` is reachable only when `wavelets`
    # is defined. Per-channel noise generation is preserved in both branches
    # (noise resampled per channel, not per event).
    for event in range(n_events):
        if wavelet_amplitude > 0:
            phases = sample_phases(mean=phase_mean, cov=phase_covariance,
                                   samples=1, rng=rng_obj)[0]
            wavelets = generate_event_wavelets(phases=phases,
                                               duration_s=duration_s,
                                               frequency=connectivity_frequency_Hz,
                                               sampling_rate=sample_rate_Hz,
                                               amplitude=wavelet_amplitude,
                                               morlet_reps=morlet_reps,
                                              )
            for channel in range(n_channels):
                data[event, channel] = wavelets[channel]
                if pinknoise_amplitude > 0:
                    noise = generate_pink_noise(N_samples=N_samples_per_epoch,
                                                pinknoise_amplitude=pinknoise_amplitude,
                                                pinknoise_exponent=pinknoise_exponent,
                                                rng=rng_obj)
                    data[event, channel] += noise
        elif pinknoise_amplitude > 0:
            for channel in range(n_channels):
                noise = generate_pink_noise(N_samples=N_samples_per_epoch,
                                            pinknoise_amplitude=pinknoise_amplitude,
                                            pinknoise_exponent=pinknoise_exponent,
                                            rng=rng_obj)
                data[event, channel] = noise
    if np.isnan(data).any():
        raise ValueError('NaNs detected in EEG!')
    eeg = TimeSeries.create(  # pyright: ignore[reportUnknownMemberType]
        data,
        samplerate=sample_rate_Hz,
        coords=coords,
        dims=list(coords.keys()),
    )
    
    return eeg


def compute_ppc_matrix_nonoverlapping(
    phase: xr.DataArray,
    sample_rate: float,
    epoch_width_ms: float = 200,
    time_axis: int = -1,
) -> xr.DataArray:
    """Non-overlapping PPC matrix across electrode pairs, binned in epoch_width_ms windows.

    Returns a (channel1, channel2, frequency, epoch) DataArray, symmetrized.
    """
    # Runtime defensive check despite the annotation: callers from untyped
    # scripts can still pass non-DataArrays.
    if not isinstance(phase, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError()
    electrode_count = len(phase.channel)
    freq_count = len(phase.frequency)
    duration_ms = phase['time'].max() - phase['time'].min()
    epoch_count = int(np.round(duration_ms / epoch_width_ms))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    ppcs = np.full((electrode_count, electrode_count, freq_count, epoch_count), np.nan)
    for iElec in np.arange(electrode_count):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        for jElec in np.arange(electrode_count):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            diff = cast(NDArrayAny,
                (phase.isel(channel=jElec) - phase.isel(channel=iElec)).data)
            from helper import timebin_phase_timeseries  # lazy: avoid circular import
            diff = timebin_phase_timeseries(diff, sample_rate, bin_width_ms=epoch_width_ms)
            ppcs[iElec, jElec] = ppc(diff)

    ppcs = symmetrize(ppcs)
    ppcs_da = xr.DataArray(data=ppcs,
                        dims=('channel1', 'channel2', 'frequency', 'epoch'),
                        coords={'channel1': phase.channel.values,
                                'channel2': phase.channel.values,
                                'frequency': phase.frequency,
                                'epoch': np.arange(epoch_count) * epoch_width_ms + phase['time'].min().item(),  # pyright: ignore[reportUnknownMemberType]
                        })
    return ppcs_da


def ppc_matrix_to_wrapped_normal_covariance(ppc_matrix: NDArrayAny) -> NDArrayAny:
    """Build a PD covariance matrix whose wrapped-normal first moment squared
    matches `ppc_matrix` element-wise.

    Construction: sigma_ij^2 = -log(PPC_ij); cov = diag(max_var) - delta_var/2
    on off-diagonals so cov diag - cov off-diag = sigma_ij^2 / 2. Validated
    via is_symmetric + is_positive_definite (raises if either fails).
    """
    is_square(ppc_matrix, require=True)
    ppc_variances = ppc_to_wrapped_normal_sigma(ppc_matrix) ** 2
    ppc_variances = strict_triu(ppc_variances)
    max_var = float(ppc_variances.max())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportCallIssue]
    covar = np.ones(ppc_matrix.shape) * max_var
    for i in range(covar.shape[0]):
        for j in range(i + 1, covar.shape[0]):
            ppc_delta_var = ppc_variances[i, j] / 2
            covar[i, j] -= ppc_delta_var
            covar[j, i] -= ppc_delta_var
    is_symmetric(covar, require=True)
    is_positive_definite(covar, require=True)
    return covar


def coherence_matrix_to_wrapped_normal_covariance(msc_matrix: NDArrayAny) -> NDArrayAny:
    # In this codebase ppc = exp(-sigma^2) (the squared mean-resultant length;
    # see cstat.wrapped_normal_sigma_to_ppc), and under the constant-amplitude
    # Morlet generative model used here coh = plv = exp(-sigma^2/2), so
    # coh^2 = plv^2 = ppc = MSC. The PPC target IS the MSC target: no
    # transform is needed, this is just a relabelled alias.
    return ppc_matrix_to_wrapped_normal_covariance(msc_matrix)


def get_block_diagonal_ppc_matrix(
    n_channels: int = 16,
    n_regions: int = 4,
    n_region_groups: int = 2,
    regions: list[Any] | None = None,
    region_groups: list[Any] | None = None,
    global_ppc: float = 0.05,
    within_group_ppc: float = 0.2,
    within_region_ppc: float = 0.4,
    verbose: bool = False,
) -> NDArrayAny:
    """Build a target PPC matrix with three-level block structure.

    When `regions` is None (synthetic uniform layout): channels are split
    evenly into n_regions and n_region_groups; off-diagonal entries within a
    region get within_region_ppc, within a region-group get within_group_ppc,
    and the rest get global_ppc.

    When `regions` (and `region_groups`) are passed: entries are set per-pair
    using the provided per-channel region/group labels.

    Diagonal is set to 1. PPCs must all be > 0 because PPC=0 corresponds to
    infinite linear variance in the wrapped-normal model.
    """
    # global_ppc models global connectivity between all channels or lack thereof
    # Note: all PPC values must be positive given a PPC of zero corresponds to a wrapped normal with 
    # infinite linear variance (which could be implemented with a separate outer uniform random 
    # phase sample for disjoint connectivity subgraphs), so use a minimum PPC well below 
    # empirically observed values to model zero connectivity
    assert global_ppc > 0
    assert within_group_ppc > 0
    assert within_region_ppc > 0
    
    
    if regions is None:
        ppc_matrix = np.ones((n_channels, n_channels)) * global_ppc
        assert n_channels % n_regions == 0
        n_channels_per_region = n_channels // n_regions

        assert n_channels % n_region_groups == 0
        n_channels_per_group = n_channels // n_region_groups

        # model functional connectivity as connected groups of internally connected regions
        for region in range(0, n_channels, n_channels_per_group):
            ppc_matrix[region:region + n_channels_per_group, 
                       region:region + n_channels_per_group] = within_group_ppc

        for channel in range(0, n_channels, n_channels_per_region):
            ppc_matrix[channel:channel + n_channels_per_region, 
                       channel:channel + n_channels_per_region] = within_region_ppc
    else:
        assert isinstance(regions, list) and all(
            [isinstance(region, str) or np.isnan(region) for region in regions])
        assert region_groups is not None, "region_groups required when regions is set"
        assert within_group_ppc <= within_region_ppc
        n_channels = len(regions)
        ppc_matrix = np.ones((n_channels, n_channels)) * global_ppc
        # n_regions kept for parity with the regions=None branch even if
        # downstream code in this block doesn't currently use it.
        _unique_regions = np.unique(regions)
        n_regions = len(_unique_regions)

        for idx1, (region1, group1) in enumerate(zip(regions, region_groups)):
            for idx2, (region2, group2) in enumerate(zip(regions, region_groups)):
                if group1 == group2:
                    ppc_matrix[idx1, idx2] = within_group_ppc
                if region1 == region2:
                    ppc_matrix[idx1, idx2] = within_region_ppc

    np.fill_diagonal(ppc_matrix, 1)

    if verbose:
        ppcs = strict_triu(ppc_matrix).ravel()
        ppcs = ppcs[ppcs > 0]
        print(f'Mean PPC: {ppcs.mean()}')  # pyright: ignore[reportUnknownMemberType]
        plt.figure()  # pyright: ignore[reportUnknownMemberType]
        plt.imshow(ppc_matrix)  # pyright: ignore[reportUnknownMemberType]
        plt.colorbar()  # pyright: ignore[reportUnknownMemberType]
        plt.title('PPC Matrix')  # pyright: ignore[reportUnknownMemberType]
        if regions:
            ppc_matrix_sorted = sort_array_across_order(ppc_matrix, regions, axis=[0, 1])
            plt.figure()  # pyright: ignore[reportUnknownMemberType]
            plt.imshow(ppc_matrix_sorted)  # pyright: ignore[reportUnknownMemberType]
            plt.colorbar()  # pyright: ignore[reportUnknownMemberType]
            plt.title('PPC Matrix Sorted by Region')  # pyright: ignore[reportUnknownMemberType]

    return ppc_matrix


# --- Pipeline hook: swap a session's real EEG for simulated EEG ------------
# Called by helper.get_eeg when a simulation_tag is set; real montage, regions
# and events are kept.
def _simulated_electrode_regionalizations(
    eeg: TimeSeries, dfrow: pd.Series,
) -> tuple[list[Any], list[str]]:
    """Per-channel region labels + L/R hemisphere groups for the simulated
    EEG, used to build the block-diagonal coupling target (phase covariance
    OR amplitude-envelope correlation). Factored out so the phase and AEC
    DGP branches of replace_w_simulated_EEG share one region mapping."""
    pairs = get_pairs(dfrow)
    assert len(pairs) == len(eeg.channel)
    regionalizations = regionalize_electrodes_by_type(pairs)
    region_series = pd.Series(regionalizations)
    has_hemisphere_mask = region_series.str.startswith('L ') | region_series.str.startswith('R ') | region_series.isna()
    if not has_hemisphere_mask.all():
        raise ValueError
    # use hemispheres for simple regional grouping (rare NaN region -> 'Right')
    region_groups = ['Left' if left else 'Right' for left in pd.Series(regionalizations).str.startswith('L ')]
    return list(regionalizations), list(region_groups)


def _target_lobe_mask(regionalizations: Sequence[Any], lobes: Sequence[str]) -> NDArrayAny:
    """True for channels whose region ('L frontal pole' ...) is in one of `lobes` (Burke lobes)."""
    lobe_of = pd.read_csv(Path(__file__).resolve().parent / 'region_to_burke_lobe.csv').set_index('region')['burke_lobe']
    return np.array([isinstance(r, str) and lobe_of.get(r.split(' ', 1)[1]) in lobes for r in regionalizations])


def _simulate_hg_power_eeg(eeg: TimeSeries, dfrow: pd.Series, p: dict[str, Any]) -> TimeSeries:
    """Pink noise + `band`-limited noise on every channel; on target-lobe channels the
    band-limited part is scaled by `hg_gain` from stimulus onset (t >= 0) on."""
    import mne
    from simulate_eeg import generate_pink_noise
    n_ev, n_ch, n_t = eeg.shape
    pink = np.array([generate_pink_noise(n_t, p['pinknoise_amplitude'], p['pinknoise_exponent'])
                     for _ in range(n_ev * n_ch)]).reshape(n_ev, n_ch, n_t)
    hg = p['hg_amplitude'] * mne.filter.filter_data(np.random.standard_normal((n_ev, n_ch, n_t)),
                                                    float(eeg.samplerate), *p['band'], verbose=False)
    target = _target_lobe_mask(regionalize_electrodes_by_type(get_pairs(dfrow)), p['target_lobes'])
    hg[:, target] *= np.where(np.asarray(eeg.time) >= 0, p['hg_gain'], 1.0)
    return eeg.copy(data=pink + hg)


def _simulate_aec_envelope_eeg(
    original_eeg: TimeSeries,
    eeg: TimeSeries,
    dfrow: pd.Series,
    parameters: dict[str, Any],
    condition_mask: NDArrayAny,
    time_unit: str,
    random_state: int | None,
    verbose: bool = False,
) -> TimeSeries:
    """AEC amplitude-envelope DGP for the pipeline path. Builds a
    block-diagonal cross-channel ENVELOPE correlation target per contrast arm
    (cond0 = ~mask, cond1 = mask) from within_region/within_group/global _aec
    params, then generates envelope-coupled EEG via fc_aec_dgp.sample_eeg_aec
    (Cholesky-planted envelopes modulating a carrier). Mirrors the phase path's
    split / assemble / event-reorder steps so word_on/voc pre-post and en/rm
    matched contrasts flow through compute_session_fc identically."""
    regionalizations, region_groups = _simulated_electrode_regionalizations(eeg, dfrow)
    from simulate_eeg import get_block_diagonal_ppc_matrix

    def _aec_corr(suffix: str) -> NDArrayAny:
        # values are envelope correlations, planted directly via Cholesky
        # (no wrapped-normal transform — that is phase-DGP only).
        return get_block_diagonal_ppc_matrix(
            n_channels=None, n_regions=None, n_region_groups=None,
            regions=list(regionalizations), region_groups=list(region_groups),
            global_ppc=parameters[f"global_aec{suffix}"],
            within_group_ppc=parameters[f"within_group_aec{suffix}"],
            within_region_ppc=parameters[f"within_region_aec{suffix}"],
            verbose=verbose,
        )
    envcorr0 = _aec_corr("0")
    envcorr1 = _aec_corr("1")

    start_time_ms = original_eeg.time.min()
    duration_ms = original_eeg.time.max() - start_time_ms
    if time_unit == 'second':
        start_time_ms *= 1000
        duration_ms *= 1000

    eeg = eeg.assign_coords(_index=("event", np.arange(len(eeg['event'])))).set_index(event='_index', append=True)
    eeg0 = eeg[~condition_mask]
    eeg1 = eeg[condition_mask]

    from fc_aec_dgp import sample_eeg_aec  # pyright: ignore[reportMissingImports]

    carrier_freq_Hz = parameters['carrier_freq_Hz']
    envelope_cutoff_Hz = parameters['envelope_cutoff_Hz']
    phase_mode = parameters.get('phase_mode', 'independent')
    noise_amplitude = parameters.get('noise_amplitude', 0.1)
    sr = float(original_eeg.samplerate)

    def _gen(n_ev: int, corr: NDArrayAny, seed_offset: int) -> NDArrayAny:
        rng = None if random_state is None else (random_state + seed_offset) % (2**32 - 1)
        ts = sample_eeg_aec(
            n_events=int(n_ev), n_channels=len(eeg.channel), target_corr=corr,
            sample_rate_Hz=sr, duration_ms=float(duration_ms),
            carrier_freq_Hz=carrier_freq_Hz, phase_mode=phase_mode,
            envelope_cutoff_Hz=envelope_cutoff_Hz, noise_amplitude=noise_amplitude,
            rng=rng)
        return np.asarray(ts.values)

    data0 = _gen(len(eeg0.event), envcorr0, 0)
    data1 = _gen(len(eeg1.event), envcorr1, 1)
    # sample_eeg_aec emits 0-based time; rebuild absolute-time coords matching
    # the real epoch window (length = generator n_samples).
    n_t = data0.shape[-1]
    aec_times = np.linspace(float(start_time_ms), float(start_time_ms) + float(duration_ms), n_t)
    simulated_eeg0 = TimeSeries.create(
        data=data0,
        coords={'event': eeg0.coords['event'], 'channel': eeg0.coords['channel'], 'time': aec_times},
        dims=('event', 'channel', 'time'), samplerate=sr)
    simulated_eeg1 = TimeSeries.create(
        data=data1,
        coords={'event': eeg1.coords['event'], 'channel': eeg1.coords['channel'], 'time': aec_times},
        dims=('event', 'channel', 'time'), samplerate=sr)

    del eeg0, eeg1
    from matrix_operations import sort_multi_index_coord
    if not np.all(simulated_eeg0.time == simulated_eeg1.time):
        raise ValueError('Time values in simulated EEG do not match across conditions')
    simulated_eeg = xr.concat([simulated_eeg0, simulated_eeg1], 'event')
    simulated_eeg = sort_multi_index_coord(simulated_eeg, 'event', '_index')
    simulated_eeg = simulated_eeg.reset_index('_index', drop=True)
    simulated_eeg.attrs['samplerate'] = original_eeg.samplerate
    assert original_eeg.channel.equals(simulated_eeg.channel)
    assert original_eeg.samplerate.equals(simulated_eeg.samplerate)
    if 'samplerate' in original_eeg.attrs:
        simulated_eeg.attrs['samplerate'] = simulated_eeg.samplerate
    if time_unit == 'second':
        simulated_eeg = simulated_eeg.assign_coords({'time': simulated_eeg['time'] / 1000})
    return simulated_eeg


def replace_w_simulated_EEG(
    original_eeg: TimeSeries,
    dfrow: pd.Series,
    condition_mask: NDArrayAny,
    simulation_tag: str | None = None,
    time_unit: str = 'millisecond',
    random_state: int | None = None,
    random_state_type: str = 'offset_from_eeg_hash',
    verbose: bool = False,
) -> TimeSeries:
    assert isinstance(original_eeg, TimeSeries)
    assert simulation_tag in AVAILABLE_SIMULATIONS
    
    if random_state_type == 'offset_from_eeg_hash':
        # fix random state to hash of original EEG
        # ensures unique, reproducible random states for each unique input EEG recording
        eeg_hash = hash(str(original_eeg.data))
        random_state = eeg_hash if random_state is None else eeg_hash + random_state
        random_state %= 2**32 - 1
    elif random_state_type == 'standard':
        pass
    else:
        raise ValueError
    if random_state is not None:
        np.random.seed(random_state)
    
    if simulation_tag in ['standard', '', None]:
        return original_eeg
    eeg = original_eeg.copy()
    
    parameters = simulation_parameters[simulation_tag]

    if parameters.get('data_generating_process') == 'aec_envelope':
        return _simulate_aec_envelope_eeg(
            original_eeg, eeg, dfrow, parameters, condition_mask,
            time_unit, random_state, verbose)

    if parameters.get('data_generating_process') == 'hg_power':
        return _simulate_hg_power_eeg(original_eeg, dfrow, parameters)

    wavelet_amplitude = parameters['wavelet_amplitude']
    get_phase_covariance = parameters['phase_covariance_function']
    if get_phase_covariance == 'within_region_group':
        pairs = get_pairs(dfrow)
        assert len(pairs) == len(eeg.channel)
        regionalizations = regionalize_electrodes_by_type(pairs)
        region_series = pd.Series(regionalizations)
        has_hemisphere_mask = region_series.str.startswith('L ') | region_series.str.startswith('R ') | region_series.isna()
        if not has_hemisphere_mask.all():
            # print(region_series[~has_hemisphere_mask])
            # display(region_series)
            raise ValueError
        # use hemispheres for simple regional grouping (put rare NaN region channels in 'Right' group)
        region_groups = ['Left' if left else 'Right' for left in pd.Series(regionalizations).str.startswith('L ')]
        if parameters.get('target_lobes'):   # couple only the target-lobe channels, to each other
            target = _target_lobe_mask(regionalizations, parameters['target_lobes'])
            regionalizations = region_groups = ['target' if t else f'c{i}' for i, t in enumerate(target)]

        from simulate_eeg import get_block_diagonal_ppc_matrix, ppc_matrix_to_wrapped_normal_covariance

        # Read FC targets on the metric's natural scale:
        #   *_ppc*  (PPC sims)  → values are PPC (R²); used directly.
        #   *_coh*  (Coh sims)  → values are coh-modulus (R); squared to PPC.
        # The wrapped-normal cov machinery expects PPC inputs, so both
        # conventions are normalized to PPC here before cov construction.
        def _ppc_target(scope, suffix):
            coh_key = f"{scope}_coh{suffix}"
            if coh_key in parameters:
                return parameters[coh_key] ** 2
            return parameters[f"{scope}_ppc{suffix}"]

        ppc_matrix0 = get_block_diagonal_ppc_matrix(n_channels=None,
                                                    n_regions=None,
                                                    n_region_groups=None,
                                                    regions=list(regionalizations),
                                                    region_groups=list(region_groups),
                                                    global_ppc=_ppc_target("global", "0"),
                                                    within_group_ppc=_ppc_target("within_group", "0"),
                                                    within_region_ppc=_ppc_target("within_region", "0"),
                                                    verbose=verbose,
                                                   )
        cov0 = ppc_matrix_to_wrapped_normal_covariance(ppc_matrix0)

        ppc_matrix1 = get_block_diagonal_ppc_matrix(n_channels=None,
                                                    n_regions=None,
                                                    n_region_groups=None,
                                                    regions=list(regionalizations),
                                                    region_groups=list(region_groups),
                                                    global_ppc=_ppc_target("global", "1"),
                                                    within_group_ppc=_ppc_target("within_group", "1"),
                                                    within_region_ppc=_ppc_target("within_region", "1"),
                                                    verbose=verbose,
                                                   )
        cov1 = ppc_matrix_to_wrapped_normal_covariance(ppc_matrix1)
        
    elif get_phase_covariance is None:
        cov0 = None
        cov1 = None
    else:
        raise NotImplementedError(f'Phase covariance method {get_phase_covariance} is not implemented!')
    oscillation_frequency = parameters['oscillation_frequency']
    morlet_reps = parameters['morlet_reps']
    
    pinknoise_amplitude = parameters['pinknoise_amplitude']
    pinknoise_exponent = parameters['pinknoise_exponent']
    
    start_time_ms = original_eeg.time.min()
    duration_ms = original_eeg.time.max() - start_time_ms
    if time_unit == 'second':
        start_time_ms *= 1000
        duration_ms *= 1000
    
    eeg = eeg.assign_coords(_index=("event", np.arange(len(eeg['event'])))).set_index(event='_index', append=True)
    eeg0 = eeg[~condition_mask]
    eeg1 = eeg[condition_mask]
    
    simulated_eeg0 = sample_eeg(n_events=len(eeg0.event),
                                n_channels=len(eeg0.channel),
                                sample_rate_Hz=eeg0.samplerate,
                                start_time_ms=start_time_ms,
                                duration_ms=duration_ms,
                                connectivity_frequency_Hz=oscillation_frequency,
                                morlet_reps=morlet_reps,
                                wavelet_amplitude=wavelet_amplitude,
                                phase_mean=np.zeros(len(cov0)) if cov0 is not None else None,
                                phase_covariance=cov0,
                                pinknoise_amplitude=pinknoise_amplitude,
                                pinknoise_exponent=pinknoise_exponent,
    )
    
    simulated_eeg1 = sample_eeg(n_events=len(eeg1.event),
                                n_channels=len(eeg1.channel),
                                sample_rate_Hz=eeg1.samplerate,
                                start_time_ms=start_time_ms,
                                duration_ms=duration_ms,
                                connectivity_frequency_Hz=oscillation_frequency,
                                morlet_reps=morlet_reps,
                                wavelet_amplitude=wavelet_amplitude,
                                phase_mean=np.zeros(len(cov1)) if cov1 is not None else None,
                                phase_covariance=cov1,
                                pinknoise_amplitude=pinknoise_amplitude,
                                pinknoise_exponent=pinknoise_exponent,
    )
    simulated_eeg0 = TimeSeries.create(data=simulated_eeg0.values,
                                       coords={'event': eeg0.coords['event'],
                                               'channel': eeg0.coords['channel'],
                                               'time': simulated_eeg0.coords['time']},
                                       dims=simulated_eeg0.dims,
                                       samplerate=simulated_eeg0.samplerate.item())
    simulated_eeg1 = TimeSeries.create(data=simulated_eeg1.values,
                                       coords={'event': eeg1.coords['event'],
                                               'channel': eeg1.coords['channel'],
                                               'time': simulated_eeg1.coords['time']},
                                       dims=simulated_eeg1.dims,
                                       samplerate=simulated_eeg1.samplerate.item())
    
    del eeg0, eeg1
    from matrix_operations import sort_multi_index_coord
    if not np.all(simulated_eeg0.time == simulated_eeg1.time):
        raise ValueError('Time values in simulated EEG do not match across conditions')
    
    simulated_eeg = xr.concat([simulated_eeg0, simulated_eeg1], 'event')
    # sort back into original event order
    simulated_eeg = sort_multi_index_coord(simulated_eeg, 'event', '_index')
    simulated_eeg = simulated_eeg.reset_index('_index', drop=True)
    simulated_eeg.attrs['samplerate'] = original_eeg.samplerate
    
    # n_evs = 0
    # for i_ev, (sim_ev, orig_ev) in enumerate(zip(simulated_eeg.event, original_eeg.event)):
    #     if not sim_ev.equals(orig_ev):
    #         print(i_ev)
    #         print(sim_ev)
    #         print()
    #         print(orig_ev)
    #         print()
    #         print('sim_ev == sim_ev', sim_ev.equals(sim_ev))
    #         print('orig_ev == orig_ev', orig_ev.equals(orig_ev))
    #         print()
    #         print()
    #         n_evs += 1
    #         if n_evs == 5:
    #             break
    
    # print(original_eeg.event)
    # print(simulated_eeg.event)
    # print(original_eeg.item_name)
    # print(simulated_eeg.item_name)
    
    # matches for most sessions, but some get implicitly type cast by xarray in workshop_311 environment
    # assert original_eeg.event.equals(simulated_eeg.event)
    assert original_eeg.channel.equals(simulated_eeg.channel)
    assert original_eeg.samplerate.equals(simulated_eeg.samplerate)
    if 'samplerate' in original_eeg.attrs:
        simulated_eeg.attrs['samplerate'] = simulated_eeg.samplerate
    
    if time_unit == 'second':
        simulated_eeg = simulated_eeg.assign_coords({'time': simulated_eeg['time'] / 1000})

    return simulated_eeg
