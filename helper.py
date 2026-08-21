"""helper.py — core IO / signal-processing / FC-orchestration toolkit.

The widest-reach module in the repo. Reads CML / PTSA events + EEG, applies
notch-filter and mirror-buffer prep, builds bipolar pairs, runs Morlet phase
or power filtering, time-bins, regionalizes electrode-level results, and
orchestrates per-session FC pipelines.

This file is dominated by pandas method chains + ptsa + cmlreaders +
mne_connectivity + matplotlib pyplot calls — third-party stubs are weak
across all of them. Narrow `reportUnknownMemberType` and friends to warning
at the file level so the strict-mode signal we DO want (missing annotations,
unbound vars, real type drift across our own functions) still surfaces.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning, reportConstantRedefinition=warning
from __future__ import annotations

import functools
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import os
from pathlib import Path

import cmlreaders as cml  # noqa: F401  # pyright: ignore[reportMissingTypeStubs]
from os.path import join, exists as ex

import pandas as pd
import xarray as xr  # pyright: ignore[reportMissingTypeStubs]

import ptsa_patches as _ptsa_patches  # noqa: F401 — applies monkey-patches on import; must precede ptsa.* imports below
_ = _ptsa_patches
from ptsa.data.readers import EEGReader, TalReader  # pyright: ignore[reportMissingTypeStubs]
from ptsa.data.filters import MonopolarToBipolarMapper, MorletWaveletFilter  # pyright: ignore[reportMissingTypeStubs]
from ptsa.data.timeseries import TimeSeries  # pyright: ignore[reportMissingTypeStubs]


from cstat import *  # noqa: F401,F403
from misc import *  # noqa: F401,F403
from misc import ftag  # explicit for type-checker visibility
from matrix_operations import *  # noqa: F401,F403
from simulate_eeg import AVAILABLE_SIMULATIONS, simulation_parameters, sample_eeg

NDArrayAny = npt.NDArray[Any]

# Module-level `root_dir` is set by callers (prepare_sessions, build_subj_mat,
# load_events) immediately after import: `helper.root_dir = str(SCRATCH_DIR)`
# where SCRATCH_DIR comes from project_paths. Do NOT initialize it here —
# `from helper import *` in fc_comparison_functions would then shadow that
# module's local `root_dir = str(SCRATCH_DIR)` with the default value,
# silently routing FC compute to an empty path. Declare it via TYPE_CHECKING
# only so static-checkers see the attribute without creating a runtime symbol.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    root_dir: str  # type: ignore[no-redef]

from project_paths import BANDS as bands  # noqa: E402
from project_paths import RESAMPLE_HZ, NOTCH_HARMONICS_UP_TO_HZ  # noqa: E402

# CFG_FLOW_VERIFY: helper.bands must equal project_paths.BANDS (config.yaml: bands)
# Per-key pins: pre-migration helper.bands had {"alpha": (8,12), "theta": (4,9),
# "low": (5,11), "gamma": (80,160)}. alpha was reconciled (8,12) -> (8,13) at
# unification (canonical is fc_comparison_functions.bands which had (8,13);
# no runtime consumer of helper.bands["alpha"] in production path).
from project_paths import BANDS as _CFG_BANDS  # noqa: E402
assert bands is _CFG_BANDS, "helper.bands drift from project_paths.BANDS"
assert bands["alpha"] == (8.0, 13.0), f"alpha drift: {bands['alpha']}"  # CFG_FLOW_VERIFY
assert bands["theta"] == (4.0, 9.0), f"theta drift: {bands['theta']}"   # CFG_FLOW_VERIFY
assert bands["low"]   == (3.0, 8.0), f"low drift: {bands['low']}"      # CFG_FLOW_VERIFY
assert bands["gamma"] == (70.0, 110.0), f"gamma drift: {bands['gamma']}"  # CFG_FLOW_VERIFY

beh_to_event_windows = {'en': [250, 1250],
                     'en_all': [250, 1250],
                     'rm': [-1000, 0],
                     'rm_all': [-1000, 0],
                     'ri': [-1000, 0],
                     'word_on': [-650, 600],
                     'voc': [-1000, 1000]}

beh_to_epochs = {'en': np.arange(250, 1250, 200),
              'en_all': np.arange(250, 1250, 200),
              'rm': np.arange(-1000, 0, 200),
              'rm_all': np.arange(-1000, 0, 200),
              'ri': np.arange(-1000, 0, 200),
              'word_on': np.arange(-650, 600, 200),
              'voc': np.arange(-1000, 1000, 200)}

behavioral_names = {'en': 'Encoding',
                    'rm': 'Retrieval',
                    'rm_all': 'All Retrieval',
                    'ri': 'Recall Accuracy',
                    'word_on': 'Word on Screen',
                    'voc': 'Vocalization'}

# root_dir set in main analysis notebook
def print_root_dir() -> None:
    """Diagnostic: print the current module-level root_dir."""
    print(root_dir)


def load_events(dfrow: pd.Series, beh: str) -> pd.DataFrame:
    
    '''
    Loads behavioral events for a particular experimental session and behavioral contrast.
    Requires that the events data and metadata file have already been saved out (see the "Get behavioral events" section in WholeBrainConnectivityPPCRevision.ipynb).
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        beh : str
            Behavioral contrast label ('en', 'en_all', 'rm', or 'ri').
        
    Returns:
        events : pandas.DataFrame
            Behavioral events for a particular experimental session and behavioral contrast.
    '''
    
    evs_path = join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json')
    evs_metadata_path = join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json')
    if not ex(evs_path): return None
    events = pd.read_json(evs_path)
    events_metadata = pd.read_json(evs_metadata_path, typ='series')
    events.attrs = events_metadata
    events.attrs['mask'] = np.asarray(events.attrs['mask'])

    return events


def _load_cmlreaders_eeg_with_retry(
    reader: Any,
    events: pd.DataFrame,
    start: int,
    end: int,
    pairs: pd.DataFrame,
    sess_tag: str,
    n_retries: int = 12,
    backoff_s: float = 5.0,
) -> Any:
    """Wrap `reader.load_eeg(...)` with retry-on-FileNotFoundError.

    Penn cluster compute nodes occasionally return FileNotFoundError from
    cmlreaders' params-resolution step on the FIRST access to a session
    directory under /data/eeg/<sub>/, even though the file exists on the
    underlying disk (autofs/NFS mount race). A short fixed-interval poll
    with a directory-stat flush between attempts clears the race in
    every case observed so far.

    n_retries=12 × backoff_s=5 → up to ~60 s of polling per session.
    """
    import time
    last_err: Exception | None = None
    for attempt in range(n_retries):
        try:
            return reader.load_eeg(events, start, end, scheme=pairs)
        except FileNotFoundError as e:
            last_err = e
            # Stat the parent dirs to force autofs to mount/refresh, then sleep.
            eegfile = str(events['eegfile'].iloc[0]) if 'eegfile' in events.columns and len(events) else ''
            for d in (eegfile, os.path.dirname(eegfile),
                      os.path.dirname(os.path.dirname(eegfile))):
                if d:
                    try: os.stat(d)
                    except Exception: pass
            print(f"[{sess_tag}] cmlreaders FileNotFoundError "
                  f"(attempt {attempt+1}/{n_retries}); retrying in {backoff_s}s")
            time.sleep(backoff_s)
    assert last_err is not None
    raise last_err


def _event_cell_is_list(v: object) -> bool:
    return isinstance(v, list)


def _coerce_event_cell(v: object) -> object:
    """Empty list -> "" (no item); any other list -> str repr; else unchanged."""
    if isinstance(v, list):
        return "" if not v else str(v)
    return v


def coerce_unhashable_event_fields(events: pd.DataFrame) -> None:
    """Replace list-valued event cells with a hashable scalar, in place.

    cmlreaders' ``to_ptsa()`` factorizes the event columns into the event-dim
    MultiIndex, which raises ``TypeError: unhashable type: 'list'`` on any
    list-valued cell. pyFR intrusion vocalizations carry ``item == []`` (an
    empty list, i.e. no recalled dictionary word), so the ``voc`` contrast —
    which keeps all vocalizations including intrusions — trips this while
    ``rm`` (correct matched recalls only) does not. Coerce: empty list -> ""
    (no item), any other list -> its string repr. Non-list cells untouched.
    Operates only on object-dtype columns that actually contain a list.
    """
    for col in events.columns:
        if events[col].dtype != object:
            continue
        if events[col].map(_event_cell_is_list).any():
            events[col] = events[col].map(_coerce_event_cell)


def get_eeg(
    dfrow: pd.Series,
    events: pd.DataFrame,
    start: int,
    end: int,
    simulation_tag: str | None = None,
) -> TimeSeries:
    
    '''
    Returns EEG signal for a particular session and set of behavioral events.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        start : float
            Time (ms) at which returned EEG clip should begin, relative to a particular event.
        end : float
            Time (ms) at which returned EEG clip should end, relative to a particular event.
        simulation_tag : str
            Label of parameter set used to generate simulated EEG signal.
    
    Returns:
        eeg : ptsa.data.TimeSeries
            EEG clip.
        numpy.array
            List of boolean variables indicating whether the event was a successful memory event (True) or an unsuccessful memory event (False).
    '''
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    sess_list_df = pd.read_json(join(root_dir, 'sess_list_df.json'))
    sess_list_df.set_index(['sub', 'exp', 'sess', 'loc', 'mon'], inplace=True)
    eeg_data_source = sess_list_df.loc[(sub, exp, sess, loc, mon), 'eeg_data_source']
    
    events['event_idx'] = np.arange(len(events))
    events.sort_values(by=['mstime', 'eegoffset'], inplace=True)
    events.attrs['mask'] = events.attrs['mask'][events['event_idx']]
    events.drop('event_idx', axis=1, inplace=True)

    # pyFR intrusion vocalizations carry list-valued `item` ([]), which breaks
    # cmlreaders' to_ptsa() event-index factorization. Coerce to a hashable
    # scalar before the EEG load so both loader branches are safe.
    coerce_unhashable_event_fields(events)

    if eeg_data_source == 'cmlreaders':
        reader = cml.CMLReader(subject=sub,
                               experiment=exp,
                               session=sess,
                               localization=loc,
                               montage=mon)
        pairs = get_pairs(dfrow)
        eeg = _load_cmlreaders_eeg_with_retry(reader, events, start, end, pairs,
                                              ftag(dfrow))
        # Workaround cmlreaders' float-rounding bug in _make_time_array:
        # for some session sample rates (e.g. sr=499.7071 → rate=2.001172 ms)
        # `np.arange(start, n*rate+start, rate)` accumulates enough FP error
        # to include one extra element, producing a time coord of length n+1
        # for data of length n. Without this, to_ptsa() below raises a
        # `conflicting sizes for dimension 'time'` ValueError and the session
        # is dropped. Rebuild the time array using integer-arange.
        n_t = eeg.data.shape[-1]
        rate_ms = 1000.0 / eeg.samplerate
        eeg.time = eeg.time[0] + np.arange(n_t) * rate_ms
        eeg = eeg.to_ptsa()

    elif eeg_data_source == 'ptsa':
        # get_ptsa_eeg normalizes its time coord to ms by default so the
        # two loaders return matching units. Without that normalization,
        # downstream time-axis masks like `(t <= -50)` silently produced
        # empty selections on PTSA sessions and dropped ~3% of the dataset.
        eeg = get_ptsa_eeg(dfrow, events, start, end)
    else:
        raise ValueError(f"Unknown eeg_data_source {eeg_data_source} for session {ftag(dfrow)}")

    sr = float(eeg.samplerate)
    if 'sr' in events.attrs:
        assert events.attrs['sr'] == sr, 'sampling rate is wrong'

    dim_map = {}
    if 'events' in eeg.dims: dim_map['events'] = 'event'
    if 'channels' in eeg.dims: dim_map['channels'] = 'channel'
    eeg = eeg.rename(dim_map)
    eeg = eeg.transpose('event', 'channel', 'time')

    if simulation_tag not in ['standard', '', None]:
        # replace experimentally recorded EEG with simulated EEG to validate analysis pipeline
        eeg = replace_w_simulated_EEG(eeg,
                                      dfrow,
                                      eeg_data_source=eeg_data_source,
                                      time_unit='millisecond',
                                      condition_mask=events.attrs['mask'],
                                      simulation_tag=simulation_tag)
    
    return eeg, events.attrs['mask']

def get_ptsa_eeg(
    dfrow: pd.Series,
    events: pd.DataFrame,
    start: int,
    end: int,
    normalize_time_to_ms: bool = True,
) -> TimeSeries:

    '''
    Returns EEG signal for a particular session and set of behavioral events, using the ptsa readers. Used to load EEG for pyFR experimental sessions whose data could not be loaded with cmlreaders.

    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        start : float
            Time (ms) at which returned EEG clip should begin, relative to a particular event.
        end : float
            Time (ms) at which returned EEG clip should end, relative to a particular event.
        normalize_time_to_ms : bool
            PTSA's EEGReader returns the time coord in seconds. When True
            (default), the time coord is converted to milliseconds so
            downstream consumers can use a single unit across both loaders.
            Pass False to preserve the native seconds coord.

    Returns:
        eeg : ptsa.data.TimeSeries
            EEG clip.
    '''

    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    mon_ = '' if mon==0 else f'_{mon}' #for tal_reader path name

    events = events.to_records()
    tal_reader = TalReader(filename=f'/data/eeg/{sub}{mon_}/tal/{sub}{mon_}_talLocs_database_bipol.mat')
    channels = tal_reader.get_monopolar_channels()
    eeg = EEGReader(events=events, channels=channels,
                    start_time=start/1000, end_time=end/1000).read()

    bipolar_pairs = tal_reader.get_bipolar_pairs()
    pairs = get_pairs(dfrow)
    pair_tuples_select = [tuple((int(pair[0]), int(pair[1]))) for pair in pairs[['contact_1', 'contact_2']].values]
    bipolar_pairs = np.asarray([pair for pair in bipolar_pairs if tuple((int(pair[0]), int(pair[1]))) in pair_tuples_select], dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
    mapper= MonopolarToBipolarMapper(bipolar_pairs=bipolar_pairs)
    eeg = mapper.filter(timeseries=eeg)

    if normalize_time_to_ms:
        eeg = eeg.assign_coords(time=eeg.time * 1000.0)

    return eeg

def get_beh_eeg(
    dfrow: pd.Series,
    events: pd.DataFrame,
    save: bool = True,
    simulation_tag: str | None = None,
    mirror_buffer_ms: float = 0,
    real_data_buffer_ms: float = 0,
    window: tuple[float, float] | None = None,
) -> tuple[TimeSeries, Any]:

    '''
    Returns processed EEG signal to be analyzed for a particular behavioral contrast.

    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        save : bool
            Whether to save out the loaded raw EEG signal (True) or not (False).
        simulation_tag : str
            Label of parameter set used to generate simulated EEG signal.
        mirror_buffer_ms : float
            Length (ms) of symmetric mirror buffer appended to each side of the
            EEG before resampling/notching. 0 (default) skips mirroring. The
            historical pipeline used 1000 for {rm, rm_all, ri, word_on, voc};
            callers that want that behavior must pass it explicitly.
        real_data_buffer_ms : float
            Length (ms) of REAL adjacent EEG loaded on each side of the analysis
            window (distinct from mirror_buffer_ms — real samples, not a mirror).
            The widened window is resampled/notched as one piece; downstream
            crops the buffer before multitaper and before AEC/AEC-c/PAC envelope
            estimation. 0 (default) loads exactly the analysis window. Mutually
            exclusive with mirror_buffer_ms (project_paths.REAL_DATA_BUFFER_MS).
        window : tuple[float, float] | None
            Optional (start, end) ms window overriding the per-behavior default
            in beh_to_event_windows. Used by the word_on/voc separate pre/post
            event-locked loads (each group loaded at its own window).

    Returns:
        eeg : ptsa.data.TimeSeries
            EEG clip.
    '''

    if mirror_buffer_ms > 0 and real_data_buffer_ms > 0:
        raise NotImplementedError(
            "mirror_buffer_ms and real_data_buffer_ms are mutually exclusive "
            f"(got mirror={mirror_buffer_ms}, real={real_data_buffer_ms}). The "
            "mirror path is pre-resample while the real-data buffer is a window "
            "extension; mixing them is unsupported."
        )
    if mirror_buffer_ms > 0:
        raise NotImplementedError(
            "Mirror buffering is disabled pending the fixed-epoch-length "
            "windowing fix (code_issues #88). The legacy path mirrors BEFORE "
            "resampling, which entangles the buffer with the post-resample "
            "fixed-sample-count epoch selection. To reintroduce: (1) select the "
            "fixed-N epoch on the resampled, un-buffered signal in "
            "compute_session_fc; (2) mirror around the SELECTED epoch "
            "(post-resample) via helper.mirror_buffer; (3) clip the buffer back "
            "to exactly N samples after filtering. Do NOT restore the "
            "mirror-before-resample order."
        )

    beh = events.attrs['beh']
    # `window` overrides the default per-behavior window (used by the word_on/voc
    # separate event-locked pre/post loads); otherwise the beh default applies.
    start, end = window if window is not None else beh_to_event_windows[beh]
    # Real-data buffer widens the loaded window; the buffer is cropped downstream
    # (removed before multitaper; kept for AEC/AEC-c/PAC then cropped pre-estimate).
    # Bounds stay integer ms (window dict + buffer are integer-valued).
    start = int(round(start - real_data_buffer_ms))
    end = int(round(end + real_data_buffer_ms))

    eeg, mask = get_eeg(dfrow, events, start, end, simulation_tag=simulation_tag)
    if save: np.save(join(root_dir, beh, 'eeg', f'{ftag(dfrow)}_raw_eeg.npy'), eeg.data)

    eeg = eeg.resampled(RESAMPLE_HZ)
    # NOTCH_HARMONICS_UP_TO_HZ is None by default -> fundamental only, i.e. every
    # pre-existing low-frequency result is unchanged. Set it in config.yaml for
    # high-gamma runs, where the line harmonics fall inside the analysis band.
    eeg = notch_filter(eeg, dfrow['sub'],
                       harmonics_up_to_hz=NOTCH_HARMONICS_UP_TO_HZ)

    return eeg, mask

def notch_filter(
    eeg: TimeSeries, sub: str, harmonics_up_to_hz: float | None = None,
) -> TimeSeries:

    '''
    Applies a Butterworth filter to EEG signal at 60 or 50 Hz to remove line noise.

    Parameters:
        eeg : ptsa.data.TimeSeries
            EEG signal to be notch-filtered.
        sub : str
            Subject code. Used to decide notch filter frequency: if a German (Freiburg) subject, filter at 50 Hz, else at 60 Hz.
        harmonics_up_to_hz : float or None
            None (default) notches ONLY the fundamental -- the historical
            behaviour, bit-identical to the pre-high-gamma pipeline. A float
            additionally notches every integer harmonic of the fundamental up to
            that frequency, each with the same +/-2 Hz stopband.

            Required for bands that reach the harmonics. bands.high_gamma
            [70,150] contains the US 2nd harmonic (120 Hz) and both the German
            2nd and 3rd (100, 150 Hz) -- and at Freiburg sites 100 Hz is the
            band centre, so leaving this None there contaminates the result at
            exactly the frequency of interest.

            Harmonics whose stopband would reach Nyquist are skipped.

    Returns:
        eeg : ptsa.data.TimeSeries
            Notch-filtered EEG signal.
    '''

    fundamental = 50. if 'FR' in sub else 60.
    filter_freqs = [48., 52.] if 'FR' in sub else [58., 62.]

    from ptsa.data.filters import ButterworthFilter
    b_filter = ButterworthFilter(freq_range=filter_freqs, filt_type='stop', order=4)
    eeg = b_filter.filter(timeseries=eeg)

    if harmonics_up_to_hz is not None:
        nyquist = float(eeg.samplerate) / 2.0
        # SITE-UNIFORM: notch the UNION of BOTH mains standards' harmonics for
        # every subject, not just this subject's own fundamental.
        #
        # Per-site notching would give subjects different effective passbands --
        # a Freiburg (50 Hz) subject loses 100 and 150 Hz, a US (60 Hz) subject
        # loses 120 Hz. In high_gamma [70,150] that is 10% vs 5% of the band,
        # and the German hole sits at the band CENTRE. Averaging such subjects
        # would mean averaging over different frequency content, so any apparent
        # site difference in a group result could be a filtering artifact.
        #
        # The union costs 15% of the band for everyone instead, but every
        # subject then contributes an identical passband. Uniform loss you can
        # state in one Methods sentence beats a per-site confound.
        harmonics = sorted({
            f * k
            for f in (50., 60.)
            for k in range(2, int(harmonics_up_to_hz // f) + 1)
        })
        for f_k in harmonics:
            # A stopband reaching Nyquist cannot be fit; the rest are higher.
            if f_k + 2. >= nyquist:
                break
            hb = ButterworthFilter(freq_range=[f_k - 2., f_k + 2.],
                                   filt_type='stop', order=4)
            eeg = hb.filter(timeseries=eeg)

    return eeg

def mirror_buffer(
    eeg: TimeSeries,
    buffer_length: int,
    axis: int = -1,
    quiet: bool = False,
) -> TimeSeries:
    '''Append a mirror buffer to both sides of an EEG signal.

    For input (x_1, ..., x_n) the result is
    (x_k, ..., x_1, x_1, ..., x_n, x_n, ..., x_{n-k+1})
    where k = round(buffer_length_ms * sample_rate / 1000).

    Time coords are extended (not mirrored) so the buffered series has a
    monotonic, constant-step time axis — operations that rely on time
    (resample, time-range select) see a regular sampling grid.

    If the requested buffer exceeds the signal length, the buffer is
    clipped to all available samples and a RuntimeWarning is emitted
    (silence via `quiet=True`). See code_issues #68 for the methodology
    issue around the physiological-behavior windows.

    Parameters
    ----------
    eeg : ptsa.data.TimeSeries
        EEG to buffer.
    buffer_length : float
        Buffer duration in milliseconds.
    quiet : bool
        Suppress the buffer-clipping warning.
    '''
    sr = float(eeg.samplerate)
    n_samples = eeg.shape[-1]
    tmpt_length = int(buffer_length * (1 / 1000) * sr)
    if tmpt_length <= 0:
        raise ValueError(
            f"mirror_buffer: buffer_length={buffer_length}ms at sr={sr}Hz "
            f"gives tmpt_length={tmpt_length} (<= 0); buffer must be positive."
        )
    if tmpt_length > n_samples:
        if not quiet:
            import warnings
            warnings.warn(
                f"mirror_buffer: requested buffer_length={buffer_length}ms at "
                f"sr={sr}Hz ({tmpt_length} samples) exceeds signal length "
                f"{n_samples}. Clipping buffer to {n_samples} samples. "
                f"See code_issues #68.",
                RuntimeWarning, stacklevel=2,
            )
        tmpt_length = n_samples

    left_data = np.asarray(eeg[..., :tmpt_length].data)[..., ::-1]
    right_data = np.asarray(eeg[..., -tmpt_length:].data)[..., ::-1]

    time = np.asarray(eeg.time.data)
    dt = float(time[1] - time[0]) if len(time) >= 2 else 1000.0 / sr
    total_offset = tmpt_length * dt
    left_time = time[:tmpt_length] - total_offset
    right_time = time[-tmpt_length:] + total_offset

    coords_left = {k: eeg.coords[k] for k in eeg.coords if k != 'time'}
    coords_right = dict(coords_left)
    coords_left['time'] = left_time
    coords_right['time'] = right_time

    left = TimeSeries(data=left_data, dims=eeg.dims, coords=coords_left)
    right = TimeSeries(data=right_data, dims=eeg.dims, coords=coords_right)
    return xr.concat([left, eeg, right], dim='time')

def get_pairs(dfrow: pd.Series) -> pd.DataFrame | None:
    
    '''
    Returns the bipolar electrode pairs data for a session.
    Requires that the bipolar electrode pairs data have been already saved out (see 'Check data availability' section in WholeBrainConnectivityPPCRevision.ipynb). 
    
    Parameters:
        dfrow : pandas.Series
            Session label.
    
    Returns:
        pandas.DataFrame
            Bipolar electrode pairs data.
    
    '''
    
    path = join(root_dir, 'electrode_information', 'pairs', f'{ftag(dfrow)}_pairs.json')
    if ex(path): return pd.read_json(path).fillna('nan')
    else: return None

def get_localization(dfrow: pd.Series) -> pd.DataFrame | None:
    
    '''
    Returns the localization data for a session. Requires that the localization data have already been saved out (see the 'Check data availability' section in WholeBrainConnectivityPPCRevision.ipynb). 
    
    Parameters:
        dfrow : pandas.Series
            Session label.
    
    Returns:
        localization : pandas.DataFrame
            Localization data.
    '''
    
    path = join(root_dir, 'electrode_information', 'localization', f'{ftag(dfrow)}_localization.json')
    if ex(path): localization = pd.read_json(path).fillna('nan')
    else: return []
    localization['level_1'] = localization.apply(lambda r: tuple(r['level_1']) if isinstance(r['level_1'], list) else r['level_1'], axis=1)
    localization = localization.set_index(['level_0', 'level_1']).rename_axis([None, None], axis='index')
    
    return localization

def get_sr(dfrow: pd.Series) -> float:
    
    '''
    Returns the sampling rate of a session.
    Requires that the localization data have been already saved out in the session list DataFrame (see 'Check data availability' section in WholeBrainConnectivityPPCRevision.ipynb). 
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        
    Returns:
        sr : float
            Sampling rate.
    '''
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    sess_list_df = pd.read_json(join(root_dir, 'sess_list_df_data_check.json'))
    sess_list_df.set_index(['sub', 'exp', 'sess', 'loc', 'mon'], inplace=True)
    sr = sess_list_df.loc[(sub, exp, sess, loc, mon), 'sr']
    
    return sr

def find_overlapping_pairs(pairs: pd.DataFrame) -> set[tuple[int, int]]:
    
    '''
    Returns a list of bipolar pairs that share a monopolar contact.
    
    Parameters:
        pairs : pandas.DataFrame
            Bipolar pairs data.
    
    Returns:
        overlapping_pairs : list
            List of tuples of the form (i, j), where i is the row index of a bipolar pair in the pairs DataFrame and j is the row index of a bipolar pair that shares a monopolar contact.
    '''

    # Vectorized: build the symmetric overlap matrix via numpy broadcasting,
    # then enumerate (i, j) in row-major order to match the legacy Python
    # double-loop output (diagonal self-pairs included — see test_helper
    # ``TestFindOverlappingPairs`` for the pinned behavior).
    c1 = pairs["contact_1"].to_numpy()
    c2 = pairs["contact_2"].to_numpy()
    n = len(pairs)
    if n == 0:
        return []
    overlap_matrix = (
        (c1[:, None] == c1[None, :]) | (c1[:, None] == c2[None, :]) |
        (c2[:, None] == c1[None, :]) | (c2[:, None] == c2[None, :])
    )
    ii, jj = np.nonzero(overlap_matrix)
    return [(int(i), int(j)) for i, j in zip(ii, jj)]

def make_overlap_mask(pairs: pd.DataFrame, n_ch: int) -> NDArrayAny:
    assert len(pairs) <= n_ch, (
        f"pairs has {len(pairs)} rows but n_ch={n_ch}; "
        "find_overlapping_pairs returns indices in [0, len(pairs)), so "
        "len(pairs) > n_ch would produce out-of-bounds writes."
    )
    overlapping_pairs = find_overlapping_pairs(pairs)
    mask = np.zeros((n_ch, n_ch), dtype=bool)

    for i, j in overlapping_pairs:
        mask[i, j] = True
        mask[j, i] = True

    np.fill_diagonal(mask, True)
    return mask

def apply_overlap_mask(
    C: NDArrayAny, overlap_mask: NDArrayAny | None
) -> NDArrayAny:
    """Set the overlap-masked entries of C to NaN. Passes C through unchanged
    when overlap_mask is None (callers default overlap_mask to None when no
    overlap masking is wanted)."""
    if C is None or overlap_mask is None:
        return C
    C = np.asarray(C, float)
    if C.shape != overlap_mask.shape:
        raise ValueError(f"FC matrix shape {C.shape} != overlap mask shape {overlap_mask.shape}")
    C = C.copy()
    C[overlap_mask] = np.nan
    return C

@functools.lru_cache(maxsize=None)
def _load_region_lists() -> pd.Series:
    '''Build the cached region-information Series.

    Reads `region_translator.csv` (next to helper.py) once and constructs the
    four-entry pandas Series returned by `get_region_information`. The
    `lru_cache` keeps the result for the lifetime of the process so repeated
    calls across hundreds of plot/compute scripts do not re-hit the disk.
    '''
    # Resolve the CSV path relative to helper.py itself so the function works
    # regardless of the caller's CWD (avoids spurious FileNotFoundError when
    # scripts are invoked from a different directory, e.g. by snakemake or
    # pytest with a non-repo-root CWD).
    _csv_path = str(Path(__file__).resolve().parent / 'region_translator.csv')
    region_translator = pd.read_csv(_csv_path, na_filter=False).set_index('atlas_label')
    original_labels = np.unique(region_translator.index)
    unique_region_names = np.sort(region_translator.query('region != "nan"')['region'].unique())
    region_labels = np.char.add(np.repeat(['L ', 'R '], len(unique_region_names)).astype(str), np.tile(unique_region_names, 2).astype(str))

    region_lists = pd.Series({'region_translator': region_translator,
                              'original_labels': original_labels,
                              'unique_region_names': unique_region_names,
                              'region_labels': region_labels})
    return region_lists


def get_region_information(key: str | None = None) -> Any:
    '''
    Returns information about the regionalization scheme.

    Parameters:
        key (str): Which information to return ('region_translator', 'original_labels', 'unique_region_names', or 'region_labels').

    Caching note:
        Backed by `_load_region_lists` with `functools.lru_cache(maxsize=None)`.
        The returned Series (and the `region_translator` DataFrame inside it)
        is the SAME object across calls. Callers MUST treat it as read-only —
        in-place mutation will leak to every subsequent caller in the
        process. If a caller needs to modify it, `.copy()` first.
    '''
    region_lists = _load_region_lists()
    return region_lists[key] if key is not None else region_lists


# --- Vectorized helpers backing get_atlas_labels / get_atlas_labels_by_type ---

# Sentinel tokens compared case-insensitively against the per-cell string.
_ATLAS_SENTINEL_TOKENS = frozenset({
    'nan', '[nan]', 'none', 'unknown', 'misc', '', ' ', 'left tc', '*',
})

# Talairach atlas columns trigger a substring-based carve-out: if the per-cell
# string contains any of these labels, treat the cell as missing.
_ATLAS_TAL_COLS = frozenset({'tal.region', 'mat.tal.region'})
_ATLAS_TAL_UNRELIABLE = ('Parahippocampal Gyrus', 'Uncus',
                         'Lentiform Nucleus', 'Caudate', 'Thalamus')


def _column_string_and_validity(
    col_values: pd.Series, atlas: str,
) -> tuple[np.ndarray, np.ndarray]:
    """For one atlas column, return (string-view, validity-mask) numpy arrays.

    String view: per-cell `str(value)` (matches the original
    `str(pair[atlas])`). Validity mask: True iff the cell is a usable atlas
    label (non-sentinel, plus the Talairach substring carve-out).
    """
    raw = col_values.to_numpy()
    # Per-cell str() — same as the original `str(pair[atlas])`. We do this in
    # plain Python because the values may be a mix of str / float-NaN / etc.,
    # which pandas' .astype(str) handles but at a much higher overhead than
    # a list comprehension on the underlying ndarray.
    str_arr = np.array([str(v) for v in raw], dtype=object)
    # Lowercase once via the same list comprehension pattern.
    lower_arr = np.array([s.lower() for s in str_arr], dtype=object)
    not_sentinel = ~np.isin(lower_arr, list(_ATLAS_SENTINEL_TOKENS))
    if atlas in _ATLAS_TAL_COLS:
        # Substring carve-out: True ⇔ cell contains one of the unreliable labels.
        # np.char.find expects unicode arrays; cast just for this op.
        s_unicode = str_arr.astype(str)
        bad = np.zeros(len(s_unicode), dtype=bool)
        for label in _ATLAS_TAL_UNRELIABLE:
            bad |= (np.char.find(s_unicode, label) >= 0)
        not_sentinel &= ~bad
    return str_arr, not_sentinel


def _vectorized_label_cascade(
    pairs: pd.DataFrame,
    atlases: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized atlas-priority cascade.

    For each row, returns the value (as a string) of the highest-priority
    column in `atlases` whose cell passes the sentinel + Talairach carve-out
    checks, plus the source atlas name. Rows with no valid column get the
    ('nan', 'no atlas') sentinel — matching the original `label_pair` return.
    """
    n = len(pairs)
    if n == 0 or len(atlases) == 0:
        return (
            np.array(['nan'] * n, dtype=object),
            np.array(['no atlas'] * n, dtype=object),
        )

    # Stack per-column string views and validity masks. Column j == priority j.
    str_cols: list[np.ndarray] = []
    valid_cols: list[np.ndarray] = []
    for atlas in atlases:
        s_arr, valid = _column_string_and_validity(pairs[atlas], atlas)
        str_cols.append(s_arr)
        valid_cols.append(valid)

    valid = np.stack(valid_cols, axis=1)  # (n_rows, n_atlases)
    has_any = valid.any(axis=1)
    # argmax on a boolean row returns the index of the first True (priority
    # winner). On all-False rows, returns 0 — we mask those out below.
    first_idx = valid.argmax(axis=1)

    # Gather the chosen string per row.
    str_mat = np.stack(str_cols, axis=1)  # (n_rows, n_atlases)
    chosen_label = str_mat[np.arange(n), first_idx]
    atlas_arr = np.array(atlases, dtype=object)
    chosen_source = atlas_arr[first_idx]

    labels_out = np.where(has_any, chosen_label, 'nan').astype(object)
    sources_out = np.where(has_any, chosen_source, 'no atlas').astype(object)
    return labels_out, sources_out


def _merge_localization_columns(
    pairs: pd.DataFrame,
    localization: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorized counterpart of the per-row `.apply` block that injects
    `atlases.mtl` / `atlases.dk` / `atlases.whole_brain` from `localization`
    onto `pairs`. Mutates `pairs` in place (matches existing behavior) and
    returns the reshaped localization DataFrame so callers can chain.
    """
    localization = localization.loc['pairs'].reset_index()
    # Vectorized 'c1-c2' label assembly (was a per-row .apply).
    # `localization['index']` is a column of (c1, c2) tuples; unpack with
    # numpy string concat (np.char.add handles heterogeneous numpy str dtypes
    # that the `+` operator rejects).
    idx_arr = np.asarray(localization['index'].tolist())
    left = idx_arr[:, 0].astype(str)
    right = idx_arr[:, 1].astype(str)
    localization['label'] = np.char.add(np.char.add(left, '-'), right)
    localization = localization.set_index('label')
    # Drop duplicate labels so .map() doesn't raise on a non-unique index.
    localization = localization[~localization.index.duplicated(keep='first')]
    for col in ['atlases.mtl', 'atlases.dk', 'atlases.whole_brain']:
        # Original per-row .apply unconditionally overwrites pairs[col] — NaN
        # when the label is missing from localization or the column is absent.
        if col in localization.columns:
            pairs[col] = pairs['label'].map(localization[col])
        else:
            pairs[col] = np.nan
    return localization


def get_atlas_labels(pairs: pd.DataFrame, localization: pd.DataFrame | None) -> pd.DataFrame:
    
    '''
    Returns the label from the best available brain region atlas for a session's electrode channels.
    
    Parameters:
        pairs : pandas.DataFrame
            Bipolar electrode pairs data.
        localization : pandas.DataFrame
            Localization data.
    
    Returns:
        pandas.DataFrame
            Table of bipolar electrode pairs, their best atlas label, and the atlases from which those labels were taken.
    '''
    
    if localization is not None and len(localization) > 0:
        localization = _merge_localization_columns(pairs, localization)

    _PRIORITY = ['stein.region', 'das.region', 'atlases.mtl', 'atlases.whole_brain',
                 'wb.region', 'mni.region', 'atlases.dk', 'dk.region',
                 'ind.corrected.region', 'mat.ind.corrected.region',
                 'ind.snap.region', 'mat.ind.snap.region',
                 'ind.dural.region', 'mat.ind.dural.region',
                 'ind.region', 'mat.ind.region',
                 'avg.corrected.region', 'avg.mat.corrected.region',
                 'avg.snap.region', 'avg.mat.snap.region',
                 'avg.dural.region', 'avg.mat.dural.region',
                 'avg.region', 'avg.mat.region',
                 'mat.tal.region']
    atlases = [c for c in _PRIORITY if c in pairs.columns]

    labels_arr, source_arr = _vectorized_label_cascade(pairs, atlases)
    pairs['atlas_label'] = labels_arr
    pairs['atlas'] = source_arr
    return pairs.rename({'label': 'pair_label'}, axis=1)[['pair_label', 'atlas_label', 'atlas']]

_VOLUMETRIC_ATLASES = [
    'stein.region', 'das.region',
    'atlases.mtl', 'atlases.whole_brain', 'wb.region',
    'mni.region',
    'mat.tal.region',
]

_SURFACE_ATLASES = [
    'stein.region', 'das.region',
    'atlases.dk', 'dk.region',
    'ind.corrected.region', 'mat.ind.corrected.region',
    'ind.snap.region',      'mat.ind.snap.region',
    'ind.dural.region',     'mat.ind.dural.region',
    'ind.region',           'mat.ind.region',
    'avg.corrected.region', 'avg.mat.corrected.region',
    'avg.snap.region',      'avg.mat.snap.region',
    'avg.dural.region',     'avg.mat.dural.region',
    'avg.region',           'avg.mat.region',
]

_SENTINEL_TOKENS = {'nan', '[nan]', 'none', 'unknown', 'misc', '', ' ', 'left tc', '*'}

_TAL_UNRELIABLE = {'Parahippocampal Gyrus', 'Uncus', 'Lentiform Nucleus', 'Caudate', 'Thalamus'}

# Atlas columns that exist ONLY in cmlreaders-modern pairs.json (post-2014ish).
# pyFR data has Loc1..Loc5 + mat.{ind,avg,tal}.region, but none of these
# unprefixed-or-`atlases.`-prefixed atlas columns. Used to disambiguate
# "type_1 missing because pyFR" from "type_1 missing because corrupt data".
_MODERN_ONLY_ATLAS_COLS = {
    'stein.region', 'das.region', 'atlases.mtl', 'atlases.whole_brain', 'wb.region',
    'mni.region', 'atlases.dk', 'dk.region',
    'ind.region', 'ind.corrected.region', 'ind.snap.region', 'ind.dural.region',
    'avg.region', 'avg.corrected.region', 'avg.snap.region', 'avg.dural.region',
}


def _label_pair_by_type(pair: pd.Series, atlases: Sequence[str]) -> tuple[str, str]:
    """Apply the sentinel filter + Talairach carve-out on `atlases` for one
    pair. Returns ('atlas_label', 'atlas_source') or ('nan', 'no atlas')
    when no entry survives."""
    for atlas in atlases:
        if atlas not in pair.index:
            continue
        test_region = str(pair[atlas])
        if atlas in ('tal.region', 'mat.tal.region') and any(
            unreliable in test_region for unreliable in _TAL_UNRELIABLE
        ):
            continue
        if test_region.lower() not in _SENTINEL_TOKENS:
            return test_region, atlas
    return 'nan', 'no atlas'


def get_atlas_labels_by_type(
    pairs: pd.DataFrame,
    localization: pd.DataFrame | None,
) -> pd.DataFrame:
    """Type-aware atlas labeling. Each pair is routed to the volumetric or
    surface cascade based on its `type_1` / `type_2` columns.

    Raises ValueError if either type column is missing on a non-pyFR pairs
    DataFrame (detected via absence of the modern-atlas columns), or if
    `type_1 != type_2` for any pair.
    """
    pairs = pairs.copy()

    if localization is not None and len(localization) > 0:
        localization = _merge_localization_columns(pairs, localization)

    has_type_cols = ('type_1' in pairs.columns) and ('type_2' in pairs.columns)
    has_modern_indicators = bool(_MODERN_ONLY_ATLAS_COLS.intersection(pairs.columns))

    if not has_type_cols:
        if has_modern_indicators:
            raise ValueError(
                "type_1 / type_2 columns missing in pairs.json; cannot select "
                "the volumetric vs surface cascade. (Modern atlas columns ARE "
                "present, so this is not a pyFR-era artifact.)"
            )
        # Legacy pyFR-style data: fall back to the existing Aditya cascade.
        return get_atlas_labels(pairs, localization)

    if not (pairs['type_1'] == pairs['type_2']).all():
        raise ValueError(
            "pairs.json has rows with type_1 != type_2; "
            "ambiguous which cascade to apply. Filter upstream."
        )

    # Type-aware vectorized cascade: split rows by cascade choice, run the
    # vectorized cascade on each subset, then re-assemble in original order.
    is_volumetric = pairs['type_1'].isin(('D', 'uD')).to_numpy()
    labels_out = np.empty(len(pairs), dtype=object)
    source_out = np.empty(len(pairs), dtype=object)

    for mask, cascade in ((is_volumetric, _VOLUMETRIC_ATLASES),
                          (~is_volumetric, _SURFACE_ATLASES)):
        if not mask.any():
            continue
        sub = pairs.loc[mask]
        atlases_present = [c for c in cascade if c in sub.columns]
        sub_labels, sub_sources = _vectorized_label_cascade(sub, atlases_present)
        labels_out[mask] = sub_labels
        source_out[mask] = sub_sources

    pairs['atlas_label'] = labels_out
    pairs['atlas'] = source_out
    return pairs.rename({'label': 'pair_label'}, axis=1)[
        ['pair_label', 'atlas_label', 'atlas']
    ]


def regionalize_electrodes_by_type(
    pairs: pd.DataFrame,
    localization: pd.DataFrame | None,
) -> NDArrayAny:
    """Type-aware analog of `regionalize_electrodes`. Volumetric cascade for
    D / uD electrodes, surface cascade for G / S. Falls back to the legacy
    Aditya cascade for pyFR-style pairs (no `type_1` / `type_2` columns).

    Returns the same shape output as `regionalize_electrodes`: one per-pair
    label of the form `'L amygdala'` / `'R hippocampus'`, or `np.nan` for
    unmapped channels.
    """
    regionalizations = get_atlas_labels_by_type(pairs, localization)
    region_translator = get_region_information('region_translator')
    region_translator = region_translator[~region_translator.index.duplicated(keep='first')]
    original_labels = get_region_information('original_labels')
    regionalizations['region'] = regionalizations.apply(
        lambda r: region_translator.loc[r['atlas_label'], 'region']
        if r['atlas_label'] in original_labels else 'nan',
        axis=1,
    )

    def get_hemisphere_region_label(r):
        if 'hemisphere' in pairs.columns:
            hemisphere = pairs.loc[r.name, 'hemisphere']
            if hemisphere in ['L', 'R']:
                return hemisphere
        if r['atlas_label'] in original_labels:
            hemisphere = region_translator.loc[r['atlas_label'], 'hemisphere']
            if hemisphere in ['L', 'R']:
                return hemisphere
        atlases_x = pd.DataFrame(
            [(col, i) for i, col in enumerate([
                'mni.x', 'ind.corrected.x', 'ind.snap.x', 'ind.dural.x', 'ind.x',
                'avg.corrected.x', 'avg.snap.x', 'avg.dural.x', 'avg.x',
                'tal.x', 'x',
            ])],
            columns=['atlas', 'priority'],
        ).query('atlas in @pairs.columns').sort_values('priority')['atlas'].values
        for atlas_x in atlases_x:
            x_coord = pairs.loc[r.name, atlas_x]
            if not isinstance(x_coord, (int, float)):
                continue
            if x_coord < 0:
                return 'L'
            elif x_coord > 0:
                return 'R'

    regionalizations['hemisphere'] = regionalizations.apply(
        lambda r: get_hemisphere_region_label(r), axis=1,
    )
    labels = regionalizations['hemisphere'].astype(object) + ' ' + regionalizations['region'].astype(object)
    unmapped = (regionalizations['region'] == 'nan') | regionalizations['hemisphere'].isna()
    return labels.where(~unmapped, np.nan).values


def timebin_phase_timeseries(timeseries: NDArrayAny, sample_rate: float, bin_width_ms: int = 200) -> NDArrayAny:
    return timebin_timeseries(timeseries, sample_rate, circ_mean, bin_width_ms=bin_width_ms)

def timebin_power_timeseries(timeseries: NDArrayAny, sample_rate: float, bin_width_ms: int = 200) -> NDArrayAny:
    return timebin_timeseries(timeseries, sample_rate, np.mean, bin_width_ms=bin_width_ms)

def timebin_timeseries(
    timeseries: NDArrayAny,
    sample_rate: float,
    average_function: Any,
    bin_width_ms: int = 200,
) -> NDArrayAny:
    '''Average a time series within fixed-width bins along the last axis.

    Parameters
    ----------
    timeseries : numpy.ndarray or xarray.DataArray
        Time series with timepoints along the last dimension. For xarray
        inputs the last dim must be named 'time'.
    sample_rate : float
        Sample rate (Hz).
    average_function : callable
        Reducer used per bin (e.g. ``np.mean``, ``circ_mean``).
    bin_width_ms : float
        Bin width in milliseconds.
    '''
    if hasattr(timeseries, "dims") and timeseries.dims[-1] != "time":
        raise ValueError(
            f"timebin_timeseries expects the last dimension to be 'time'; "
            f"got dims={timeseries.dims!r}."
        )
    bin_size = int(sample_rate * bin_width_ms / 1000)
    bin_count = int(np.round(timeseries.shape[-1] / bin_size))

    timebinned_timeseries = []
    for iBin in range(bin_count):
        left_edge = iBin * bin_size
        right_edge = (iBin + 1) * bin_size if iBin < bin_count - 1 else None
        this_epoch = average_function(timeseries[..., left_edge:right_edge], axis=-1)
        timebinned_timeseries.append(this_epoch)

    timebinned_timeseries = np.asarray(timebinned_timeseries)
    timebinned_timeseries = np.moveaxis(timebinned_timeseries, 0, -1)
    return timebinned_timeseries

def clip_buffer(timeseries: TimeSeries, buffer_length: int) -> TimeSeries:
    
    '''
    Returns signal after clipping buffer.
    
    Parameters:
        timeseries : xarray.DataArray, ptsa.data.TimeSeries
            Time series (EEG, power, phase) with 'time' dimension.
        buffer_length : float
            Number of samples (NOT duration) to clip from both ends of the time series.
        
    Returns
        xarray.DataArray, ptsa.data.TimeSeries
            Time series with buffer clipped.
    '''
    
    return timeseries.isel(time=np.arange(buffer_length, len(timeseries['time'])-buffer_length))

def get_phase(eeg: TimeSeries, freqs: Sequence[float] | NDArrayAny) -> Any:
    '''Spectral phase time series via Morlet wavelet convolution.

    Parameters
    ----------
    eeg : ptsa.data.TimeSeries
        EEG clip.
    freqs : numpy.array
        Wavelet frequencies at which to extract phase.
    '''
    wavelet_filter = MorletWaveletFilter(freqs=freqs, width=5, output='phase',
                                         complete=True)
    phase = wavelet_filter.filter(timeseries=eeg)
    phase = phase.transpose('event', 'channel', 'frequency', 'time')
    return phase

def get_power(eeg: TimeSeries, freqs: Sequence[float] | NDArrayAny) -> Any:
    
    '''
    Returns time series of spectral power values. Performs Morlet wavelet convolution, log-transforms power, clips buffer, and z-scores power values.
    
    Parameters:
        eeg : ptsa.data.TimeSeries
            EEG clip.
        freqs : numpy.array
            Wavelet frequencies at which to extract phase values.
            
    Returns
        power : ptsa.data.TimeSeries
            Time series of power values.
    '''
    
    wavelet_filter = MorletWaveletFilter(freqs=freqs, width=5, output='power', complete=True)
    power = wavelet_filter.filter(timeseries=eeg)
    power = power.transpose('event', 'channel', 'frequency', 'time')
    
    power = np.log10(power)
    
    sr = float(eeg.samplerate)
    buffer_length = int(sr/1000*1000)
    power = clip_buffer(power, buffer_length)
    
    mean = power.mean('time').mean('event')
    std = power.mean('time').std('event')
    power = (power-mean)/std 
    
    return power

def cohens_d(x: NDArrayAny, y: NDArrayAny) -> float:
    
    '''
    Returns Cohen's d given two independent samples.
    
    Parameters:
        x (numpy.array): First sample.
        y (numpy.array): Second sample.
        
    Returns:
        d (float): Cohen's d.
    '''
    
    s = np.sqrt(((len(x)-1)*(np.std(x, axis=0, ddof=1)**2) + (len(y)-1)*(np.std(y, axis=0, ddof=1)**2))/(len(x)+len(y)-2))
    d = (np.mean(x, axis=0) - np.mean(y, axis=0))/s
    return d

def welchs_t(x: NDArrayAny, y: NDArrayAny) -> float:
    
    '''
    Returns Welch's t-statistic for two independent samples.
    
    Parameters:
        x (numpy.array): First sample.
        y (numpy.array): Second sample.
        
    Returns:
        (float): Welch's t-statistic.
    '''
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.shape[0], y.shape[0]
    var_x = np.var(x, axis=0, ddof=1)
    var_y = np.var(y, axis=0, ddof=1)
    # Welch's t (unequal variances); equals ttest_ind(equal_var=False).statistic
    return (np.mean(x, axis=0) - np.mean(y, axis=0)) / np.sqrt(var_x / nx + var_y / ny)

def _simulated_electrode_regionalizations(
    eeg: TimeSeries, dfrow: pd.Series, eeg_data_source: str,
) -> tuple[list[Any], list[str]]:
    """Per-channel region labels + L/R hemisphere groups for the simulated
    EEG, used to build the block-diagonal coupling target (phase covariance
    OR amplitude-envelope correlation). Factored out so the phase and AEC
    DGP branches of replace_w_simulated_EEG share one region mapping."""
    pairs = get_pairs(dfrow)
    if eeg_data_source == 'ptsa':
        # confirm that EEG channels match pairs dataframe used for localizations
        contact_numbers = [[int(pair.item()[0].decode('utf-8')),
                            int(pair.item()[1].decode('utf-8')),
                            pair]
                           for pair in eeg.channel]
        contact_numbers = pd.DataFrame(contact_numbers, columns=['contact_1', 'contact_2', 'eeg_pair'])
        merge_columns = ['contact_1', 'contact_2']
        pairs = pairs.merge(contact_numbers[merge_columns], on=merge_columns)
        assert len(pairs) == len(contact_numbers)
    elif eeg_data_source != 'cmlreaders':
        raise ValueError
    localization = get_localization(dfrow)
    regionalizations = regionalize_electrodes_by_type(pairs, localization)
    region_series = pd.Series(regionalizations)
    has_hemisphere_mask = region_series.str.startswith('L ') | region_series.str.startswith('R ') | region_series.isna()
    if not has_hemisphere_mask.all():
        raise ValueError
    # use hemispheres for simple regional grouping (rare NaN region -> 'Right')
    region_groups = ['Left' if left else 'Right' for left in pd.Series(regionalizations).str.startswith('L ')]
    return list(regionalizations), list(region_groups)


def _simulate_aec_envelope_eeg(
    original_eeg: TimeSeries,
    eeg: TimeSeries,
    dfrow: pd.Series,
    parameters: dict[str, Any],
    condition_mask: NDArrayAny,
    eeg_data_source: str,
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
    regionalizations, region_groups = _simulated_electrode_regionalizations(
        eeg, dfrow, eeg_data_source)
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

    # cmldask workers don't inherit the driver's sys.path; add sim/ so
    # `from fc_aec_dgp import sample_eeg_aec` resolves on remote workers.
    import sys as _sys
    from pathlib import Path as _Path
    _sim_dir = str(_Path(__file__).resolve().parent / "sim")
    if _sim_dir not in _sys.path:
        _sys.path.insert(0, _sim_dir)
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
    eeg_data_source: str = 'cmlreaders',
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
            eeg_data_source, time_unit, random_state, verbose)

    wavelet_amplitude = parameters['wavelet_amplitude']
    get_phase_covariance = parameters['phase_covariance_function']
    if get_phase_covariance == 'within_region_group':
        pairs = get_pairs(dfrow)
        if eeg_data_source == 'ptsa':
            # confirm that EEG channels match pairs dataframe used for localizations
            contact_numbers = [[int(pair.item()[0].decode('utf-8')),
                                int(pair.item()[1].decode('utf-8')),
                                pair]
                               for pair in eeg.channel]
            contact_numbers = pd.DataFrame(contact_numbers, columns=['contact_1', 'contact_2', 'eeg_pair'])
            merge_columns = ['contact_1', 'contact_2']
            pairs = pairs.merge(contact_numbers[merge_columns], on=merge_columns)
            assert len(pairs) == len(contact_numbers)
        elif eeg_data_source != 'cmlreaders':
            raise ValueError
        
        localization = get_localization(dfrow)
        regionalizations = regionalize_electrodes_by_type(pairs, localization)
        region_series = pd.Series(regionalizations)
        has_hemisphere_mask = region_series.str.startswith('L ') | region_series.str.startswith('R ') | region_series.isna()
        if not has_hemisphere_mask.all():
            # print(region_series[~has_hemisphere_mask])
            # display(region_series)
            raise ValueError
        # use hemispheres for simple regional grouping (put rare NaN region channels in 'Right' group)
        region_groups = ['Left' if left else 'Right' for left in pd.Series(regionalizations).str.startswith('L ')]
        
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

    # attributes match for EEG loaded with cmlreaders but EEG loaded with PTSA has different attributes that appear to not matter
    # assert original_eeg.attrs == simulated_eeg.attrs, f'Attributes of simulated EEG do not match original. '
    #         f'Original attributes:\n{original_eeg.attrs}\n\nReplacement attributes:\n{simulated_eeg.attrs}'
    return simulated_eeg


