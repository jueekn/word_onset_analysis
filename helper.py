"""helper.py — core IO / signal-processing / FC-orchestration toolkit.

The widest-reach module in the repo. Reads BIDS events + EEG (OpenNeuro via
cml_data + bidsreader), applies notch-filter and mirror-buffer prep, builds
bipolar pairs, runs Morlet phase or power filtering, time-bins, regionalizes
electrode-level results, and orchestrates per-session FC pipelines.

This file is dominated by pandas method chains + ptsa + bidsreader +
mne_connectivity + matplotlib pyplot calls — third-party stubs are weak
across all of them. Narrow `reportUnknownMemberType` and friends to warning
at the file level so the strict-mode signal we DO want (missing annotations,
unbound vars, real type drift across our own functions) still surfaces.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning, reportConstantRedefinition=warning
from __future__ import annotations

import functools
import warnings
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from pathlib import Path

from os.path import join, exists as ex

import pandas as pd
import xarray as xr  # pyright: ignore[reportMissingTypeStubs]

import ptsa_patches as _ptsa_patches  # noqa: F401 — applies monkey-patches on import; must precede ptsa.* imports below
_ = _ptsa_patches
from ptsa.data.filters import MorletWaveletFilter  # pyright: ignore[reportMissingTypeStubs]
from ptsa.data.timeseries import TimeSeries  # pyright: ignore[reportMissingTypeStubs]


from cstat import *  # noqa: F401,F403
from misc import *  # noqa: F401,F403
from misc import ftag  # explicit for type-checker visibility
from matrix_operations import *  # noqa: F401,F403

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
from project_paths import RESAMPLE_HZ, NOTCH_HARMONICS_UP_TO_HZ, LONGETAL  # noqa: E402

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


def load_events(dfrow: pd.Series, beh: str) -> pd.DataFrame:
    
    '''
    Loads behavioral events for a particular experimental session and behavioral contrast.
    Written by load_events.get_events (prepare_sessions stage 2).
    
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


def bids_reader(dfrow: pd.Series, eeg: bool = True) -> Any:
    """`CMLBIDSReader` for a session, fetching the session's BIDS files from
    OpenNeuro on first use (cml_data caches them under CML_BIDS_CACHE).
    eeg=False fetches only the events / channel / electrode tables."""
    from bidsreader import CMLBIDSReader
    from cml_data import get_bids_root

    sub, exp, sess = dfrow['sub'], dfrow['exp'], int(dfrow['sess'])
    root = get_bids_root(exp, subject=sub, session=sess,
                         include_timeseries=eeg, acquisition=BIDS_ACQUISITION)
    return CMLBIDSReader(root=root, subject=sub, task=exp, session=str(sess), device='ieeg')   # bidsreader only infers ieeg for R* subjects; pyFR is TJ*/UP*


def prefetch_bids(dfrows: Sequence[pd.Series], eeg: bool = True) -> None:
    """Download every listed session's BIDS files up front (one approval),
    so parallel workers find everything cached."""
    from cml_data import prefetch

    if not len(dfrows):   # e.g. a subject batch where no session passed the data check
        return
    for (exp, sub), rows in pd.DataFrame(list(dfrows)).groupby(['exp', 'sub']):
        prefetch(str(exp), [str(sub)], sorted(set(int(s) for s in rows['sess'])),
                 include_timeseries=eeg, acquisition=BIDS_ACQUISITION)


# Bipolar pairs are the analysis channels; longetal: monopolar contacts + common average reference.
BIDS_ACQUISITION: str = LONGETAL['acquisition'] if LONGETAL else 'bipolar'
# BIDS electrode `description` -> the single-letter electrode type the
# regionalization cascade keys on (volumetric atlases for depths, surface for
# grid/strip).
_BIDS_ELECTRODE_TYPE = {'depth': 'D', 'grid': 'G', 'strip': 'S'}


def load_pairs_table(reader: Any) -> pd.DataFrame:
    """The session's bipolar-pair table in the column vocabulary the rest of the
    pipeline reads (`label`, `type_1/2`, `mni.x/y/z`, `<atlas>.region`, ...).

    Built from `CMLBIDSReader.load_combined_channels`, which joins the bipolar
    channels.tsv with the electrodes.tsv of both contacts (`*_ch1` / `*_ch2`)
    and gives the pair centroid as `*_mid` (MNI152NLin6ASym). The BIDS export
    has no pair-level atlas lookup (cmlreaders' pairs.json looked the atlas up
    at the pair midpoint), so a pair takes a contact's label: contact 1's, or
    contact 2's when contact 1 has none -- the either-contact rule the
    COGS4290 CML-vs-BIDS checks validated against pairs.json. `distance` is the
    inter-contact distance in that space. Row order is the recording's channel
    order, which `get_eeg` relies on.
    """
    t = reader.load_combined_channels(acquisition=BIDS_ACQUISITION)
    if BIDS_ACQUISITION == 'monopolar':   # contacts are the channels: give each the pair schema
        t = t[t['type'].isin(['ECOG', 'SEEG'])].reset_index(drop=True)   # no scalp/EKG channels
        cols = [c for c in ('x', 'y', 'z', 'stein.region', 'wb.region', 'ind.region') if c in t]
        single = lambda d: d.assign(ch1=d['name'], ch2=d['name'],
                                    **{f'{c}_{s}': d[c] for c in cols for s in ('ch1', 'ch2', 'mid')})
        dep = t['description'].astype(str).str.lower().eq('depth')
        if LONGETAL.get('depth_reference') == 'bipolar' and dep.any():   # depths: adjacent contacts on a lead
            d = t[dep].assign(_n=pd.to_numeric(t['name'].str.extract(r'(\d+)$')[0], errors='coerce'))
            d = d.sort_values(['group', '_n'])
            nx = d.groupby('group').shift(-1)
            ok = (nx['_n'] == d['_n'] + 1).to_numpy()
            a, b = d[ok].reset_index(drop=True), nx[ok].reset_index(drop=True)
            bip = a.assign(name=a['name'] + '-' + b['name'], ch1=a['name'], ch2=b['name'],
                           **{f'{c}_ch1': a[c] for c in cols}, **{f'{c}_ch2': b[c] for c in cols},
                           **{f'{c}_mid': (pd.to_numeric(a[c], errors='coerce') + pd.to_numeric(b[c], errors='coerce')) / 2
                              for c in ('x', 'y', 'z')})
            t = pd.concat([single(t[~dep]), bip.drop(columns='_n')], ignore_index=True)
        else:
            t = single(t)
    pairs = pd.DataFrame({
        'label': t['name'].astype(str),
        'contact_label_1': t['ch1'].astype(str),
        'contact_label_2': t['ch2'].astype(str),
    })
    etype = t['description'].astype(str).str.lower().map(_BIDS_ELECTRODE_TYPE)
    pairs['type_1'] = pairs['type_2'] = etype.fillna('nan')
    for ax in 'xyz':
        pairs[f'mni.{ax}'] = pd.to_numeric(t[f'{ax}_mid'], errors='coerce')
    c1 = t[['x_ch1', 'y_ch1', 'z_ch1']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    c2 = t[['x_ch2', 'y_ch2', 'z_ch2']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    dist = np.linalg.norm(c1 - c2, axis=1)
    if BIDS_ACQUISITION == 'monopolar':   # single contacts 0; depth pairs missing coordinates kept (not NaN)
        dist = np.where(pairs['contact_label_1'] == pairs['contact_label_2'], 0.0, np.nan_to_num(dist))
    pairs['distance'] = dist
    for atlas in ('stein.region', 'wb.region', 'ind.region'):
        a = t.get(f'{atlas}_ch1', pd.Series(np.nan, index=t.index)).replace('n/a', np.nan)
        b = t.get(f'{atlas}_ch2', pd.Series(np.nan, index=t.index)).replace('n/a', np.nan)
        pairs[atlas] = a.fillna(b).fillna('nan').astype(str)
    x = pairs['mni.x']
    pairs['hemisphere'] = np.where(x < 0, 'L', np.where(x > 0, 'R', 'nan'))
    return pairs


def get_eeg(
    dfrow: pd.Series,
    events: pd.DataFrame,
    start: int,
    end: int,
    simulation_tag: str | None = None,
) -> tuple[TimeSeries, NDArrayAny]:
    
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
            EEG clip (event x channel x time), in microvolts, time in ms.
            Channels are the session's bipolar pairs in `get_pairs` order.
        numpy.array
            List of boolean variables indicating whether the event was a successful memory event (True) or an unsuccessful memory event (False).
            Events whose clip would run off either end of the recording are
            dropped from BOTH outputs (reported on stdout).
    '''
    import mne

    # mask is positional over `events`; keep it aligned through the time sort.
    sr_expected = events.attrs.get('sr')
    mask = np.asarray(events.attrs['mask'], dtype=bool)
    events = events.reset_index(drop=True).sort_values(by=['mstime', 'eegoffset'], kind='stable')
    mask = mask[events.index.to_numpy()]

    raw = bids_reader(dfrow).load_raw(acquisition=BIDS_ACQUISITION)
    pairs = get_pairs(dfrow)
    labels = list(pairs['label'].astype(str))
    chans = labels
    if BIDS_ACQUISITION == 'monopolar':   # longetal: channels are contacts or contact differences
        c1, c2 = list(pairs['contact_label_1'].astype(str)), list(pairs['contact_label_2'].astype(str))
        chans = list(dict.fromkeys(c1 + c2))
        i1, i2 = [chans.index(c) for c in c1], [chans.index(c) for c in c2]
        single = np.array(i1) == np.array(i2)
    missing = sorted(set(chans) - set(raw.ch_names))
    if missing:
        raise ValueError(f'{ftag(dfrow)}: pairs.json channels absent from the recording: {missing}')
    raw.pick(chans)                                    # also orders the channels
    sr = float(raw.info['sfreq'])
    samples = events['eegoffset'].to_numpy(dtype=int)
    # Epoch each distinct onset once (MNE refuses repeated samples), then expand
    # back: the same onset can appear twice, e.g. WORD + its PRE_WORD copy.
    uniq, inverse = np.unique(samples, return_inverse=True)
    # Keep only onsets whose whole clip is inside the recording: MNE raises (not
    # drops) when a clip starts past the end, e.g. truncated OpenNeuro EDFs.
    inside = np.flatnonzero((uniq + round(start * sr / 1000) >= 0) &
                            (uniq + round(end * sr / 1000) < raw.n_times))
    if not len(inside):
        raise ValueError(f'{ftag(dfrow)}: no event clip lies inside the recording')
    mne_events = np.column_stack([uniq[inside], np.zeros(len(inside), int), np.ones(len(inside), int)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        epochs = mne.Epochs(raw, mne_events, tmin=start / 1000.0, tmax=end / 1000.0,
                            baseline=None, preload=True, verbose=False)
    row_of = {inside[s]: i for i, s in enumerate(epochs.selection)}   # uniq index -> epoch row
    kept = np.array([i for i, u in enumerate(inverse) if u in row_of], dtype=int)
    if len(kept) < len(samples):
        print(f'[{ftag(dfrow)}] {len(samples) - len(kept)} of {len(samples)} events dropped '
              f'(clip runs off the recording)')
    mask = mask[kept]
    # MNE reads volts; the pipeline has always worked in microvolts.
    data = epochs.get_data()[[row_of[inverse[i]] for i in kept]] * 1e6
    if BIDS_ACQUISITION == 'monopolar':
        data = np.where(single[None, :, None], data[:, i1], data[:, i1] - data[:, i2])
    eeg = TimeSeries.create(data, sr, dims=('event', 'channel', 'time'),
                            coords={'event': events.index.to_numpy()[kept],   # position in the input events
                                    'channel': labels,
                                    'time': epochs.times * 1000.0})

    if sr_expected is not None:
        assert np.isclose(float(sr_expected), sr, rtol=1e-9), f'sampling rate is wrong: events say {sr_expected}, recording is {sr}'

    if simulation_tag not in ['standard', '', None]:
        # replace experimentally recorded EEG with simulated EEG to validate analysis pipeline
        from simulate_eeg import replace_w_simulated_EEG
        eeg = replace_w_simulated_EEG(eeg,
                                      dfrow,
                                      time_unit='millisecond',
                                      condition_mask=mask,
                                      simulation_tag=simulation_tag)
    if BIDS_ACQUISITION == 'monopolar':   # common average reference, per sample, over the single contacts
        x = np.asarray(eeg.data, float).copy()
        x[:, single] -= x[:, single].mean(axis=1, keepdims=True)
        eeg = eeg.copy(data=x)
    
    return eeg, mask

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

    if RESAMPLE_HZ:   # None (longetal): keep the native rate
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
    Written by data_check.check_eeg (prepare_sessions stage 1).
    
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

def get_sr(dfrow: pd.Series) -> float:
    
    '''
    Returns the sampling rate of a session.
    Read from sess_list_df_data_check.json (written by prepare_sessions stage 1).
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        
    Returns:
        sr : float
            Sampling rate.
    '''
    
    sub, exp, sess = dfrow[['sub', 'exp', 'sess']]
    sess_list_df = pd.read_json(join(root_dir, 'sess_list_df_data_check.json'))
    sess_list_df.set_index(['sub', 'exp', 'sess'], inplace=True)
    return float(sess_list_df.loc[(sub, exp, int(sess)), 'sr'])

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
    c1 = pairs["contact_label_1"].to_numpy()
    c2 = pairs["contact_label_2"].to_numpy()
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


# --- Atlas-label cascade -----------------------------------------------------

# Cells compared case-insensitively against these are "no label".
_SENTINEL_TOKENS = frozenset({
    'nan', '[nan]', 'none', 'unknown', 'misc', 'n/a', '', ' ', 'left tc', '*',
})

# Atlas priority per electrode type. The BIDS electrode tables carry three
# atlases: stein (MTL-specific, best when present), wb (whole-brain volumetric)
# and ind (individual FreeSurfer surface). Depths get the volumetric cascade,
# grids/strips the surface one.
_VOLUMETRIC_ATLASES = ['stein.region', 'wb.region']
_SURFACE_ATLASES = ['stein.region', 'ind.region']
_ALL_ATLASES = ['stein.region', 'wb.region', 'ind.region']


def _label_cascade(
    pairs: pd.DataFrame, atlases: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """For each row, the value of the highest-priority atlas column in `atlases`
    that is not a sentinel, plus that column's name; ('nan', 'no atlas') when
    none is."""
    n = len(pairs)
    atlases = [a for a in atlases if a in pairs.columns]
    labels = np.array(['nan'] * n, dtype=object)
    source = np.array(['no atlas'] * n, dtype=object)
    for atlas in reversed(atlases):          # highest priority wins: assign it last
        vals = pairs[atlas].astype(str).to_numpy()
        ok = ~np.isin(np.char.lower(vals.astype(str)), list(_SENTINEL_TOKENS))
        labels[ok] = vals[ok]
        source[ok] = atlas
    return labels, source


def get_atlas_labels_by_type(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-pair atlas label: volumetric cascade for depth electrodes (`type_1`
    D), surface cascade for grid/strip, every atlas for pairs of unknown type.
    Returns a DataFrame with `pair_label`, `atlas_label`, `atlas` (the source
    column)."""
    etype = (pairs['type_1'].astype(str).str.upper() if 'type_1' in pairs.columns
             else pd.Series(['NAN'] * len(pairs), index=pairs.index))
    labels_out = np.empty(len(pairs), dtype=object)
    source_out = np.empty(len(pairs), dtype=object)
    if LONGETAL and LONGETAL.get('atlases'):   # Long et al.: Desikan-Killiany for every contact
        labels_out[:], source_out[:] = _label_cascade(pairs, LONGETAL['atlases'])
        return pd.DataFrame({'pair_label': pairs['label'].to_numpy(),
                             'atlas_label': labels_out, 'atlas': source_out}, index=pairs.index)
    for mask, cascade in ((etype.isin(('D', 'UD')).to_numpy(), _VOLUMETRIC_ATLASES),
                          (etype.isin(('G', 'S')).to_numpy(), _SURFACE_ATLASES),
                          (~etype.isin(('D', 'UD', 'G', 'S')).to_numpy(), _ALL_ATLASES)):
        if mask.any():
            labels_out[mask], source_out[mask] = _label_cascade(pairs.loc[mask], cascade)
    return pd.DataFrame({'pair_label': pairs['label'].to_numpy(),
                         'atlas_label': labels_out, 'atlas': source_out},
                        index=pairs.index)


def regionalize_electrodes_by_type(pairs: pd.DataFrame) -> NDArrayAny:
    """One per-pair label of the form `'L amygdala'` / `'R hippocampus'`
    (atlas label via `get_atlas_labels_by_type`, mapped through
    region_translator.csv), or `np.nan` for unmapped channels."""
    regionalizations = get_atlas_labels_by_type(pairs)
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
        x_coord = pairs.loc[r.name, 'mni.x'] if 'mni.x' in pairs.columns else np.nan
        if isinstance(x_coord, (int, float)) and x_coord < 0:
            return 'L'
        if isinstance(x_coord, (int, float)) and x_coord > 0:
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
    Returns time series of spectral power values. Performs Morlet wavelet convolution, clips buffer, and z-scores power values.
    
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
