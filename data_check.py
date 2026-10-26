"""data_check.py — pre-flight per-session data validation.

For each session OpenNeuro lists for the cohort experiments, checks that the
bipolar pair table and EEG are loadable and well-formed, drops long-range and
flat pairs, and writes the per-session data_check JSON plus the
electrode_information/pairs sidecar every later stage reads. Aggregator
build_sess_list_df_data_check() flips include=False on any failure.

Dominated by pandas / mne method chains; narrow the library-stub-noise rules
at file scope.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning, reportUnknownLambdaType=warning, reportArgumentType=warning
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import os
from os.path import join, exists as ex

import pandas as pd

import cml_data
import helper
from misc import ftag, get_dfrow
from project_paths import PAIR_DISTANCE_THRESHOLD_MM, LONGETAL

NDArrayAny = npt.NDArray[Any]

# data_check-internal QC constants. Non-critical (validation mechanics / cohort
# scope, not scientific tunables), so they live here rather than config.yaml —
# but are module-level + importable for other modules and tests.
DATA_CHECK_EEG_WINDOW_MS: tuple[int, int] = (-1500, 1500)
COHORT_EXPERIMENTS: tuple[str, ...] = tuple(LONGETAL["experiments"]) if LONGETAL else ("FR1", "catFR1", "pyFR")


def build_sess_list_df_initial(root_dir: str) -> pd.DataFrame:
    """Every (sub, exp, sess) OpenNeuro lists for COHORT_EXPERIMENTS ->
    sess_list_df_initial.csv in root_dir."""
    frames = [cml_data.session_dataframe(exp) for exp in COHORT_EXPERIMENTS]
    df = (pd.concat(frames, ignore_index=True)
            .rename(columns={'subject': 'sub', 'task': 'exp', 'session': 'sess'}))
    df['sess'] = df['sess'].astype(int)
    df = df[['sub', 'exp', 'sess']].sort_values(['exp', 'sub', 'sess']).reset_index(drop=True)
    df.to_csv(join(root_dir, 'sess_list_df_initial.csv'), index=False)
    return df


def _pairs_path(dfrow: pd.Series, root_dir: str) -> str:
    return join(root_dir, 'electrode_information', 'pairs', f'{ftag(dfrow)}_pairs.json')


def check_pairs(dfrow: pd.Series) -> tuple[pd.DataFrame | None, Any, bool]:
    """(pairs, long_distance_pairs_count, ok). Pairs farther apart than
    config.yaml pair_distance_threshold_mm are dropped."""
    try:
        pairs = helper.load_pairs_table(helper.bids_reader(dfrow))
        assert pd.isna(pairs['distance']).sum() == 0, 'missing distance values for pairs'
        far = pairs['distance'] > PAIR_DISTANCE_THRESHOLD_MM
        return pairs[~far].reset_index(drop=True), int(far.sum()), True
    except Exception as e:
        print(e)
        return None, np.nan, False


def _constant_channels(raw: Any, chunk: int = 1 << 18) -> NDArrayAny:
    """Boolean mask of channels whose whole trace equals its first sample.
    Scans in chunks and stops looking at a channel as soon as it varies, so
    live channels cost one chunk each."""
    n_ch, n_times = len(raw.ch_names), raw.n_times
    first = raw.get_data(start=0, stop=1)[:, 0]
    flat = np.ones(n_ch, dtype=bool)
    for start in range(0, n_times, chunk):
        idx = np.flatnonzero(flat)
        if not len(idx):
            break
        d = raw.get_data(picks=idx, start=start, stop=min(start + chunk, n_times))
        flat[idx] &= (d == first[idx, None]).all(axis=1)
    return flat


def check_eeg(
    dfrow: pd.Series, pairs: pd.DataFrame, root_dir: str,
    start: int = DATA_CHECK_EEG_WINDOW_MS[0], end: int = DATA_CHECK_EEG_WINDOW_MS[1],
) -> tuple[Any, pd.DataFrame | None, pd.Series]:
    """Load one WORD-locked clip through the production loader (`helper.get_eeg`)
    and drop flat pairs. Returns (eeg, pairs, data_check) with eeg None on
    failure; data_check carries the sample rate, channel list and error."""
    data_check = pd.Series({'eeg': False, 'eeg_error': np.nan, 'sr': np.nan,
                            'eeg_channels': np.nan, 'eeg_pairs_match': np.nan,
                            'no_bad_channels_removed': np.nan})
    try:
        reader = helper.bids_reader(dfrow)
        raw = reader.load_raw(acquisition=helper.BIDS_ACQUISITION)
        mono = helper.BIDS_ACQUISITION == 'monopolar'   # longetal: recording channels are contacts
        chans = (list(dict.fromkeys([*pairs['contact_label_1'], *pairs['contact_label_2']]))
                 if mono else list(pairs['label']))
        missing = sorted(set(chans) - set(raw.ch_names))
        data_check['eeg_pairs_match'] = not missing
        assert not missing, f'pairs absent from the recording: {missing}'
        raw.pick(chans)
        flat_set = set(np.asarray(chans)[_constant_channels(raw)])
        flat = ((pairs['contact_label_1'].isin(flat_set) | pairs['contact_label_2'].isin(flat_set))
                if mono else pairs['label'].isin(flat_set)).to_numpy()
        pairs = pairs[~flat].reset_index(drop=True)
        data_check['no_bad_channels_removed'] = int(flat.sum())
        assert len(pairs), 'every pair is flat'
        pairs.to_json(_pairs_path(dfrow, root_dir))

        events = reader.load_events()
        data_check['n_word'] = int((events['trial_type'] == 'WORD').sum())
        data_check['english'] = bool(events.loc[events['trial_type'] == 'WORD', 'stim_file'].astype(str).str.contains('_EN').any())   # wordpool_EN vs _SP
        word = events[events['trial_type'] == 'WORD'].iloc[:1]
        assert len(word), 'no WORD events'
        ev = pd.DataFrame({'mstime': (word['onset'] * 1000).round().astype(int).to_numpy(),
                           'eegoffset': word['sample'].astype(int).to_numpy()})
        ev.attrs['mask'] = np.ones(len(ev), dtype=bool)
        eeg, _ = helper.get_eeg(dfrow, ev, start, end)
        assert eeg.shape[0] == 1, 'the check clip runs off the recording'
        data_check['sr'] = float(eeg.samplerate)
        data_check['eeg_channels'] = list(eeg.channel.values)
        data_check['eeg'] = True
        return (eeg.resampled(helper.RESAMPLE_HZ) if helper.RESAMPLE_HZ else eeg), pairs, data_check
    except Exception as e:
        data_check['eeg_error'] = repr(e)
        return None, None, data_check


def line_peaks(eeg: Any, freqs: tuple[int, ...] = (100, 120, 150)) -> dict[str, float]:
    """Mains-harmonic power / neighbouring power in the raw (un-notched) check clip,
    median over channels (1 = no peak). Reported only: harmonics are not notched."""
    from scipy.signal import welch
    f, P = welch(np.asarray(eeg.data)[0], float(eeg.samplerate), nperseg=512)
    return {f"line_{f0}": float(np.median(P[:, np.abs(f - f0) <= 1].mean(1)
                                          / P[:, (np.abs(f - f0) >= 3) & (np.abs(f - f0) <= 8)].mean(1)))
            for f0 in freqs}


def check_data(dfrow: pd.Series, root_dir: str) -> pd.Series:
    data_check = pd.Series({'pairs': False, 'eeg': False, 'regionalizations': False})

    pairs, data_check['long_distance_pairs_count'], data_check['pairs'] = check_pairs(dfrow)
    if not data_check['pairs']:
        return data_check

    eeg, pairs, eeg_data_check = check_eeg(dfrow, pairs, root_dir)
    for k in eeg_data_check.keys():
        data_check[k] = eeg_data_check[k]
    if not data_check['eeg']:
        return data_check
    data_check['pairs_count'] = len(pairs)
    for k, v in line_peaks(eeg).items():
        data_check[k] = v

    try:
        regionalizations = helper.regionalize_electrodes_by_type(pairs)
        data_check['regionalizations'] = True
        data_check['regionalizations_count'] = len(regionalizations)
    except Exception:
        pass
    return data_check


def load_data_check(dfrow: pd.Series, root_dir: str) -> pd.Series | None:
    dfrow = get_dfrow(list(dfrow.name))
    path = join(root_dir, 'data_check', f'{ftag(dfrow)}_data_check.json')
    if ex(path):
        return pd.read_json(path, typ='series')
    return None


def apply_inclusion_rules(
    sess_list_df: pd.DataFrame, min_sample_rate_hz: float
) -> pd.DataFrame:
    """Set the rule-based `include` flags (in place) and return the frame.

    A session is excluded (include=False) if its native sample rate is missing,
    below `min_sample_rate_hz` (499 keeps the ~499.7 Hz BioSemi sessions while
    dropping genuinely sub-500 Hz recordings), or a mains harmonic in high gamma
    exceeds MAX_LINE_HARMONIC_RATIO x its neighbours (line_*). Each excluded
    session is also stamped with a first-cause `exclusion_reason`
    (`sr_missing` > `sub_500hz` > `line_noise`).
    Session-specific denylists are applied separately by the caller.
    """
    sess_list_df['sr_present'] = ~pd.isna(sess_list_df['sr'])
    sess_list_df['include'] = True
    sess_list_df['exclusion_reason'] = ''

    def _exclude(mask: "pd.Series", reason: str) -> None:
        # first-cause: only stamp sessions not already attributed to a prior rule
        fresh = mask & sess_list_df['exclusion_reason'].eq('')
        sess_list_df.loc[mask, 'include'] = False
        sess_list_df.loc[fresh, 'exclusion_reason'] = reason

    _exclude(~sess_list_df['sr_present'], 'sr_missing')
    _exclude(sess_list_df['sr'] < min_sample_rate_hz, 'sub_500hz')
    from project_paths import MAX_LINE_HARMONIC_RATIO, NOTCH_HARMONICS_UP_TO_HZ
    lines = sess_list_df.reindex(columns=['line_100', 'line_120', 'line_150'])
    if NOTCH_HARMONICS_UP_TO_HZ:   # a notched mains harmonic is no reason to exclude
        us = ~sess_list_df['sub'].astype(str).str.contains('FR')   # 60 Hz sites (helper.notch_filter)
        for f0 in (100, 120, 150):
            if f0 <= NOTCH_HARMONICS_UP_TO_HZ:
                lines.loc[(us if f0 % 60 == 0 else ~us), f'line_{f0}'] = np.nan
    _exclude((lines > MAX_LINE_HARMONIC_RATIO).any(axis=1), 'line_noise')
    return sess_list_df


# Sessions excluded by hand after inspection of full-pipeline runs.
DENYLIST: tuple[tuple[str, str, int], ...] = (
    ('R1093J', 'FR1', 0), ('R1331T', 'FR1', 0), ('CH042', 'pyFR', 2),
    ('R1277J', 'FR1', 0), ('FR140', 'pyFR', 1), ('FR160', 'pyFR', 1),
    ('FR280', 'pyFR', 0), ('UP001', 'pyFR', 3), ('R1216E', 'FR1', 0),
    ('R1216E', 'FR1', 1), ('R1235E', 'catFR1', 0), ('R1626S', 'catFR1', 8),
    ('R1100D', 'FR1', 0), ('R1100D', 'FR1', 1), ('R1408N', 'catFR1', 0),
    ('R1408N', 'catFR1', 1), ('R1275D', 'FR1', 3), ('R1310J', 'catFR1', 1),
    ('R1486J', 'catFR1', 4), ('R1486J', 'catFR1', 5), ('R1486J', 'catFR1', 6),
    ('R1486J', 'catFR1', 7),
)


def build_sess_list_df_data_check(root_dir: str) -> pd.DataFrame:
    """
    Loads sess_list_df_initial.csv, attaches load_data_check() outputs, computes include flags,
    and writes sess_list_df_data_check.json in root_dir
    """
    from project_paths import MIN_SAMPLE_RATE_HZ, UNRECOVERABLE_SESSIONS_CSV

    sess_list_df = pd.read_csv(join(root_dir, 'sess_list_df_initial.csv'))
    sess_list_df.set_index(['sub', 'exp', 'sess'], inplace=True, drop=False)

    checks = {key: load_data_check(row, root_dir) for key, row in sess_list_df.iterrows()}
    checks_df = pd.DataFrame.from_dict({k: v for k, v in checks.items() if v is not None}, orient='index')
    checks_df.index = pd.MultiIndex.from_tuples(checks_df.index, names=['sub', 'exp', 'sess'])
    if 'sr' not in checks_df.columns:
        checks_df['sr'] = np.nan
    sess_list_df = sess_list_df.join(checks_df)

    sess_list_df = apply_inclusion_rules(sess_list_df, MIN_SAMPLE_RATE_HZ)

    def _deny(key: tuple[str, str, int], reason: str, override: bool = False) -> None:
        if key in sess_list_df.index:
            if override or sess_list_df.at[key, 'exclusion_reason'] == '':
                sess_list_df.at[key, 'exclusion_reason'] = reason
            sess_list_df.at[key, 'include'] = False

    for key in DENYLIST:
        _deny(key, 'denylist')
    if LONGETAL:   # Long et al.: subjects need a complete session
        n_word = sess_list_df.get('n_word', pd.Series(0, index=sess_list_df.index)).fillna(0)
        english = sess_list_df.get('english', pd.Series(True, index=sess_list_df.index)).fillna(False).astype(bool)
        if LONGETAL.get('english_only', False):   # "complete task session in English"
            for key in sess_list_df.index[~english]:
                _deny(key, 'not_english')
        complete = (n_word >= LONGETAL['min_word_events']) & (english | (not LONGETAL.get('english_only', False)))
        has_complete = complete.groupby(sess_list_df['sub']).transform('any')
        keep_rest = LONGETAL.get('include_incomplete_sessions', False)   # then keep their other sessions too
        for key in sess_list_df.index[~complete & ~(keep_rest & has_complete)]:
            _deny(key, 'incomplete_session')

    # Per-session denylist of empirically-unrecoverable sessions (curated from
    # error triage of full-pipeline runs). Columns: subject, experiment,
    # session, error, explanation (the last two are bookkeeping).
    if UNRECOVERABLE_SESSIONS_CSV.exists():
        for _, row in pd.read_csv(UNRECOVERABLE_SESSIONS_CSV).iterrows():
            _deny((row['subject'], row['experiment'], int(row['session'])), 'unrecoverable')

    # Manual exclusion list (config/excluded_sessions.csv): curatorial removals
    # applied HERE (not after events) so the events/FC stages skip them via
    # include==False and they never enter the event log unaccounted. Reason is
    # stamped 'manual' (override) — it's the operative decision regardless of
    # any data-quality issue.
    excluded_csv = join(os.path.dirname(os.path.abspath(__file__)),
                        'config', 'excluded_sessions.csv')
    if ex(excluded_csv):
        for _, row in pd.read_csv(excluded_csv).iterrows():
            _deny((row['sub'], row['exp'], int(row['sess'])), 'manual', override=True)

    # Any remaining excluded session with no stamped reason -> catch-all, so the
    # exclusion report's per-reason tags always partition data_check_excluded.
    orphan = (~sess_list_df['include'].astype(bool)) & sess_list_df['exclusion_reason'].eq('')
    sess_list_df.loc[orphan, 'exclusion_reason'] = 'data_check_other'

    sess_list_df.to_json(join(root_dir, 'sess_list_df_data_check.json'))

    return sess_list_df
