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
from project_paths import PAIR_DISTANCE_THRESHOLD_MM

NDArrayAny = npt.NDArray[Any]

# data_check-internal QC constants. Non-critical (validation mechanics / cohort
# scope, not scientific tunables), so they live here rather than config.yaml —
# but are module-level + importable for other modules and tests.
DATA_CHECK_EEG_WINDOW_MS: tuple[int, int] = (-1500, 1500)
DATA_CHECK_PHASE_FREQ_HZ: float = 3.0
COHORT_EXPERIMENTS: tuple[str, ...] = ("FR1", "catFR1", "pyFR")


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
        missing = sorted(set(pairs['label']) - set(raw.ch_names))
        data_check['eeg_pairs_match'] = not missing
        assert not missing, f'pairs absent from the recording: {missing}'
        raw.pick(list(pairs['label']))
        flat = _constant_channels(raw)
        pairs = pairs[~flat].reset_index(drop=True)
        data_check['no_bad_channels_removed'] = int(flat.sum())
        assert len(pairs), 'every pair is flat'
        pairs.to_json(_pairs_path(dfrow, root_dir))

        events = reader.load_events()
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
        return eeg.resampled(helper.RESAMPLE_HZ), pairs, data_check
    except Exception as e:
        data_check['eeg_error'] = repr(e)
        return None, None, data_check


def check_phase(eeg: Any) -> bool:
    try:
        phase = helper.get_phase(eeg, [DATA_CHECK_PHASE_FREQ_HZ])
        helper.timebin_phase_timeseries(phase.data, float(phase.samplerate))
        return True
    except Exception:
        return False


def check_data(dfrow: pd.Series, root_dir: str) -> pd.Series:
    data_check = pd.Series({'pairs': False, 'eeg': False, 'phase': False, 'regionalizations': False})

    pairs, data_check['long_distance_pairs_count'], data_check['pairs'] = check_pairs(dfrow)
    if not data_check['pairs']:
        return data_check

    eeg, pairs, eeg_data_check = check_eeg(dfrow, pairs, root_dir)
    for k in eeg_data_check.keys():
        data_check[k] = eeg_data_check[k]
    if not data_check['eeg']:
        return data_check
    data_check['pairs_count'] = len(pairs)

    data_check['phase'] = check_phase(eeg)

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
    dropping genuinely sub-500 Hz recordings), or it is not phase-encoded.
    Each excluded session is also stamped with a first-cause `exclusion_reason`
    (`sr_missing` > `sub_500hz` > `data_quality`) for the exclusion report.
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
    _exclude(sess_list_df['phase'].eq(False), 'data_quality')
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
    for col in ('sr', 'phase'):
        if col not in checks_df.columns:
            checks_df[col] = np.nan
    sess_list_df = sess_list_df.join(checks_df)

    sess_list_df = apply_inclusion_rules(sess_list_df, MIN_SAMPLE_RATE_HZ)

    def _deny(key: tuple[str, str, int], reason: str, override: bool = False) -> None:
        if key in sess_list_df.index:
            if override or sess_list_df.at[key, 'exclusion_reason'] == '':
                sess_list_df.at[key, 'exclusion_reason'] = reason
            sess_list_df.at[key, 'include'] = False

    for key in DENYLIST:
        _deny(key, 'denylist')

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
