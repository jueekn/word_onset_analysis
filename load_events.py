"""load_events.py — per-session event extraction + behavioral stats.

For each session row in sess_list_df, walk through the source events
(matched by `match_events.MatchedEvents`), build per-behavior event tables
under <root_dir>/<beh>/events/, and dump a small per-session
behavioral_stats summary. Called from prepare_sessions.run_get_events_worker
per session, then aggregated by prepare_sessions.stage_build_final_df.

This file is dominated by pandas method chains (.query / .copy / .astype /
.to_json / .read_json) and untyped attribute access on
match_events.MatchedEvents. Narrow `reportUnknownMemberType` to warning at
the file level — the other strict-mode rules stay on so genuine signal isn't lost.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning
from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
from os.path import join, exists as ex

import pandas as pd

from misc import get_dfrow, ftag
from project_paths import SCRATCH_DIR as _SCRATCH_DIR, LONGETAL
import helper
from helper import get_sr
from exclusion_log import ExclusionLog

root_dir: str = str(_SCRATCH_DIR)

NDArrayAny = npt.NDArray[Any]


def fix_event_cols(events: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant cols + coerce a small set to int. Used before writing
    per-behavior events JSONs so the schema is stable across data sources.
    """
    for x in ['answer', 'exp_version', 'intrusion', 'is_stim', 'iscorrect', 'item_num', 'msoffset', 'recalled', 'stim_list', 'stim_params', 'test']:
        if x in events.columns:
            events = events.drop(x, axis=1)
    for x in ['session', 'list', 'trial', 'eegoffset', 'mstime']:
        if x in events.columns:
            events[x] = events[x].astype(int)
    return events


def add_pre_events(post_evs: pd.DataFrame, pre_type: str) -> tuple[pd.DataFrame, NDArrayAny]:
    """Append PRE-window copies of the post events for a pre/post physiological
    contrast (word_on, voc). The copies share the post events' onsets but are
    relabeled (type=pre_type, e.g. PRE_WORD / PRE_REC_WORD); the separate-load
    contrast pathway epochs them at the pre window. Returns (combined, mask) with
    mask True for the POST (original) events and False for the PRE copies — the
    same positive/negative convention as the en/rm contrasts.
    """
    pre_evs = post_evs.copy()
    pre_evs['type'] = pre_type
    combined = pd.concat([post_evs, pre_evs], ignore_index=True)
    mask = np.concatenate([np.ones(len(post_evs), dtype=bool),
                           np.zeros(len(pre_evs), dtype=bool)])
    return combined, mask


# Any vocalization (word recall OR non-word vocalization) contaminates a voc
# epoch; both count for proximity. Recall periods are matched to events BY TIME,
# not by the 'trial' label: pyFR REC_START events carry a -999 sentinel trial,
# so trial-keyed lookups silently miss. REC_END is unreliable (pyFR has none;
# its STOP is the ~113 s trial end, not the recall end), so the period end is
# the actual REC_END when one follows the REC_START, else REC_START +
# recall_period_length. Recall length is experiment-specific.
VOC_VOCALIZATION_TYPES = ("REC_WORD", "REC_WORD_VV")
RECALL_PERIOD_LENGTH_MS = {"FR1": 30000.0, "catFR1": 30000.0, "pyFR": 45000.0}
DEFAULT_RECALL_PERIOD_LENGTH_MS = 30000.0


def recall_periods(
    events: pd.DataFrame, recall_period_length: float,
) -> tuple[NDArrayAny, NDArrayAny]:
    """(start, end) onset arrays for each recall period, matched by time.

    Each REC_START opens a period; its end is the first actual REC_END before
    the next REC_START, else REC_START + recall_period_length. Returns
    (period_starts, period_ends) sorted by start.
    """
    rs = np.sort(events.loc[events["type"] == "REC_START", "mstime"].to_numpy(dtype=float))
    re_actual = np.sort(events.loc[events["type"] == "REC_END", "mstime"].to_numpy(dtype=float))
    ends = np.empty(len(rs), dtype=float)
    for i, s in enumerate(rs):
        nxt = rs[i + 1] if i + 1 < len(rs) else np.inf
        in_period = re_actual[(re_actual > s) & (re_actual < nxt)]
        ends[i] = float(in_period[0]) if len(in_period) else s + recall_period_length
    return rs, ends


def clean_voc_events(
    events: pd.DataFrame,
    pre_ms: float = 2000.0,
    active_ms: float = 1000.0,
    start_pad_ms: float = 2000.0,
    end_pad_ms: float = 1000.0,
    recall_period_length: float | None = None,
) -> pd.DataFrame:
    """Return only the REC_WORD events clean enough for the voc contrast.

    A REC_WORD at onset t (ms) is KEPT only when all hold:
      A. no other vocalization (REC_WORD/REC_WORD_VV) onset in [t-pre_ms, t)
         (pre-baseline window vocalization-free), and
      B. no other vocalization onset in (t, t+active_ms] (active window clean), and
      C. it falls inside the usable recall window [REC_START + start_pad_ms,
         REC_END - end_pad_ms] of the recall period containing t — i.e. it
         onsets at least start_pad_ms after the period's REC_START and at least
         end_pad_ms before its REC_END.

    The recall period is found BY TIME (the latest REC_START at or before t),
    not by trial label. REC_END is the actual event when present else
    REC_START + recall_period_length (auto-selected per experiment from
    RECALL_PERIOD_LENGTH_MS when recall_period_length is None). REC_WORD_VV is a
    contaminant, never a returned voc event. Raises if REC_WORD events are
    present but there is no REC_START to anchor the periods.
    """
    if recall_period_length is None:
        exp = str(events["experiment"].iloc[0]) if "experiment" in events.columns and len(events) else ""
        recall_period_length = RECALL_PERIOD_LENGTH_MS.get(exp, DEFAULT_RECALL_PERIOD_LENGTH_MS)
    period_starts, period_ends = recall_periods(events, recall_period_length)
    if bool((events["type"] == "REC_WORD").any()) and len(period_starts) == 0:
        raise ValueError(
            "clean_voc_events: REC_WORD events present but no REC_START found — "
            "cannot anchor recall-period boundaries for the voc contrast.")
    voc = events[events["type"].isin(VOC_VOCALIZATION_TYPES)]
    voc_onsets = voc["mstime"].to_numpy(dtype=float)
    voc_idx = voc.index.to_numpy()
    rec_word = events[events["type"] == "REC_WORD"]
    keep_idx: list[Any] = []
    for idx in rec_word.index:
        ti = float(events.loc[idx, "mstime"])
        other = voc_onsets[voc_idx != idx]                              # exclude self
        if np.any((other >= ti - pre_ms) & (other < ti)):              # A
            continue
        if np.any((other > ti) & (other <= ti + active_ms)):           # B
            continue
        j = int(np.searchsorted(period_starts, ti, side="right")) - 1  # period containing t
        if j < 0:                                                       # before any REC_START
            continue
        if ti < period_starts[j] + start_pad_ms:                       # C_start
            continue
        if ti > period_ends[j] - end_pad_ms:                           # C_end
            continue
        keep_idx.append(idx)
    return events.loc[events.index.isin(keep_idx)]


def get_events(dfrow: pd.Series | list[Any] | NDArrayAny) -> None:
    """Extract per-behavior events for one session and write them to disk.

    For each of {word_on, voc, rm_all, en, rm, ri, en_all}, writes a
    `<root_dir>/<beh>/events/<ftag>_events.json` plus a metadata sidecar.
    Then calls `analyze_behavior(matcher)` to dump behavioral_stats.

    Args:
        dfrow: session row in any of the accepted forms (Series with sub/exp/
            sess, or a positional tuple/list/array of the same three).
    """
    helper.root_dir = root_dir
    
    np.random.seed(202406)
    
    # pd.Series is iterable but pyright's stubs don't expose it as a
    # Sequence; cast at the boundary.
    dfrow = get_dfrow(cast("list[Any]", list(dfrow)) if isinstance(dfrow, pd.Series) else dfrow)
    sr = get_sr(dfrow)

    beh_to_proximity_buffer = {'en': 1, 'rm': 5000, 'ri': 5000, 
                               'word_on': 1, 'voc': 1, 'rm_all':1} 
    beh_to_event_count_threshold = {'en': 10, 'rm': 10, 'ri': 10, 
                                    'word_on': 0, 'voc': 0, 'rm_all':0}
    # beh_to_event_count_threshold = {'en': 6, 'rm': 6, 'ri': 6}
    
    import match_events
    matcher = match_events.MatchedEvents(dfrow, sr)
    
    for beh in ['word_on']:
        elog = ExclusionLog(dfrow, beh)
        all_word = matcher.events.query('type == "WORD"')
        elog.input(all_word)
        # first word of each list follows the countdown, not a blank screen
        word_evs = all_word.copy() if (LONGETAL or {}).get('include_first_word') else all_word.query('serialpos > 1').copy()
        elog.excluded('word_on_serialpos1', all_word, word_evs); elog.final(word_evs); elog.write()
        word_evs = fix_event_cols(word_evs)

        # Add PRE_WORD copies (pre window); mask True=WORD (post), False=PRE_WORD.
        word_evs, mask = add_pre_events(word_evs, 'PRE_WORD')

        metadata = pd.Series({'beh': beh, 'sr': sr, 'mask': mask, 'matching_successful': True})
        metadata.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json'))
        word_evs.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json'))

    for beh in ['voc']:
        # Keep only REC_WORD events whose pre-baseline / active windows are free
        # of other vocalizations (REC_WORD or REC_WORD_VV) and that sit clear of
        # the recall-period boundaries — mirrors the rm contamination rule.
        elog = ExclusionLog(dfrow, beh)
        all_rec = matcher.events.query('type == "REC_WORD"')
        elog.input(all_rec)
        voc_evs = clean_voc_events(matcher.events).copy()
        elog.excluded('voc_cleaning', all_rec, voc_evs); elog.final(voc_evs); elog.write()

        voc_evs = fix_event_cols(voc_evs)

        # Add PRE_REC_WORD copies (pre window); mask True=REC_WORD (post), False=pre.
        voc_evs, mask = add_pre_events(voc_evs, 'PRE_REC_WORD')

        metadata = pd.Series({'beh': beh, 'sr': sr, 'mask': mask, 'matching_successful': True})
        metadata.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json'))
        voc_evs.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json'))
        
    for beh in ['rm_all']:
        # MatchedEvents.all_recs is typed as DataFrame | None but is always a
        # DataFrame when get_events is reached (matcher's __init__ populates
        # it from the source events table).
        rec_evs = cast(pd.DataFrame, matcher.all_recs).copy()
        rec_evs = rec_evs.query('type == "REC_WORD"')
        rec_evs = fix_event_cols(rec_evs)
        mask = np.ones(len(rec_evs), dtype=bool)

        metadata = pd.Series({'beh': beh,
                              'sr': sr,
                              'mask': mask,
                              'matching_successful': True})
        metadata.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json'))
        rec_evs.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json'))
    
    for beh in ['en', 'rm', 'ri']:
        
        proximity_buffer = beh_to_proximity_buffer[beh]
        event_count_threshold = beh_to_event_count_threshold[beh]
        matched_events = matcher.match_events(beh, proximity_buffer, event_count_threshold,
                                              rec_window=[-1000, 0], post_rec_distance=1000, pre_rec_distance=1000)
        mask = matcher.mask[beh]
        status = pd.Series(matcher.status[beh])
        metadata = pd.Series({'beh': beh,
                              'sr': sr,
                              'mask': mask,
                              'proximity_buffer': proximity_buffer,
                              'event_count_threshold': event_count_threshold})
        metadata = pd.concat([status, metadata], axis=0)
        metadata.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json'))

        if matcher.status[beh]['matching_successful']:
            # match_events return type can be DataFrame | None; only reached
            # here when matching_successful, so cast to DataFrame.
            matched_events = fix_event_cols(cast(pd.DataFrame, matched_events))
            if beh in ('en', 'rm'):
                # Track each matched arm separately (counts should agree; a
                # mismatch flags a matching bug). Primary arm drives the
                # input/excluded/final invariant; the other arm is an extra tag.
                elog = ExclusionLog(dfrow, beh)
                if beh == 'rm':
                    all_rec = cast(pd.DataFrame, matcher.all_recs).query('type == "REC_WORD"')
                    mrec = matched_events.query('type == "REC_WORD"')
                    elog.input(all_rec); elog.excluded('rm_deliberation', all_rec, mrec); elog.final(mrec)
                    elog.count('matched_deliberation_count', len(matched_events.query('type != "REC_WORD"')))
                else:  # en
                    recalled = cast(pd.DataFrame, matcher.study_events).query('correct_recall == 1')
                    mrec = matched_events.query('correct_recall == 1')
                    elog.input(recalled); elog.excluded('en_subsequent_memory', recalled, mrec); elog.final(mrec)
                    elog.count('matched_notrecalled_count', len(matched_events.query('correct_recall == 0')))
                elog.write()
            matched_events.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json'))
         
    if matcher.study_events is not None:
        en_all_events = matcher.study_events

        mask = en_all_events['correct_recall'].values == 1
        en_all_events_metadata = pd.Series({'beh': 'en_all', 
                                            'sr': sr,
                                            'mask': mask})
        en_all_events_metadata.to_json(join(root_dir, 'en_all', 'events', f'{ftag(dfrow)}_events_metadata.json'))
        en_all_events = fix_event_cols(en_all_events)
        en_all_events.to_json(join(root_dir, 'en_all', 'events', f'{ftag(dfrow)}_events.json'))
    
    analyze_behavior(matcher)
    
def analyze_behavior(matcher: Any) -> None:
    """Compute per-session behavioral summary stats and dump to disk.

    Writes `<root_dir>/behavioral_stats/<ftag>_behavioral_stats.json` with
    counts of presented/recalled/intrusions and per-behavior match counts +
    mean event times (succ vs unsucc).

    Args:
        matcher: a `match_events.MatchedEvents` instance. Typed as Any here
            because match_events is a cluster-only import; no stubs.

    TODO(types): once match_events.MatchedEvents has explicit annotations,
    swap Any for that class. The matcher carries .dfrow, .study_events,
    .status[beh], .matched_events[beh], .mask[beh].
    """
    
    no_presented = len(matcher.study_events) if matcher.study_events is not None else np.nan
    no_recalled = matcher.status['rm']['no_successful']
    no_intrusions = matcher.status['ri']['no_unsuccessful']
    
    no_matches = {}
    mean_succ_times = {}
    mean_unsucc_times = {}
    beh_to_time_col = {'en': 'serialpos', 
                       'rm': 'rectime', 
                       'ri': 'rectime',
                       'word_on': 'serialpos',
                       'voc': 'rectime',
                       'rm_all': 'rectime', 
                      }
    
    for beh in ['en', 'rm', 'ri']:
        if matcher.status[beh]['matching_successful']:       
            no_matches[beh] = matcher.status[beh]['no_matched']
            matched_events = matcher.matched_events[beh]
            mask = matcher.mask[beh]
            time_col = beh_to_time_col[beh]
            mean_succ_times[beh] = np.mean(matched_events[time_col][mask])
            mean_unsucc_times[beh] = np.mean(matched_events[time_col][~mask])
        else:
            no_matches[beh] = np.nan
            mean_succ_times[beh] = np.nan
            mean_unsucc_times[beh] = np.nan
            
    # Build the stats dict explicitly rather than going through locals() so
    # pyright can see every name. Behavior is identical: same keys + values,
    # same order, same Series construction.
    by_var = {
        'no_matches': no_matches,
        'mean_succ_times': mean_succ_times,
        'mean_unsucc_times': mean_unsucc_times,
    }
    stats: dict[str, Any] = {
        'no_presented': no_presented,
        'no_recalled': no_recalled,
        'no_intrusions': no_intrusions,
    }
    for beh in ['en', 'rm', 'ri']:
        for var, src in by_var.items():
            stats[f'{var}_{beh}'] = src[beh]

    behavioral_stats = pd.Series(stats)
    behavioral_stats.to_json(join(root_dir, 'behavioral_stats', f'{ftag(matcher.dfrow)}_behavioral_stats.json'))
    
def load_behavioral_stats(dfrow: pd.Series) -> pd.Series:
    """Load the per-session behavioral_stats Series written by analyze_behavior.

    The dfrow's .name attribute is expected to be a tuple-like (sub, exp,
    sess) — caller is typically `df.apply(...)` over a sess_list_df.
    """
    # df.apply over sess_list_df sets row.name to a tuple multi-index entry;
    # pandas stubs type that as Hashable. Cast to tuple at the boundary.
    name_tuple = cast("tuple[Any, ...]", dfrow.name)
    dfrow = get_dfrow(list(name_tuple))
    data_check = pd.read_json(
        join(root_dir, 'behavioral_stats', f'{ftag(dfrow)}_behavioral_stats.json'),
        typ='series')
    return data_check


def check_events(dfrow: pd.Series, beh: str) -> bool:
    """True if a per-behavior events JSON exists on disk for this session/behavior.

    Used by prepare_sessions.stage_build_final_df to mark sessions
    missing required event files as include=False.
    """
    name_tuple = cast("tuple[Any, ...]", dfrow.name)
    dfrow = get_dfrow(list(name_tuple))
    fname = join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json')
    return ex(fname)