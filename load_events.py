"""load_events.py — per-session word_on event extraction.

For each session row, build the word_on event table (WORD + PRE_WORD copies)
under <root_dir>/word_on/events/. Called from prepare_sessions.run_get_events_worker
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
    contrast. The copies share the post events' onsets but are relabeled
    (type=pre_type, e.g. PRE_WORD); the separate-load contrast pathway epochs
    them at the pre window. Returns (combined, mask) with mask True for the POST
    (original) events and False for the PRE copies.
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


def recall_matching_ok(matcher: Any) -> bool:
    """Cohort rule kept from the multi-contrast pipeline: a session is used only if
    its recalls match for encoding (en) and retrieval (rm) and its vocalization
    events can be cleaned -- so the juee cohort stays identical although only
    word_on is analysed. (Not applied under longetal, which never required it.)"""
    try:
        clean_voc_events(matcher.events)
        for beh, proximity_buffer in (('en', 1), ('rm', 5000)):
            matcher.match_events(beh, proximity_buffer, 10, rec_window=[-1000, 0],
                                 post_rec_distance=1000, pre_rec_distance=1000)
            if not matcher.status[beh]['matching_successful']:
                return False
        return True
    except Exception:
        return False


def get_events(dfrow: pd.Series | list[Any] | NDArrayAny) -> None:
    """Write `<root_dir>/word_on/events/<ftag>_events.json` (+ metadata) for one
    session: WORD events and their PRE_WORD copies (mask True = WORD). Nothing is
    written for a session failing recall_matching_ok, which excludes it."""
    helper.root_dir = root_dir
    np.random.seed(202406)
    dfrow = get_dfrow(cast("list[Any]", list(dfrow)) if isinstance(dfrow, pd.Series) else dfrow)
    sr = get_sr(dfrow)

    import match_events
    matcher = match_events.MatchedEvents(dfrow, sr)
    if not LONGETAL and not recall_matching_ok(matcher):
        return

    beh = 'word_on'
    elog = ExclusionLog(dfrow, beh)
    all_word = matcher.events.query('type == "WORD"')
    elog.input(all_word)
    # first word of each list follows the countdown, not a blank screen
    word_evs = all_word.copy() if (LONGETAL or {}).get('include_first_word') else all_word.query('serialpos > 1').copy()
    elog.excluded('word_on_serialpos1', all_word, word_evs); elog.final(word_evs); elog.write()
    word_evs, mask = add_pre_events(fix_event_cols(word_evs), 'PRE_WORD')
    metadata = pd.Series({'beh': beh, 'sr': sr, 'mask': mask, 'matching_successful': True})
    metadata.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json'))
    word_evs.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json'))


def check_events(dfrow: pd.Series, beh: str) -> bool:
    """True if a per-behavior events JSON exists on disk for this session/behavior.

    Used by prepare_sessions.stage_build_final_df to mark sessions
    missing required event files as include=False.
    """
    name_tuple = cast("tuple[Any, ...]", dfrow.name)
    dfrow = get_dfrow(list(name_tuple))
    fname = join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json')
    return ex(fname)