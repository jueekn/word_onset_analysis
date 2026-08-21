# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportMissingTypeStubs=none
"""presentation_times.py — RepFR1 word-presentation-duration bug handling.

A firmware bug set item presentation to 750 ms (intended 1600 ms) for many
RepFR1 sessions. The encoding analysis window
(config.yaml behaviors.en.event_window = [250, 1250]) then runs 500 ms past
word offset on those sessions, blending word-viewing with post-offset
inter-stimulus activity. So the encoding behaviors (``en``, ``en_all``) are
dropped for any RepFR1 session below the standard presentation duration; every
other behavior is unaffected — ``word_on`` uses a [0, 600] post window that fits
within 750 ms, and ``rm`` / ``voc`` are retrieval- / vocalization-locked.

Per-(subject, session) presentation times live in the tracked CSV at
config.yaml ``paths.repfr1_presentation_times_csv`` (project_paths.
REPFR1_PRESENTATION_TIMES_CSV). The exclusion is applied at the single FC-compute
source (build_subj_mat.run_all_sess_fc): the per-session encoding pickle is never
written for a buggy session, so every downstream consumer that globs
``<beh>/fc_mats/...`` (region subj_mat, channel sess_mat, pooled case studies,
count / exclusion logs) excludes it consistently.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import pandas as pd

from project_paths import REPFR1_PRESENTATION_TIMES_CSV

# Behaviors whose EEG window extends past a 750 ms word (encoding SME + the
# non-contrast all-encoding robustness check). Their config event_window ends at
# 1250 ms; retrieval / word_on(0-600) / vocalization windows all fit within 750.
PRESENTATION_SENSITIVE_BEHAVIORS: tuple[str, ...] = ("en", "en_all")
# Intended RepFR1 word-presentation duration (ms). Sessions recorded below this
# were hit by the firmware bug and are unsafe for the [250, 1250] encoding window.
STANDARD_PRESENTATION_MS: float = 1600.0
# Only RepFR1 carries this bug; the CSV holds RepFR1 sessions exclusively.
REPFR1_EXPERIMENT: str = "RepFR1"


@lru_cache(maxsize=1)
def short_presentation_sessions() -> frozenset[tuple[str, int]]:
    """``{(subject, session)}`` for RepFR1 sessions with a below-standard word
    presentation time (the 750 ms bug). Empty if the CSV is absent."""
    if not REPFR1_PRESENTATION_TIMES_CSV.exists():
        return frozenset()
    df = pd.read_csv(REPFR1_PRESENTATION_TIMES_CSV)
    bad = df[df["presentation_time_ms"] < STANDARD_PRESENTATION_MS]
    return frozenset(
        (str(sub), int(sess))
        for sub, sess in zip(bad["subject"], bad["session"])
    )


def is_short_presentation(sub: str, exp: str, sess: int) -> bool:
    """True if ``(sub, exp, sess)`` is a RepFR1 session hit by the 750 ms bug."""
    return (
        exp == REPFR1_EXPERIMENT
        and (str(sub), int(sess)) in short_presentation_sessions()
    )


def encoding_safe_behaviors(
    sub: str, exp: str, sess: int, behaviors: Sequence[str],
) -> list[str]:
    """Drop the presentation-sensitive encoding behaviors for a
    short-presentation RepFR1 session; return ``behaviors`` unchanged otherwise.
    """
    if not is_short_presentation(sub, exp, sess):
        return list(behaviors)
    return [b for b in behaviors if b not in PRESENTATION_SENSITIVE_BEHAVIORS]
