"""exclusion_log.py — event/session exclusion-count logging for the FC pipeline.

Each (session, contrast) accumulates input / per-stage-excluded / final event
COUNTS as a long-format table and writes ONE per-session CSV under
``EXCLUSION_COUNTS_DIR/per_session/`` — race-free under the parallel (per-session)
pipeline. ``aggregate_exclusion_logs`` concatenates them into
``event_exclusions.csv``; ``check_invariant`` confirms
``input_count - sum(*_excluded) == final_count`` per session×contrast (so a
missed event-drop stage surfaces); ``summarize_event_exclusions`` builds the
subject-level distribution report (counts, %-of-input, %-of-previous-stage).

Long schema (one row per count):
    config, subject, experiment, session, contrast, tag, value
``tag`` is ``input_count``, ``<stage>_excluded`` (one per ordered exclusion
stage), or ``final_count``; ``value`` is an event count. Paths + the ``config``
stamp come from config.yaml via project_paths.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportUnknownLambdaType=warning, reportAttributeAccessIssue=warning
from __future__ import annotations

import glob
import os
from os.path import join
from typing import Any, Sized, cast

import numpy as np
import pandas as pd

from project_paths import CONFIG_NAME, EXCLUSION_COUNTS_DIR

NDArrayAny = np.ndarray[Any, np.dtype[Any]]

ID_COLS = ["config", "subject", "experiment", "session", "contrast"]
LONG_COLS = ID_COLS + ["tag", "value"]


def _count(x: "int | Sized") -> int:
    """Accept either an int count or any sized object (DataFrame/array/list)."""
    return int(x) if isinstance(x, (int, np.integer)) else int(len(x))


def _session_keys(dfrow: pd.Series) -> dict[str, Any]:
    row = cast("dict[str, Any]", dict(dfrow))
    return {
        "subject": str(row["sub"]),
        "experiment": str(row["exp"]),
        "session": int(row["sess"]),
    }


class ExclusionLog:
    """Accumulate event-exclusion counts for one (session, contrast).

    Usage (each call is one line at the call site):
        log = ExclusionLog(dfrow, "voc")
        log.input(all_rec_words)                      # input_count
        log.excluded("voc_cleaning", before, after)   # voc_cleaning_excluded
        log.final(kept)                               # final_count
        log.write()
    ``input``/``final``/``excluded`` accept either counts or sized objects.
    """

    def __init__(self, dfrow: pd.Series, contrast: str,
                 config: str = CONFIG_NAME,
                 out_dir: "str | os.PathLike[str]" = EXCLUSION_COUNTS_DIR) -> None:
        self.keys = _session_keys(dfrow)
        self.contrast = str(contrast)
        self.config = str(config)
        self.out_dir = str(out_dir)
        self.records: list[tuple[str, int]] = []

    def count(self, tag: str, value: "int | Sized") -> "ExclusionLog":
        self.records.append((str(tag), _count(value)))
        return self

    def input(self, events: "int | Sized", tag: str = "input_count") -> "ExclusionLog":
        return self.count(tag, events)

    def final(self, events: "int | Sized", tag: str = "final_count") -> "ExclusionLog":
        return self.count(tag, events)

    def excluded(self, name: str, before: "int | Sized", after: "int | Sized") -> "ExclusionLog":
        return self.count(f"{name}_excluded", _count(before) - _count(after))

    def to_frame(self) -> pd.DataFrame:
        base = {"config": self.config, **self.keys, "contrast": self.contrast}
        return pd.DataFrame(
            [{**base, "tag": t, "value": v} for t, v in self.records],
            columns=LONG_COLS)

    def _filename(self) -> str:
        k = self.keys
        return f"{self.config}__{k['experiment']}_{k['subject']}_{k['session']}__{self.contrast}.csv"

    def write(self) -> str:
        d = join(self.out_dir, "per_session")
        os.makedirs(d, exist_ok=True)
        path = join(d, self._filename())
        self.to_frame().to_csv(path, index=False)
        return path


def aggregate_exclusion_logs(
    out_dir: "str | os.PathLike[str]" = EXCLUSION_COUNTS_DIR,
) -> pd.DataFrame:
    """Concatenate per-session logs into ``out_dir/event_exclusions.csv``.

    Manually-excluded sessions never appear here — manual exclusion is applied
    at the data_check stage, so get_events (and its per-session log) skips them.
    """
    files = sorted(glob.glob(join(str(out_dir), "per_session", "*.csv")))
    if not files:
        df = pd.DataFrame(columns=LONG_COLS)
    else:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(join(str(out_dir), "event_exclusions.csv"), index=False)
    return df


SESSION_LONG_COLS = ["config", "contrast", "tag", "value"]


def build_session_exclusions(
    root_dir: "str | os.PathLike[str]",
    contrasts: "list[str]" = ["en", "rm", "word_on", "voc"],
    config: str = CONFIG_NAME,
    out_dir: "str | os.PathLike[str]" = EXCLUSION_COUNTS_DIR,
    fc_cond: str = "diff",
    fc_band: str = "low",
) -> pd.DataFrame:
    """Session-attrition funnel per contrast, from on-disk session-list artifacts.

    Reads ``sess_list_df_data_check.json`` (all candidate sessions + the
    data-check ``include`` flag), ``config/excluded_sessions.csv`` (manual list),
    and ``sess_list_df.json`` (per-contrast ``<beh>_events`` availability, which
    also reflects runtime get_events failures — a session that errors writes no
    event file). When per-session FC pickles exist under
    ``<root_dir>/<beh>/fc_mats/<fc_cond>/<fc_band>/`` a final
    ``required_beh_gating_excluded`` stage counts event-available sessions that
    produced no FC output — dominated by the shared session list requiring ALL
    of ``REQUIRED_EVENT_BEHS`` (a session with this contrast's events is still
    dropped if it lacks another required contrast's events), and absorbing any
    true downstream FC-compute runtime error. ``final_count`` is then the
    sessions FC was actually computed on. Funnel stages share the long schema +
    invariant of the event log: ``input_count − Σ(*_excluded) == final_count``.
    Writes ``out_dir/session_exclusions.csv`` (columns config, contrast, tag,
    value).
    """
    from misc import ftag
    rd = str(root_dir)

    dc = pd.read_json(join(rd, "sess_list_df_data_check.json"))
    n_input = len(dc)
    # Per-reason data_check breakdown (partitions the excluded set). Manual
    # exclusions are applied at the data_check stage now (reason 'manual'), so
    # they appear here rather than as a separate post-events stage. Falls back to
    # a single lumped tag for older caches without the exclusion_reason column.
    if "exclusion_reason" in dc.columns:
        rc = dc["exclusion_reason"].fillna("").value_counts()
        data_check_stages = [(f"{r}_excluded", int(rc.get(r, 0) or 0)) for r in
                             ("sr_missing", "sub_500hz", "data_quality",
                              "denylist", "unrecoverable", "manual",
                              "data_check_other")]
    else:
        data_check_stages = [("data_check_excluded",
                              int((~dc["include"].astype(bool)).sum()))]

    # sess_list_df.json holds the data-check-passing rows, which already exclude
    # manual sessions (manual is applied in data_check) — so no separate manual
    # stage and no manual masking here.
    final = pd.read_json(join(rd, "sess_list_df.json"))

    rows: list[dict[str, Any]] = []
    for beh in contrasts:
        col = f"{beh}_events"
        ev_avail = final[final[col].astype(bool)] if col in final else final.iloc[0:0]
        avail = len(ev_avail)
        n_events_excl = len(final) - avail
        stages = [("input_count", n_input),
                  *data_check_stages,
                  ("events_unavailable_excluded", n_events_excl)]
        # optional downstream FC-compute runtime stage (if FC pickles on disk)
        fc_dir = join(rd, beh, "fc_mats", fc_cond, fc_band)
        if os.path.isdir(fc_dir):
            n_fc = sum(os.path.isfile(join(fc_dir, f"{ftag(r)}_fc_mats.pkl"))
                       for _, r in ev_avail.iterrows())
            stages.append(("required_beh_gating_excluded", avail - n_fc))
            stages.append(("final_count", n_fc))
        else:
            stages.append(("final_count", avail))
        for tag, value in stages:
            rows.append({"config": config, "contrast": beh, "tag": tag, "value": value})
    out = pd.DataFrame(rows, columns=SESSION_LONG_COLS)
    os.makedirs(str(out_dir), exist_ok=True)
    out.to_csv(join(str(out_dir), "session_exclusions.csv"), index=False)
    return out


def check_invariant(df: pd.DataFrame) -> list[tuple[Any, int, int, int]]:
    """Per (config, …, contrast): input_count − Σ(*_excluded) must equal
    final_count. Returns the offending (keys, input, total_excluded, final)
    tuples — empty means every session is internally consistent (so no
    event-drop stage was missed)."""
    bad: list[tuple[Any, int, int, int]] = []
    id_cols = [c for c in ID_COLS if c in df.columns] or ["config"]
    for keys, g in df.groupby(id_cols):
        tags = dict(zip(g["tag"], g["value"]))
        if "input_count" in tags and "final_count" in tags:
            excl = int(sum(v for t, v in tags.items() if t.endswith("_excluded")))
            if int(tags["input_count"]) - excl != int(tags["final_count"]):
                bad.append((keys, int(tags["input_count"]), excl, int(tags["final_count"])))
    return bad


def assert_invariant(df: pd.DataFrame, label: str = "") -> None:
    """Raise ValueError if any group violates ``input − Σ(excluded) == final``.

    Thin raising wrapper over :func:`check_invariant` for use in report
    generation, where a balance failure should abort rather than print.
    """
    bad = check_invariant(df)
    if bad:
        lines = [f"{keys}: input={inp} - excluded={excl} != final={fin}"
                 for keys, inp, excl, fin in bad]
        tag = f" ({label})" if label else ""
        raise ValueError(
            f"exclusion-count invariant violated{tag}: {len(bad)} group(s) where "
            f"input - sum(excluded) != final:\n  " + "\n  ".join(lines))


def check_session_event_consistency(events: pd.DataFrame,
                                    sessions: pd.DataFrame) -> None:
    """Cross-check the event- and session-level tables, raising on mismatch.

    The event log records one entry per (session, contrast) that produced an
    event file, so the count of distinct sessions in the event log for a
    contrast must equal that contrast's ``events_available`` in the session
    funnel (``input − data_check − manual − events_unavailable``). A mismatch
    means the two tables came from different states — e.g. a stale or partial
    event log regenerated for only some sessions — and is raised rather than
    reported silently.
    """
    sess_id = [c for c in ("subject", "experiment", "session") if c in events.columns]
    bad: list[tuple[str, int, int]] = []
    for beh in sessions["contrast"].unique():
        funnel = dict(zip(sessions.loc[sessions["contrast"] == beh, "tag"],
                          sessions.loc[sessions["contrast"] == beh, "value"]))
        # events_available = input - every pre-FC exclusion (all *_excluded tags
        # except the FC-compute gating stage). Robust to the data_check reason
        # split, which replaces a single data_check_excluded with per-reason tags.
        # events here must already be restricted to ANALYZED (post-manual)
        # sessions — the caller drops manually-excluded sessions before this
        # check — so the event-log count equals events_available exactly.
        avail = int(funnel.get("input_count", 0)) - sum(
            int(v) for t, v in funnel.items()
            if t.endswith("_excluded") and t != "required_beh_gating_excluded")
        n_ev = int(events.loc[events["contrast"] == beh, sess_id]
                   .drop_duplicates().shape[0])
        if n_ev != avail:
            bad.append((str(beh), n_ev, avail))
    if bad:
        lines = [f"contrast '{b}': event-log sessions={n} != "
                 f"funnel events_available={a}" for b, n, a in bad]
        raise ValueError(
            "event/session exclusion tables disagree (stale or partial event "
            "log? re-run Stage 0 get_events for all sessions):\n  "
            + "\n  ".join(lines))


def _ordered_stages(df_contrast: pd.DataFrame) -> list[str]:
    """Ordered stage tags for one contrast: input_count, exclusions in
    first-appearance order, final_count last."""
    seen: list[str] = []
    for t in df_contrast["tag"]:
        if t not in seen:
            seen.append(t)
    excl = [t for t in seen if t.endswith("_excluded")]
    head = ["input_count"] if "input_count" in seen else []
    tail = ["final_count"] if "final_count" in seen else []
    return head + excl + tail


def summarize_event_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Subject-level distribution summary per contrast × stage.

    Sums each tag over sessions within subject, then describes the distribution
    ACROSS subjects (mean, sd, median, IQR, min, max) in three forms:
      - count          : raw summed event count
      - pct_of_input   : value / subject input_count × 100
      - pct_of_prev    : value / (events remaining entering this stage) × 100
                         (for an exclusion stage; for final_count it is the
                          surviving fraction of the last remaining count)
    Returns a long summary table (contrast, stage, metric, mean, sd, median,
    q25, q75, iqr, min, max, n_subjects).
    """
    def _describe(vals: NDArrayAny) -> dict[str, float]:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {k: float("nan") for k in
                    ("mean", "sd", "median", "q25", "q75", "iqr", "min", "max")}
        q25, q75 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "median": float(np.median(v)), "q25": q25, "q75": q75,
                "iqr": q75 - q25, "min": float(v.min()), "max": float(v.max())}

    rows: list[dict[str, Any]] = []
    for contrast, dc in df.groupby("contrast"):
        stages = _ordered_stages(dc)
        # subject × tag totals (sum over sessions within subject)
        wide = cast(pd.DataFrame,
                    dc.groupby(["subject", "tag"])["value"].sum().unstack("tag")
                    ).reindex(columns=stages)
        excl_stages = [s for s in stages if s.endswith("_excluded")]
        inp = wide["input_count"] if "input_count" in wide else None
        # Verify the funnel balances at the subject level (the unit of final
        # analysis): input − Σ(excluded) must equal final for every subject.
        if inp is not None and "final_count" in wide:
            excl_sum = (wide[excl_stages].sum(axis=1) if excl_stages
                        else pd.Series(0.0, index=wide.index))
            imbalance = (wide["input_count"].fillna(0) - excl_sum.fillna(0)
                         - wide["final_count"].fillna(0))
            offenders = imbalance[imbalance != 0]
            if len(offenders):
                raise ValueError(
                    f"exclusion-count invariant violated for contrast "
                    f"'{contrast}': {len(offenders)} subject(s) where "
                    f"input - sum(excluded) != final "
                    f"(e.g. {list(offenders.index)[:10]})")
        for stage in stages:
            if stage not in wide:
                continue
            cnt = wide[stage].to_numpy(dtype=float)
            # remaining entering this stage = input - cumulative prior exclusions.
            # input_count has no previous stage (-> 100%); an exclusion stage's
            # "previous" excludes only stages before it; final follows all.
            if stage == "input_count":
                prior: list[str] = []
            elif stage in excl_stages:
                prior = excl_stages[:excl_stages.index(stage)]
            else:
                prior = excl_stages
            if inp is not None:
                remaining = inp.to_numpy(dtype=float) - (wide[prior].sum(axis=1).to_numpy(dtype=float) if prior else 0.0)
            else:
                remaining = np.full_like(cnt, np.nan)
            forms = {"count": cnt}
            if inp is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    forms["pct_of_input"] = 100.0 * cnt / inp.to_numpy(dtype=float)
                    forms["pct_of_prev"] = 100.0 * cnt / remaining
            for metric, vals in forms.items():
                rows.append({"contrast": contrast, "stage": stage, "metric": metric,
                             "n_subjects": int(np.isfinite(np.asarray(vals, float)).sum()),
                             **_describe(vals)})
    cols = ["contrast", "stage", "metric", "mean", "sd", "median", "q25", "q75",
            "iqr", "min", "max", "n_subjects"]
    return pd.DataFrame(rows, columns=cols)
