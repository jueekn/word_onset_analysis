"""
prepare_sessions.py

  Stage 0: build sess_list_df_initial.csv from what OpenNeuro lists for the
    cohort experiments (data_check.COHORT_EXPERIMENTS)

  Stage 1: for every session in the initial list,
    run data_check.check_data and write data_check/<sid>_data_check.json,
    plus electrode_information/pairs/*. Then aggregate to
    sess_list_df_data_check.json

  Stage 2: for every session passing data check, run
    load_events.get_events to write per-behavior event JSONs and behavioral
    stats, then form sess_list_df

Each session's BIDS files are downloaded from OpenNeuro on first use and cached
(cml_data). You are asked once per run before anything is downloaded; set
CML_AUTO_APPROVE=1 for unattended runs.

Usage:
    python prepare_sessions.py
    python prepare_sessions.py --test
    python prepare_sessions.py --n-sessions 5
    python prepare_sessions.py --n-subjects 3 --workers 4
    python prepare_sessions.py --subjects R1111M R1065J
    python prepare_sessions.py --skip-data-check
    python prepare_sessions.py --skip-events
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning
from __future__ import annotations

from typing import Any, Sequence

import argparse
import os
from os.path import join

import numpy.typing as npt
import pandas as pd

import helper
import data_check
import fc_comparison_functions as fc
from project_paths import SCRATCH_DIR, BEHAVIORS_ALL, BEHAVIORS_MAIN

NDArrayAny = npt.NDArray[Any]

BEH_DIRS = BEHAVIORS_ALL
REQUIRED_EVENT_BEHS = BEHAVIORS_MAIN
ALL_EVENT_BEHS = BEHAVIORS_ALL

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDED_SESSIONS_CSV = join(REPO_ROOT, "config", "excluded_sessions.csv")
KEY_COLS = ["sub", "exp", "sess"]


def apply_manual_session_exclusions(
    df: pd.DataFrame,
    csv_path: str = EXCLUDED_SESSIONS_CSV,
) -> pd.DataFrame:
    """Flip ``include`` to False on rows matching ``config/excluded_sessions.csv``.

    The CSV schema is ``sub, exp, sess, reason``. A row matches when all of
    ``sub, exp, sess`` equal a CSV row. Missing CSV → returns ``df`` unchanged.
    Returns a new DataFrame (does not mutate input).
    """
    if not os.path.isfile(csv_path):
        return df
    excl = pd.read_csv(csv_path)
    if excl.empty:
        return df
    df = df.copy()
    excl_keys = set(map(tuple, excl[KEY_COLS].itertuples(index=False, name=None)))
    df_keys = list(map(tuple, df[KEY_COLS].itertuples(index=False, name=None)))
    mask = pd.Series([k in excl_keys for k in df_keys], index=df.index)
    df.loc[mask, "include"] = False
    return df


def setup_dirs(root_dir: str) -> None:
    for beh in BEH_DIRS:
        os.makedirs(join(root_dir, beh, "eeg"), exist_ok=True)
        os.makedirs(join(root_dir, beh, "events"), exist_ok=True)
    # `ri` (recall-intrusion) isn't a thesis behavior (BEHAVIORS_ALL excludes
    # it) but load_events.get_events writes per-session ri events here as a
    # working artifact. Create the dir up-front so the write does not crash.
    os.makedirs(join(root_dir, "ri", "events"), exist_ok=True)
    os.makedirs(join(root_dir, "data_check"), exist_ok=True)
    os.makedirs(join(root_dir, "behavioral_stats"), exist_ok=True)
    os.makedirs(join(root_dir, "electrode_information", "pairs"), exist_ok=True)


def stage_setup(root_dir: str, force: bool = False) -> None:
    setup_dirs(root_dir)
    initial = join(root_dir, "sess_list_df_initial.csv")
    if force or not os.path.exists(initial):
        print(f"[setup] building {initial}")
        data_check.build_sess_list_df_initial(root_dir)
    else:
        print(f"[setup] {initial} exists, skipping (pass --force-setup to rebuild)")


def run_check_data_worker(dfrow: pd.Series, root_dir: str) -> str:
    """Compute and save data_check JSON for one session."""
    import helper
    from misc import ftag
    helper.root_dir = root_dir
    check = data_check.check_data(dfrow, root_dir)
    check.to_json(join(root_dir, "data_check", f"{ftag(dfrow)}_data_check.json"))
    return ftag(dfrow)


def run_get_events_worker(dfrow: pd.Series, root_dir: str) -> str:
    """Compute and save per-behavior event JSONs for one session."""
    import helper
    import load_events
    from misc import ftag
    helper.root_dir = root_dir
    load_events.root_dir = root_dir
    load_events.get_events(dfrow)
    return ftag(dfrow)


def stage_build_data_check_df(root_dir: str) -> pd.DataFrame:
    print("[aggregate] building sess_list_df_data_check.json")
    df = data_check.build_sess_list_df_data_check(root_dir)
    n_pass = len(df.query("include == True"))
    print(f"[aggregate] {n_pass}/{len(df)} sessions passed data check")
    return df


def stage_build_final_df(root_dir: str) -> pd.DataFrame:
    """Build sess_list_df.json under root_dir.

    Reads sess_list_df_data_check.json + per-behavior events JSONs from
    root_dir, scans each behavior for event availability, marks sessions
    missing any REQUIRED_EVENT_BEHS as include=False, and writes the
    resulting sess_list_df.json that downstream FC compute consumes.
    """
    import load_events
    load_events.root_dir = root_dir

    in_path = join(root_dir, "sess_list_df_data_check.json")
    df = pd.read_json(in_path).query("include == True").copy()
    df = df.set_index(KEY_COLS, drop=False)

    for beh in ALL_EVENT_BEHS:
        df[f"{beh}_events"] = df.apply(
            lambda r, b=beh: load_events.check_events(r, b), axis=1,  # pyright: ignore[reportUnknownLambdaType]
        )

    for beh in REQUIRED_EVENT_BEHS:
        df.loc[df.eval(f"{beh}_events == False"), "include"] = False

    df = df.reset_index(drop=True)
    n_before = int(df["include"].sum())
    df = apply_manual_session_exclusions(df)
    n_after = int(df["include"].sum())
    print(f"[final] manual exclusions dropped {n_before - n_after} session(s)")
    out_path = join(root_dir, "sess_list_df.json")
    df.to_json(out_path)

    print(f"[final] -> {out_path}")
    n_inc = len(df.query("include == True"))
    print(f"[final] {n_inc}/{len(df)} sessions included")
    for beh in ALL_EVENT_BEHS:
        n_subs = len(
            df.query(f"include == True & {beh}_events == True")["sub"].unique()
        )
        print(f"  {beh}: events available for {n_subs} subjects")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--test", action="store_true",
                   help="shortcut: 1 session, root_dir=./test_root")
    p.add_argument("--n-sessions", type=int, default=None,
                   help="row-cap: take the first N rows of sess_list_df. "
                        "A subject may occupy multiple rows.")
    p.add_argument("--n-subjects", type=int, default=None,
                   help="subject-cap: take the first K distinct subjects and "
                        "include ALL their sessions. Use this when downstream "
                        "needs ≥K subjects (e.g. Smin enforcement).")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="only these subjects (applied before the caps)")
    p.add_argument("--workers", type=int, default=1,
                   help="sessions processed at once in separate processes "
                        "(1 = in this process). Every session's files are "
                        "downloaded first when > 1.")

    p.add_argument("--skip-setup", action="store_true",
                   help="don't build sess_list_df_initial")
    p.add_argument("--force-setup", action="store_true",
                   help="rebuild setup files even if they exist")
    p.add_argument("--skip-data-check", action="store_true")
    p.add_argument("--skip-events", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="skip the post-stage build_*_df steps")

    p.add_argument("--root-dir", default=None,
                   help="defaults to project_paths.SCRATCH_DIR (config.yaml: paths.scratch_dir)")

    args = p.parse_args()
    if args.n_sessions is not None and args.n_subjects is not None:
        raise ValueError(
            f"cannot pass both --n-sessions={args.n_sessions} and "
            f"--n-subjects={args.n_subjects}; pick one")
    return args


def select_sessions(df: pd.DataFrame, args: argparse.Namespace) -> list[pd.Series]:
    """Rows of `df` as dfrows, restricted to --subjects, then capped by
    --n-subjects (all their sessions) or --n-sessions (first N rows)."""
    if args.subjects:
        df = df[df["sub"].isin(args.subjects)]
    if args.n_subjects is not None:
        first_k = df["sub"].drop_duplicates().iloc[:args.n_subjects]
        df = df[df["sub"].isin(first_k)]
    elif args.n_sessions is not None:
        df = df.iloc[:args.n_sessions]
    return [row[KEY_COLS] for _, row in df.iterrows()]


def run_stage(name: str, sess_list: Sequence[pd.Series], worker: Any,
              root_dir: str, workers: int) -> None:
    n_subj = len({r["sub"] for r in sess_list})
    print(f"[{name}] {len(sess_list)} sessions ({n_subj} subjects)")
    if workers > 1:
        helper.prefetch_bids(sess_list)
    fc.run_sessions(worker, sess_list, name, workers=workers, quiet=True, root_dir=root_dir)


def main() -> None:
    args = parse_args()

    if args.test:
        args.n_sessions = args.n_sessions or 1
        args.root_dir = args.root_dir or "./test_root"
        print(f"[test] {args.n_sessions} session(s), root_dir={args.root_dir}")

    root_dir = args.root_dir or str(SCRATCH_DIR)
    helper.root_dir = root_dir
    print(f"[setup] root_dir = {root_dir}")

    if not args.skip_setup:
        stage_setup(root_dir, force=args.force_setup)
    else:
        print("[skip] stage 0 (setup)")

    if not args.skip_data_check:
        sess_list = select_sessions(pd.read_csv(join(root_dir, "sess_list_df_initial.csv")), args)
        run_stage("check_data", sess_list, run_check_data_worker, root_dir, args.workers)
        if not args.skip_aggregate:
            stage_build_data_check_df(root_dir)
    else:
        print("[skip] stage 1 (data_check)")

    if not args.skip_events:
        data_check_df = pd.read_json(
            join(root_dir, "sess_list_df_data_check.json")
        ).query("include == True")
        sess_list = select_sessions(data_check_df, args)
        run_stage("get_events", sess_list, run_get_events_worker, root_dir, args.workers)
        if not args.skip_aggregate:
            stage_build_final_df(root_dir)
    else:
        print("[skip] stage 2 (events)")


if __name__ == "__main__":
    main()
