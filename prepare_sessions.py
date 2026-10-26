"""
prepare_sessions.py

  Stage 0: build sess_list_df_initial.csv from what OpenNeuro lists for the
    cohort experiments (data_check.COHORT_EXPERIMENTS)

  Stage 1: for every session in the initial list,
    run data_check.check_data and write data_check/<sid>_data_check.json,
    plus electrode_information/pairs/*. Then aggregate to
    sess_list_df_data_check.json

  Stage 2: for every session passing data check, run
    load_events.get_events to write the word_on event JSONs, then form sess_list_df

Each session's BIDS files are downloaded from OpenNeuro on first use and cached
(cml_data). You are asked once per run before anything is downloaded; set
CML_AUTO_APPROVE=1 for unattended runs.

Stage 0 runs only when sess_list_df_initial.csv is missing (delete it to
rebuild). Simulations reuse these real sessions, events and pairs.

Usage:
    python prepare_sessions.py
    python prepare_sessions.py --n-subjects 3 --workers 4
    python prepare_sessions.py --subjects R1111M R1065J
    python prepare_sessions.py --setup-only            # stage 0 only
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
from project_paths import SCRATCH_DIR

NDArrayAny = npt.NDArray[Any]

BEH = "word_on"

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
    os.makedirs(join(root_dir, BEH, "events"), exist_ok=True)
    os.makedirs(join(root_dir, "data_check"), exist_ok=True)
    os.makedirs(join(root_dir, "electrode_information", "pairs"), exist_ok=True)


def stage_setup(root_dir: str) -> None:
    setup_dirs(root_dir)
    initial = join(root_dir, "sess_list_df_initial.csv")
    if not os.path.exists(initial):
        print(f"[setup] building {initial}")
        data_check.build_sess_list_df_initial(root_dir)


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
    """sess_list_df.json: data-check passes that have a word_on events file
    (load_events.get_events writes none for a session failing recall matching),
    minus config/excluded_sessions.csv."""
    import load_events
    load_events.root_dir = root_dir
    df = pd.read_json(join(root_dir, "sess_list_df_data_check.json")).query("include == True").copy()
    df = df.set_index(KEY_COLS, drop=False)
    df[f"{BEH}_events"] = df.apply(lambda r: load_events.check_events(r, BEH), axis=1)
    df.loc[~df[f"{BEH}_events"], "include"] = False
    df = apply_manual_session_exclusions(df.reset_index(drop=True))
    out_path = join(root_dir, "sess_list_df.json")
    df.to_json(out_path)
    print(f"[final] {int(df['include'].sum())}/{len(df)} sessions included -> {out_path}")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-subjects", type=int, default=None,
                   help="only the first K subjects (all their sessions)")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="only these subjects (applied before --n-subjects)")
    p.add_argument("--workers", type=int, default=1,
                   help="sessions processed at once in separate processes")
    p.add_argument("--setup-only", action="store_true",
                   help="only build the session listing (stage 0)")
    return p.parse_args()


def select_sessions(df: pd.DataFrame, args: argparse.Namespace) -> list[pd.Series]:
    """Rows of `df` as dfrows, restricted to --subjects, then the first
    --n-subjects subjects (all their sessions)."""
    if args.subjects:
        df = df[df["sub"].isin(args.subjects)]
    if args.n_subjects is not None:
        df = df[df["sub"].isin(df["sub"].drop_duplicates().iloc[:args.n_subjects])]
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
    root_dir = str(SCRATCH_DIR)
    helper.root_dir = root_dir
    print(f"[setup] root_dir = {root_dir}")
    stage_setup(root_dir)
    if args.setup_only:
        return

    sess_list = select_sessions(pd.read_csv(join(root_dir, "sess_list_df_initial.csv")), args)
    run_stage("check_data", sess_list, run_check_data_worker, root_dir, args.workers)
    stage_build_data_check_df(root_dir)

    data_check_df = pd.read_json(join(root_dir, "sess_list_df_data_check.json")).query("include == True")
    sess_list = select_sessions(data_check_df, args)
    run_stage("get_events", sess_list, run_get_events_worker, root_dir, args.workers)
    stage_build_final_df(root_dir)


if __name__ == "__main__":
    main()
