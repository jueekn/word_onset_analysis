"""
prepare_sessions.py

  Stage 0: build sess_list_df_initial.csv and
    channels_to_drop_df.json 

  Stage 1: for every session in the initial list,
    run data_check.check_data and write data_check/<sid>_data_check.json,
    plus electrode-information/pairs/* and electrode-information/localization/*.
    Then aggregate to sess_list_df_data_check.json

  Stage 2: for every session passing data check, run
    load_events.get_events to write per-behavior event JSONs and behavioral
    stats, then form sess_list_df

Usage:
    python prep_sessions.py
    python prep_sessions.py --test                    
    python prep_sessions.py --smokescreen --local     
    python prep_sessions.py --skip-data-check         
    python prep_sessions.py --skip-events             
    python prep_sessions.py --n-sessions 5 --local
"""

# pandas method chains + dask + cmldask method chains dominate; narrow stub noise.
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning
from __future__ import annotations

from typing import Any, Callable, Sequence

import argparse
import os
from os.path import join

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm.auto import tqdm

import helper
import data_check

def _new_exps():
    """Lazy import of the new-experiment cohort module (iCatFR / IFR / RepFR).

    Only the --newexps path needs it. Imported on demand so this repo can ship
    the standard FR1 / catFR1 / pyFR session pipeline without carrying 800 lines
    for cohorts it does not analyse; the error below says exactly what to copy
    if someone does need them.
    """
    try:
        import data_check_new_exps
    except ModuleNotFoundError:
        raise SystemExit(
            "--newexps needs data_check_new_exps.py, which is not part of this "
            "repo. Copy it from fc_methods_comparison_cml if you need the "
            "iCatFR / IFR / RepFR cohorts; the default FR1 / catFR1 / pyFR "
            "pipeline does not use it."
        ) from None
    return data_check_new_exps

from project_paths import (
    CLUSTER_LOGS_DIR,
    SCRATCH_DIR,
    BEHAVIORS_ALL,
    BEHAVIORS_MAIN,
    IS_NEWEXPS,
    slurm_exclude_directives,
)

try:
    from dask_client import project_dask_client as cl
    from dask.distributed import as_completed  # pyright: ignore[reportMissingTypeStubs]
except ImportError:
    cl = None  # pyright: ignore[reportConstantRedefinition]
    as_completed = None  # pyright: ignore[reportConstantRedefinition]

NDArrayAny = npt.NDArray[Any]

BEH_DIRS = BEHAVIORS_ALL
REQUIRED_EVENT_BEHS = BEHAVIORS_MAIN
ALL_EVENT_BEHS = BEHAVIORS_ALL

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDED_SESSIONS_CSV = join(REPO_ROOT, "config", "excluded_sessions.csv")

# The new experiment cohort (ICatFR1/IFR1/RepFR1) writes the SAME session-list
# filenames as the default cohort. Output separation is by directory: the
# FC_NEWEXPS / --newexps reroute in project_paths sends SCRATCH_DIR (and all
# results/intermediate dirs) to a `newexps_icat_ifr_repfr` subtree, so the whole
# downstream pipeline runs unchanged against the new cohort's own dirs. The only
# things `--newexps` changes here are WHICH sessions are enumerated and that the
# curated denylists / channel drops are skipped.


def apply_manual_session_exclusions(
    df: pd.DataFrame,
    csv_path: str = EXCLUDED_SESSIONS_CSV,
) -> pd.DataFrame:
    """Flip ``include`` to False on rows matching ``config/excluded_sessions.csv``.

    The CSV schema is ``sub, exp, sess, loc, mon, reason`` (same key columns
    as ``sess_list_df``). A row matches when all of ``sub, exp, sess, loc,
    mon`` equal a CSV row. Missing CSV → returns ``df`` unchanged.
    Returns a new DataFrame (does not mutate input).
    """
    if not os.path.isfile(csv_path):
        return df
    excl = pd.read_csv(csv_path)
    if excl.empty:
        return df
    key_cols = ["sub", "exp", "sess", "loc", "mon"]
    df = df.copy()
    excl_keys = set(map(tuple, excl[key_cols].itertuples(index=False, name=None)))
    df_keys = list(map(tuple, df[key_cols].itertuples(index=False, name=None)))
    mask = pd.Series([k in excl_keys for k in df_keys], index=df.index)
    df.loc[mask, "include"] = False
    return df

def get_default_root_dir() -> str:
    """SCRATCH_DIR as a str. Routes through project_paths (config.yaml)."""
    return str(SCRATCH_DIR)


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
    os.makedirs(join(root_dir, "electrode_information", "localization"), exist_ok=True)

def stage_setup(root_dir: str, force: bool = False, newexps: bool = False) -> None:
    setup_dirs(root_dir)

    initial = join(root_dir, "sess_list_df_initial.csv")
    if force or not os.path.exists(initial):
        print(f"[setup] building {initial}")
        if newexps:
            _new_exps().build_newexps_session_list_df(root_dir)
        else:
            data_check.build_sess_list_df_initial(root_dir)
    else:
        print(f"[setup] {initial} exists, skipping (pass --force-setup to rebuild)")

    # channels_to_drop_df.json is an old-cohort curated artifact. The newexps
    # check_data tolerates its absence (no curated channel drops), so only build
    # it for the default cohort.
    if newexps:
        print("[setup] newexps cohort: skipping channels_to_drop_df.json (no curated drops)")
        return

    chans = join(root_dir, "channels_to_drop_df.json")
    if force or not os.path.exists(chans):
        print(f"[setup] building {chans}")
        data_check.channels_to_drop(root_dir)
    else:
        print(f"[setup] {chans} exists, skipping")

def run_check_data_worker(
    dfrow_arr: Sequence[Any] | NDArrayAny, root_dir: str, newexps: bool = False
) -> None:
    """Compute and save data_check JSON for one session.

    `newexps` selects data_check_new_exps (which tolerates a missing curated
    channels_to_drop_df.json) instead of the default data_check module.
    """
    import helper
    helper.root_dir = root_dir
    if newexps:
        dc = _new_exps()
    else:
        import data_check as dc

    try:
        from misc import get_dfrow, ftag
    except ImportError:
        from helper import get_dfrow, ftag

    dfrow = get_dfrow(dfrow_arr)
    check = dc.check_data(dfrow, root_dir)
    check.to_json(join(root_dir, "data_check", f"{ftag(dfrow)}_data_check.json"))


def run_get_events_worker(dfrow_arr: Sequence[Any] | NDArrayAny, root_dir: str) -> None:
    """
    Compute and save per-behavior event JSONs for one session
    """
    import helper
    helper.root_dir = root_dir
    import load_events
    load_events.root_dir = root_dir   

    load_events.get_events(dfrow_arr)  # pyright: ignore[reportArgumentType]

def run_on_cluster(
    name: str,
    sess_list: NDArrayAny | Sequence[Any],
    worker_fn: Callable[..., Any],
    root_dir: str,
    n_workers: int,
    mem: str,
    walltime: int,
    cluster_log_dir: str,
    **worker_kwargs: Any,
) -> None:
    if cl is None:
        raise RuntimeError(
            "cmldask not available in this environment; pass --local instead."
        )
    os.makedirs(cluster_log_dir, exist_ok=True)
    client = cl(
        name, mem, n_workers, walltime=walltime,  # pyright: ignore[reportArgumentType]
        job_extra_directives=[
            f"--output={cluster_log_dir}/{name}-%j.out",
            f"--error={cluster_log_dir}/{name}-%j.err",
            # Broken-node exclusion (config.yaml cluster.excluded_nodes).
            *slurm_exclude_directives(),
        ],
    )
    try:
        futures = client.map(
            worker_fn, sess_list,
            root_dir=root_dir, pure=False, **worker_kwargs,
        )
        fut_to_item: dict[Any, Any] = dict(zip(futures, sess_list))
        n_ok, n_err = 0, 0
        errors: list[tuple[Any, str]] = []
        # `as_completed` is set non-None whenever `cl` is, by the import-time
        # check above.
        pbar = tqdm(as_completed(futures), total=len(futures),  # pyright: ignore[reportOptionalCall]
                    desc=name, smoothing=0.05)
        for fut in pbar:
            try:
                fut.result()  # pyright: ignore[reportAttributeAccessIssue]
                n_ok += 1
            except Exception as e:
                n_err += 1
                errors.append((fut_to_item.get(fut), repr(e)))
            pbar.set_postfix(ok=n_ok, err=n_err)
        print(f"[{name}] done: ok={n_ok}, err={n_err}")
        # Surface every failure (session + exception) rather than a silent
        # count — a worker crash means that session wrote no artifact and would
        # otherwise be misattributed downstream.
        for item, err in errors:
            key = tuple(item) if item is not None else "?"
            print(f"[{name}] ERROR {key}: {err}")
    finally:
        client.shutdown()


def run_locally(
    name: str,
    sess_list: NDArrayAny | Sequence[Any],
    worker_fn: Callable[..., Any],
    root_dir: str,
    **worker_kwargs: Any,
) -> None:
    n_ok, n_err = 0, 0
    pbar = tqdm(sess_list, desc=f"{name} (local)")
    for item in pbar:
        try:
            worker_fn(item, root_dir=root_dir, **worker_kwargs)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[error] {tuple(item)}: {e!r}")
        pbar.set_postfix(ok=n_ok, err=n_err)
    print(f"[{name}] done (local): ok={n_ok}, err={n_err}")

def stage_build_data_check_df(root_dir: str, newexps: bool = False) -> pd.DataFrame:
    print("[aggregate] building sess_list_df_data_check.json")
    if newexps:
        # newexps: data-driven inclusion rules only, no curated denylists.
        df = _new_exps().build_newexps_sess_list_df_data_check(root_dir)
    else:
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
    df = df.set_index(["sub", "exp", "sess", "loc", "mon"], drop=False)

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
                   help="shortcut: 1 session, local, root_dir=./test_root")
    p.add_argument("--smokescreen", action="store_true",
                   help="first 30 sessions (rows)")
    p.add_argument("--n-sessions", type=int, default=None,
                   help="row-cap: take the first N rows of sess_list_df. "
                        "A subject may occupy multiple rows.")
    p.add_argument("--n-subjects", type=int, default=None,
                   help="subject-cap: take the first K distinct subjects and "
                        "include ALL their sessions. Use this when downstream "
                        "needs ≥K subjects (e.g. Smin enforcement).")
    p.add_argument("--local", action="store_true",
                   help="run sequentially in-process, no dask/SLURM")

    p.add_argument("--newexps", action="store_true",
                   help="operate on the new experiment cohort "
                        "(ICatFR1/IFR1/RepFR1): enumerate ALL sessions with no "
                        "curated denylists / channel drops. Outputs use the same "
                        "filenames but land in the FC_NEWEXPS directory reroute "
                        "(set FC_NEWEXPS=1 for the downstream scripts too).")

    p.add_argument("--skip-setup", action="store_true",
                   help="don't build sess_list_df_initial / channels_to_drop")
    p.add_argument("--force-setup", action="store_true",
                   help="rebuild setup files even if they exist")
    p.add_argument("--skip-data-check", action="store_true")
    p.add_argument("--skip-events", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="skip the post-stage build_*_df steps")

    p.add_argument("--root-dir", default=None,
                   help="defaults to project_paths.SCRATCH_DIR (config.yaml: paths.scratch_dir)")
    p.add_argument("--cluster-log-dir", default=str(CLUSTER_LOGS_DIR))

    p.add_argument("--data-check-workers", type=int, default=100)
    p.add_argument("--data-check-mem", default="20G")

    p.add_argument("--events-workers", type=int, default=200)
    p.add_argument("--events-mem", default="10G")

    p.add_argument("--walltime", type=int, default=10000)

    args = p.parse_args()
    if args.n_sessions is not None and args.n_subjects is not None:
        raise ValueError(
            f"cannot pass both --n-sessions={args.n_sessions} and "
            f"--n-subjects={args.n_subjects}; pick one")
    return args


def _slice_for_subjects(sess_array: NDArrayAny, n_subjects: int | None) -> NDArrayAny:
    """Filter a (sub, exp, sess, loc, mon) array to the first K distinct subjects' rows."""
    if n_subjects is None:
        return sess_array
    subs_seen = []
    kept = []
    for row in sess_array:
        sub = row[0]
        if sub not in subs_seen:
            if len(subs_seen) >= n_subjects:
                continue
            subs_seen.append(sub)
        kept.append(row)
    return np.array(kept) if kept else sess_array[:0]


def apply_test_defaults(args: argparse.Namespace) -> None:
    args.local = True
    if args.n_sessions is None:
        args.n_sessions = 1
    if args.root_dir is None:
        args.root_dir = "./test_root"
    print(f"[test] 1 session, local, root_dir={args.root_dir}")


def main() -> None:
    args = parse_args()

    if args.test:
        apply_test_defaults(args)
    if args.n_sessions is None and args.smokescreen:
        args.n_sessions = 30

    # newexps is enabled by the --newexps flag OR the FC_NEWEXPS env var (the
    # latter also drives the project_paths directory reroute, so a bare
    # `export FC_NEWEXPS=1` turns on both the new-cohort enumeration here and the
    # separate output tree downstream).
    newexps = args.newexps or IS_NEWEXPS

    root_dir = args.root_dir or get_default_root_dir()
    helper.root_dir = root_dir
    print(f"[setup] root_dir = {root_dir}")
    if newexps:
        print("[setup] newexps cohort (ICatFR1/IFR1/RepFR1)")

    if not args.skip_setup:
        stage_setup(root_dir, force=args.force_setup, newexps=newexps)
    else:
        print("[skip] stage 0 (setup)")

    if not args.skip_data_check:
        initial = join(root_dir, "sess_list_df_initial.csv")
        sess_list_df = pd.read_csv(initial)
        sess_list: NDArrayAny = np.asarray(
            sess_list_df[["sub", "exp", "sess", "loc", "mon"]].values)
        if args.n_subjects is not None:
            sess_list = _slice_for_subjects(sess_list, args.n_subjects)
        elif args.n_sessions is not None:
            sess_list = sess_list[: args.n_sessions]
        n_subj = len(set(row[0] for row in sess_list))
        print(f"[stage 1] data_check on {len(sess_list)} sessions ({n_subj} subjects)")

        if args.local:
            run_locally("check_data", sess_list, run_check_data_worker, root_dir,
                        newexps=newexps)
        else:
            run_on_cluster(
                "check_data", sess_list, run_check_data_worker, root_dir,
                n_workers=args.data_check_workers, mem=args.data_check_mem,
                walltime=args.walltime, cluster_log_dir=args.cluster_log_dir,
                newexps=newexps,
            )

        if not args.skip_aggregate:
            stage_build_data_check_df(root_dir, newexps=newexps)
    else:
        print("[skip] stage 1 (data_check)")

    if not args.skip_events:
        data_check_df = pd.read_json(
            join(root_dir, "sess_list_df_data_check.json")
        ).query("include == True")
        sess_list = np.asarray(data_check_df[["sub", "exp", "sess", "loc", "mon"]].values)
        if args.n_subjects is not None:
            sess_list = _slice_for_subjects(sess_list, args.n_subjects)
        elif args.n_sessions is not None:
            sess_list = sess_list[: args.n_sessions]
        n_subj = len(set(row[0] for row in sess_list))
        print(f"[stage 2] get_events on {len(sess_list)} sessions ({n_subj} subjects)")

        if args.local:
            run_locally("get_events", sess_list, run_get_events_worker, root_dir)
        else:
            run_on_cluster(
                "get_events", sess_list, run_get_events_worker, root_dir,
                n_workers=args.events_workers, mem=args.events_mem,
                walltime=args.walltime, cluster_log_dir=args.cluster_log_dir,
            )

        if not args.skip_aggregate:
            stage_build_final_df(root_dir)
    else:
        print("[skip] stage 2 (events)")


if __name__ == "__main__":
    main()