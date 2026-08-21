"""
project_paths.py

Centralized output path resolution for the fc_methods_comparison_cml project.

Loads `config/config.yaml` and exposes every entry under the `paths:` block as
a module-level `pathlib.Path` constant. The YAML supports `${VAR}` template
expansion (string.Template.safe_substitute) over a small set of bootstrap vars:

    ${USER}                  resolved from $USER env, falls back to getpass
    ${SCRATCH_DIR}           resolved from paths.scratch_dir (after ${USER})
    ${RESULTS_DIR}           resolved from paths.results_dir
    ${PROCESSED_RESULTS_DIR} resolved from paths.processed_results_dir

Expansion runs in two passes:
    1. Resolve the four bootstrap vars above against $USER and the
       results_dir / processed_results_dir / scratch_dir entries (themselves
       relative to repo root, ${USER}-expanded only).
    2. Expand every other entry against the bootstrap map.

Relative paths (e.g. "results", "processed_results") are made absolute by
joining with the repo root (the directory containing config/config.yaml).

Usage:
    from project_paths import (
        ALLFCMATS, BEHCORRS, CONNECTION_CONVERGENCE, HUB_CONVERGENCE,
        HUB_IOU_FIGS, MEASURECORRS, MULTI_METRIC_DETECTION, SELFIOU,
        SPLITHALF, SPLITHALF_HUBS, SPLITHALF_METRIC_CORR,
        SUBSAMPLESIMILARITY, PAIRWISE_IOU, LISTSPLIT,
        INTERMEDIATE_SUBJ_MAT,
        SCRATCH_DIR,
        RESULTS_DIR,
        PROCESSED_RESULTS_DIR,
        get,
    )
"""

from __future__ import annotations

import getpass
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, cast

import yaml


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dict[str, Any]. yaml.safe_load returns Any; we
    isinstance-narrow at this single boundary so the rest of the module
    type-checks cleanly without scattering Any everywhere. Returns {} for
    empty files or non-mapping top-level values.
    """
    with open(path) as f:
        loaded: Any = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        return {}
    return cast("dict[str, Any]", loaded)


# Smokescreen-mode routing. When active, each of the 3 parent paths
# (results_dir, processed_results_dir, scratch_dir) has `/.smokescreen`
# appended before the rest of config.yaml's Template expansion runs. All
# downstream subdirs (measurecorrs, subj_mat, fc_mats, ...) follow via
# `${RESULTS_DIR}` / `${PROCESSED_RESULTS_DIR}` / `${SCRATCH_DIR}` expansion,
# so a smokescreen run cannot overwrite full-run outputs and `rm -rf */.smokescreen`
# wipes only the smokescreen artifacts.
#
# Activated by either:
#   - FC_SMOKESCREEN env var ∈ {1, true, yes} (case-insensitive); or
#   - "--smokescreen" present in sys.argv (auto-detected for scripts that
#     accept the CLI flag — no need to manually set the env var).
SMOKESCREEN_DIRNAME = ".smokescreen"


def _is_smokescreen_active() -> bool:
    if os.environ.get("FC_SMOKESCREEN", "0").lower() in ("1", "true", "yes"):
        return True
    if "--smokescreen" in sys.argv:
        return True
    return False


# New-experiment-cohort routing. Mirrors the smokescreen reroute: when active,
# each of the 3 parent paths gets `/<NEWEXPS_DIRNAME>` appended before the rest
# of config.yaml's Template expansion, so the entire pipeline (prepare_sessions,
# build_subj_mat, every plot_*) reads/writes a separate tree from the default
# FR1/catFR1/pyFR cohort with no per-script changes. Combine with smokescreen to
# get `.../newexps_icat_ifr_repfr/.smokescreen`.
#
# Activated by either:
#   - FC_NEWEXPS env var ∈ {1, true, yes} (case-insensitive) — use this for the
#     downstream scripts, which have no --newexps flag; or
#   - "--newexps" present in sys.argv (auto-detected for prepare_sessions).
NEWEXPS_DIRNAME = "newexps_icat_ifr_repfr"


def _is_newexps_active() -> bool:
    if os.environ.get("FC_NEWEXPS", "0").lower() in ("1", "true", "yes"):
        return True
    if "--newexps" in sys.argv:
        return True
    return False

# Repo root = directory containing this file.
REPO_ROOT: Path = Path(__file__).resolve().parent
_CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"


def _user() -> str:
    return os.environ.get("USER") or getpass.getuser()


def _abs(p: str) -> Path:
    """Make `p` absolute. If relative, join with REPO_ROOT."""
    path = Path(p)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_raw() -> Dict[str, str]:
    """Read the `paths:` block of config.yaml as a flat dict[str, str].

    Values are coerced to str at load time so downstream Template expansion
    has a uniform input type, regardless of whether the YAML parsed an
    entry as int / float / bool.
    """
    cfg = _safe_load_yaml(_CONFIG_PATH)
    paths_block = cast(dict[str, Any], cfg.get("paths") or {})
    return {k: str(v) for k, v in paths_block.items()}


_REQUIRED_BOOTSTRAP_KEYS = ("scratch_dir", "results_dir", "processed_results_dir")


def _resolve(raw: Dict[str, str]) -> Dict[str, Path]:
    """Two-pass ${VAR} expansion. Returns dict of absolute Path objects.

    Misconfiguration is a hard error: every key referenced by the bootstrap
    map (and every key fetched via `get(...)`) must be present in
    config/config.yaml. We do not silently fall back to in-code defaults,
    because a missing entry usually means the YAML drifted from the code
    that consumes it.
    """
    user = _user()

    # Pass 1: expand ${USER} in scratch_dir, results_dir, processed_results_dir.
    bootstrap_src = {
        "USER": user,
    }
    pass1: Dict[str, str] = {}
    for k, v in raw.items():
        pass1[k] = Template(v).safe_substitute(bootstrap_src)

    missing = [k for k in _REQUIRED_BOOTSTRAP_KEYS if k not in pass1]
    if missing:
        raise KeyError(
            f"config/config.yaml `paths:` block is missing required key(s): "
            f"{missing}. These are bootstrap vars referenced by other entries."
        )

    # Cohort/smokescreen reroutes: append a subdir to each of the 3 parents
    # BEFORE building the bootstrap map for pass 2, so every downstream subdir
    # automatically inherits the reroute via ${SCRATCH_DIR} / ${RESULTS_DIR} /
    # ${PROCESSED_RESULTS_DIR} expansion. newexps is applied first so a
    # smokescreen newexps run nests as `.../newexps_icat_ifr_repfr/.smokescreen`.
    if _is_newexps_active():
        for k in _REQUIRED_BOOTSTRAP_KEYS:
            pass1[k] = f"{pass1[k]}/{NEWEXPS_DIRNAME}"
    if _is_smokescreen_active():
        for k in _REQUIRED_BOOTSTRAP_KEYS:
            pass1[k] = f"{pass1[k]}/{SMOKESCREEN_DIRNAME}"

    # Build the bootstrap map for pass 2: keys are uppercase aliases.
    bootstrap = {
        "USER": user,
        "RESULTS_DIR": str(_abs(pass1["results_dir"])),
        "PROCESSED_RESULTS_DIR": str(_abs(pass1["processed_results_dir"])),
        "SCRATCH_DIR": pass1["scratch_dir"],
    }

    # Pass 2: expand every entry against the bootstrap map.
    resolved: Dict[str, Path] = {}
    for k, v in pass1.items():
        expanded = Template(v).safe_substitute(bootstrap)
        resolved[k] = _abs(expanded)
    return resolved


_PATHS: Dict[str, Path] = _resolve(_load_raw())


def get(key: str) -> Path:
    """Look up a path entry by key (e.g. 'between_metric_comparisons')."""
    if key not in _PATHS:
        raise KeyError(
            f"Unknown path key {key!r}; available: {sorted(_PATHS.keys())}"
        )
    return _PATHS[key]


# ---------------------------------------------------------------------------
# Random-state plumbing
# ---------------------------------------------------------------------------

def _load_seed_offset() -> int:
    """Read the project-wide RNG offset from config.yaml.

    Required top-level integer key. Raises KeyError if absent — there is no
    in-code default because a missing offset usually means the YAML drifted
    out of sync with the code that consumes it.
    """
    cfg = _safe_load_yaml(_CONFIG_PATH)
    if "seed_offset" not in cfg:
        raise KeyError(
            "config/config.yaml is missing required top-level key 'seed_offset' "
            "(integer; project-wide RNG offset)."
        )
    return int(cfg["seed_offset"])


SEED_OFFSET: int = _load_seed_offset()


# ---------------------------------------------------------------------------
# Argparse-friendly defaults (Smin, band, cond, n_resamples, Ns_*, ...)
# ---------------------------------------------------------------------------

def _load_params() -> dict[str, Any]:
    """Top-level config values (everything except the `paths:` block).

    Newexps overrides (`newexps_overrides:`) are merged when `FC_NEWEXPS` is
    set, then smokescreen overrides (`smokescreen_overrides:`) when
    `FC_SMOKESCREEN` is set — same ordering as the path reroute (newexps first,
    so a combined run still collapses to smokescreen's tiny values).
    """
    cfg = _safe_load_yaml(_CONFIG_PATH)
    params: dict[str, Any] = {str(k): v for k, v in cfg.items() if k != "paths"}
    if _is_newexps_active():
        overrides = cast(dict[str, Any], cfg.get("newexps_overrides") or {})
        for k, v in overrides.items():
            params[str(k)] = v
    if _is_smokescreen_active():
        overrides = cast(dict[str, Any], cfg.get("smokescreen_overrides") or {})
        for k, v in overrides.items():
            params[str(k)] = v
    return params


_PARAMS: dict[str, Any] = _load_params()


IS_SMOKESCREEN: bool = _is_smokescreen_active()


IS_NEWEXPS: bool = _is_newexps_active()


def get_default(key: str) -> Any:
    """Return an argparse-friendly default from config.yaml.

    Use for `default=` in `argparse.add_argument` so all scripts share one
    source of truth (Smin, band, cond, n_resamples, Ns_similarity, ...).
    Raises KeyError on unknown keys — there is no silent fallback.
    """
    if key not in _PARAMS:
        raise KeyError(
            f"Unknown config key {key!r}; available: {sorted(_PARAMS.keys())}"
        )
    return _PARAMS[key]


def get_compute_profile(name: str) -> dict[str, Any]:
    """SLURM resource defaults for one of the major analysis clusters.

    `name` is ``'fc_matrix'`` (heavy Stage-A/B FC computation) or
    ``'downstream'`` (light resampling / aggregation compute_*/plot_* jobs).
    Returns a dict with ``n_workers``, ``mem``, ``walltime`` from
    config.yaml:compute_profiles, so every script's argparse defaults share one
    source of truth. Raises KeyError on an unknown profile."""
    profiles = cast("dict[str, Any]", get_default("compute_profiles"))
    if name not in profiles:
        raise KeyError(
            f"Unknown compute profile {name!r}; available: {sorted(profiles)}"
        )
    return cast("dict[str, Any]", profiles[name])


# ---------------------------------------------------------------------------
# SLURM broken-node exclusion — single project-wide knob. Edit config.yaml
# `cluster.excluded_nodes` ONCE to add/remove nodes; it propagates to every
# in-code cmldask dispatch via slurm_exclude_directives(), to scripts/env.sh
# (SBATCH_EXCLUDE), and to the Snakefile shell.prefix. Empty list = no exclusion.
# ---------------------------------------------------------------------------

def _load_excluded_nodes() -> list[str]:
    cluster = cast("dict[str, Any]", get_default("cluster"))
    nodes = cast("list[Any]", cluster.get("excluded_nodes") or [])
    return [str(n).strip() for n in nodes if n is not None and str(n).strip()]


EXCLUDED_NODES: list[str] = _load_excluded_nodes()


def sbatch_exclude_value() -> str:
    """Comma-joined excluded-node list for the ``SBATCH_EXCLUDE`` env var or an
    ``sbatch --exclude=`` argument. Empty string when nothing is excluded.
    Single source: config.yaml ``cluster.excluded_nodes``."""
    return ",".join(EXCLUDED_NODES)


def slurm_exclude_directives() -> list[str]:
    """``#SBATCH``-style directives to splice into a cmldask
    ``job_extra_directives=[...]`` list so workers avoid the configured broken
    nodes. Returns ``["--exclude=node18,node37,node47"]`` (one element) or ``[]``
    when ``cluster.excluded_nodes`` is empty."""
    value = sbatch_exclude_value()
    return [f"--exclude={value}"] if value else []


# ---------------------------------------------------------------------------
# Spectral bands — single project-wide source for all FC compute paths.
# Loaded as tuples once at import time. Consumers import BANDS directly:
#     from project_paths import BANDS
#     fmin, fmax = BANDS[band]
# ---------------------------------------------------------------------------

def _load_bands() -> dict[str, tuple[float, float]]:
    raw = cast("dict[str, Any]", get_default("bands"))
    return {str(name): (float(lo), float(hi)) for name, (lo, hi) in raw.items()}


BANDS: dict[str, tuple[float, float]] = _load_bands()


# ---------------------------------------------------------------------------
# Behaviors (data) — per-beh dict {cond, event_window, metrics}.
# Display strings (title) live in figure_io.BEHAVIOR_DISPLAY.
# ---------------------------------------------------------------------------

def _load_behaviors_cfg() -> dict[str, dict[str, Any]]:
    raw = cast("dict[str, Any]", get_default("behaviors"))
    out: dict[str, dict[str, Any]] = {}
    for beh, entry in raw.items():
        e = cast("dict[str, Any]", entry)
        out[str(beh)] = {
            "cond": str(e["cond"]),
            "event_window": (int(e["event_window"][0]), int(e["event_window"][1])),
            "metrics": list(e["metrics"]),
        }
    return out


BEHAVIORS_CFG: dict[str, dict[str, Any]] = _load_behaviors_cfg()


# ---------------------------------------------------------------------------
# Metric registry — split by purpose (compute vs plot vs slow-default-skip).
# ---------------------------------------------------------------------------

# Metrics permanently retired from the pipeline. `dpli` was never one of the
# thesis measures (exploratory only); banning it here makes any reintroduction
# a hard error at config-load time — i.e. at the very start of any snakemake
# run or script import — rather than silently flowing through as exploratory
# output. To retire another metric, add its name here.
BANNED_METRICS: frozenset[str] = frozenset({"dpli"})


def _reject_banned_metrics() -> None:
    """Fail loudly if a retired metric (BANNED_METRICS) appears in any configured
    metric list. Raised at import time so the guard fires before any compute."""
    sources = {
        "computation_metrics": get_default("computation_metrics"),
        "plotting_metrics": get_default("plotting_metrics"),
        "slow_metrics": get_default("slow_metrics"),
        "smokescreen_fast_metrics": get_default("smokescreen_fast_metrics"),
    }
    for beh, entry in BEHAVIORS_CFG.items():
        sources[f"behaviors.{beh}.metrics"] = entry["metrics"]
    offenders = {
        key: sorted(BANNED_METRICS.intersection(metrics))
        for key, metrics in sources.items()
        if BANNED_METRICS.intersection(metrics)
    }
    if offenders:
        raise ValueError(
            "config.yaml lists permanently-retired metric(s) "
            f"{sorted(BANNED_METRICS)}: {offenders}. These were never thesis "
            "measures and must not be reintroduced — remove them from the "
            "offending metric list(s) in config.yaml."
        )


_reject_banned_metrics()

COMPUTATION_METRICS: list[str] = list(get_default("computation_metrics"))
PLOTTING_METRICS: list[str] = list(get_default("plotting_metrics"))
SLOW_METRICS: tuple[str, ...] = tuple(get_default("slow_metrics"))


# ---------------------------------------------------------------------------
# Case-study connections / hubs (per behavior). compute_connection_convergence
# and compute_hub_convergence read defaults from here so standalone CLI runs
# match the Snakefile-orchestrated parameters.
# ---------------------------------------------------------------------------

def _load_case_study() -> dict[str, dict[str, Any]]:
    raw = cast("dict[str, Any]", get_default("case_study"))
    out: dict[str, dict[str, Any]] = {}
    for beh, entry in raw.items():
        e = cast("dict[str, Any]", entry)
        out[str(beh)] = {
            "connection_pair": tuple(e["connection_pair"]),
            "hub_region": str(e["hub_region"]),
        }
    return out


CASE_STUDY: dict[str, dict[str, Any]] = _load_case_study()


@dataclass(frozen=True)
class CaseStudyConnection:
    """One named connection case study.

    `region_pairs` is the list of ordered (region_a, region_b) pairs whose
    channel-level connections are pooled into a single connectivity value. A
    plain `pair` entry yields one region-pair; a `pool` entry yields several.
    `is_pool` is True iff more than one region-pair contributes.
    """

    name: str
    pretty_name: str
    region_pairs: tuple[tuple[str, str], ...]
    beh: str

    @property
    def is_pool(self) -> bool:
        return len(self.region_pairs) > 1


def _load_case_study_connections() -> dict[str, list[CaseStudyConnection]]:
    """Parse config.case_study[beh].connections into typed case studies.

    Behaviors without a `connections:` block fall back to the legacy
    `connection_pair` single edge so existing callers keep a sensible default.
    """
    raw = cast("dict[str, Any]", get_default("case_study"))
    out: dict[str, list[CaseStudyConnection]] = {}
    for beh, entry in raw.items():
        e = cast("dict[str, Any]", entry)
        cases: list[CaseStudyConnection] = []
        conns = e.get("connections")
        if conns is None:
            # Legacy fallback: a single unnamed case from connection_pair.
            pair = tuple(str(r) for r in e["connection_pair"])
            cases.append(CaseStudyConnection(
                name="default", pretty_name=f"{pair[0]} – {pair[1]}",
                region_pairs=(cast("tuple[str, str]", pair),), beh=str(beh),
            ))
        else:
            for c in cast("list[dict[str, Any]]", conns):
                if ("pair" in c) == ("pool" in c):
                    raise ValueError(
                        f"case_study[{beh}].connections entry {c.get('name')!r} "
                        "must have exactly one of `pair` or `pool`"
                    )
                if "pair" in c:
                    rps: tuple[tuple[str, str], ...] = (
                        (str(c["pair"][0]), str(c["pair"][1])),
                    )
                else:
                    rps = tuple(
                        (str(p[0]), str(p[1])) for p in c["pool"]
                    )
                cases.append(CaseStudyConnection(
                    name=str(c["name"]),
                    pretty_name=str(c["pretty_name"]),
                    region_pairs=rps,
                    beh=str(beh),
                ))
        out[str(beh)] = cases
    return out


CASE_STUDY_CONNECTIONS: dict[str, list[CaseStudyConnection]] = \
    _load_case_study_connections()


def _load_case_study_hubs() -> dict[str, list[str]]:
    """Per-behavior list of case-study hub regions.

    Falls back to the legacy single `hub_region` when a behavior has no `hubs`
    list. compute_hub_convergence still runs one hub region per invocation; this
    registry just enumerates the hubs of interest for each behavioral contrast.
    """
    raw = cast("dict[str, Any]", get_default("case_study"))
    out: dict[str, list[str]] = {}
    for beh, entry in raw.items():
        e = cast("dict[str, Any]", entry)
        hubs = e.get("hubs")
        out[str(beh)] = (
            [str(h) for h in hubs] if hubs is not None
            else [str(e["hub_region"])]
        )
    return out


CASE_STUDY_HUBS: dict[str, list[str]] = _load_case_study_hubs()


@dataclass(frozen=True)
class CaseStudyHub:
    """One named hub case study.

    `regions` is the list of subregions whose pool forms the hub: one region
    for a plain single-region hub, several for a pooled (e.g. bilateral) hub.
    `is_pool` is True iff more than one subregion contributes.
    """

    name: str
    pretty_name: str
    regions: tuple[str, ...]
    beh: str

    @property
    def is_pool(self) -> bool:
        return len(self.regions) > 1


def _load_case_study_hub_cases() -> dict[str, list[CaseStudyHub]]:
    """Parse config.case_study[beh].hub_cases into typed hub case studies.

    Behaviors without a `hub_cases` block fall back to the legacy single
    `hub_region`. Each entry has exactly one of `region` (single) or `pool`.
    """
    raw = cast("dict[str, Any]", get_default("case_study"))
    out: dict[str, list[CaseStudyHub]] = {}
    for beh, entry in raw.items():
        e = cast("dict[str, Any]", entry)
        cases: list[CaseStudyHub] = []
        hub_cases = e.get("hub_cases")
        if hub_cases is None:
            region = str(e["hub_region"])
            cases.append(CaseStudyHub(
                name="default", pretty_name=region,
                regions=(region,), beh=str(beh),
            ))
        else:
            for c in cast("list[dict[str, Any]]", hub_cases):
                if ("region" in c) == ("pool" in c):
                    raise ValueError(
                        f"case_study[{beh}].hub_cases entry {c.get('name')!r} "
                        "must have exactly one of `region` or `pool`"
                    )
                regions = ((str(c["region"]),) if "region" in c
                           else tuple(str(r) for r in c["pool"]))
                cases.append(CaseStudyHub(
                    name=str(c["name"]), pretty_name=str(c["pretty_name"]),
                    regions=regions, beh=str(beh),
                ))
        out[str(beh)] = cases
    return out


CASE_STUDY_HUB_CASES: dict[str, list[CaseStudyHub]] = _load_case_study_hub_cases()


def case_study_hub(beh: str, name: str) -> CaseStudyHub:
    """Look up one named hub case study by (behavior, name)."""
    for c in CASE_STUDY_HUB_CASES.get(beh, []):
        if c.name == name:
            return c
    avail = [c.name for c in CASE_STUDY_HUB_CASES.get(beh, [])]
    raise KeyError(
        f"no case_study hub named {name!r} for beh={beh!r}; available: {avail}"
    )


def case_study_connection(beh: str, name: str) -> CaseStudyConnection:
    """Look up one named connection case study by (behavior, name)."""
    for c in CASE_STUDY_CONNECTIONS.get(beh, []):
        if c.name == name:
            return c
    avail = [c.name for c in CASE_STUDY_CONNECTIONS.get(beh, [])]
    raise KeyError(
        f"no case_study connection named {name!r} for beh={beh!r}; "
        f"available: {avail}"
    )


# ---------------------------------------------------------------------------
# Signal-processing constants (config-flow only; no behavior change at
# migration time). mt_bandwidth=None keeps MNE's default — code_issues #79.
# ---------------------------------------------------------------------------

MIRROR_BUFFER_MS: int = int(get_default("mirror_buffer_ms"))
REAL_DATA_BUFFER_MS: float = float(get_default("real_data_buffer_ms"))
RESAMPLE_HZ: float = float(get_default("resample_hz"))
MIN_SAMPLE_RATE_HZ: float = float(get_default("min_sample_rate_hz"))
# notch_harmonics_up_to_hz: None = fundamental-only notch (historical default,
# leaves every existing result bit-identical). A float switches on harmonic
# notching up to that frequency — required for bands that reach the line
# harmonics, e.g. high_gamma [70,150]. See config.yaml for the full rationale.
_notch_harm_raw = get_default("notch_harmonics_up_to_hz")
NOTCH_HARMONICS_UP_TO_HZ: float | None = (
    None if _notch_harm_raw is None else float(_notch_harm_raw))
# pair_distance_threshold_mm: scientific bipolar-pair distance cutoff (data_check).
PAIR_DISTANCE_THRESHOLD_MM: float = float(get_default("pair_distance_threshold_mm"))
GC_N_LAGS: int = int(get_default("gc_n_lags"))
_mt_bw_raw = get_default("mt_bandwidth")
MT_BANDWIDTH: float | None = None if _mt_bw_raw is None else float(_mt_bw_raw)
# Morlet (cwt_morlet) FC mode. FC_MODE selects the estimator for the phase
# metrics; the CWT_* values configure the wavelet bank and its edge buffer.
# See config.yaml for the full rationale on each.
FC_MODE: str = str(get_default("fc_mode"))
CWT_FNUM: int = int(get_default("cwt_fnum"))
CWT_MORLET_REPS: int = int(get_default("cwt_morlet_reps"))
CWT_BUFFER_N_SIGMA: float = float(get_default("cwt_buffer_n_sigma"))
# time_bin_ms: latency-axis bin width for the Morlet path (ignored by multitaper).
TIME_BIN_MS: int = int(get_default("time_bin_ms"))
# mt_window_ms: sliding-window length for the windowed multitaper latency axis.
MT_WINDOW_MS: int = int(get_default("mt_window_ms"))


# ---------------------------------------------------------------------------
# Behavior (task-contrast) registry
# ---------------------------------------------------------------------------

def _load_behaviors() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load the two behavior lists from config.yaml.

    `behaviors_main` are the contrastive behaviors that drive group-level
    analyses by default (en, rm, word_on, voc). `behaviors_noncontrast`
    are the all-events robustness variants (en_all, rm_all) that are only
    run when explicitly requested.
    """
    cfg = _safe_load_yaml(_CONFIG_PATH)
    missing = [k for k in ("behaviors_main", "behaviors_noncontrast") if k not in cfg]
    if missing:
        raise KeyError(
            f"config/config.yaml is missing required top-level keys: {missing}"
        )
    main = cast("list[str]", cfg["behaviors_main"])
    nc = cast("list[str]", cfg["behaviors_noncontrast"])
    return tuple(main), tuple(nc)


BEHAVIORS_MAIN, BEHAVIORS_NONCONTRAST = _load_behaviors()
BEHAVIORS_ALL: tuple[str, ...] = BEHAVIORS_MAIN + BEHAVIORS_NONCONTRAST


def behavior_contrast_mask(supported: Iterable[str]) -> list[str]:
    """Intersect a script's locally-supported behaviors with the global default.

    Returns the elements of ``supported`` that are also in ``BEHAVIORS_MAIN``,
    preserving the order of ``supported``. This is the argparse default for
    any plot script: each script keeps its own list of supported behaviors
    (e.g. ``BEH_CONFIG.keys()``), and this helper trims it down to what
    config.yaml has currently enabled by default.
    """
    main_set = set(BEHAVIORS_MAIN)
    return [b for b in supported if b in main_set]


def seed_for(name: str) -> int:
    """Deterministic per-analysis seed.

    Computed as ``(sha256(name) + SEED_OFFSET) mod 2**32``. Changing
    ``seed_offset`` in config.yaml rerolls every analysis to an independent
    draw; the sha256 hash spreads collisions evenly so different analyses
    sharing the same offset don't collide.
    """
    import hashlib
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    h = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return (h + SEED_OFFSET) % (2 ** 32)


# Module-level Path constants. Use the keys we know exist from config.yaml.
RESULTS_DIR: Path = get("results_dir")
PROCESSED_RESULTS_DIR: Path = get("processed_results_dir")
SCRATCH_DIR: Path = get("scratch_dir")

SIM_TMP_DIR: Path = get("sim_tmp_dir")
LOGS_DIR: Path = get("logs_dir")
CLUSTER_LOGS_DIR: Path = get("cluster_logs_dir")

# Event/session exclusion-count logs + the config name stamped into them.
EXCLUSION_COUNTS_DIR: Path = get("exclusion_counts")
CONFIG_NAME: str = str(get_default("config_name"))

# Per-script result subdirs (collaborator's argparse defaults).
ALLFCMATS: Path = get("allfcmats")
BEHCORRS: Path = get("behcorrs")
CONNECTION_CONVERGENCE: Path = get("connection_convergence")
HUB_CONVERGENCE: Path = get("hub_convergence")
HUB_IOU_FIGS: Path = get("hub_iou_figs")
MEAN_CONNECTIVITY: Path = get("mean_connectivity")
MEASURECORRS: Path = get("measurecorrs")
MULTI_METRIC_DETECTION: Path = get("multi_metric_detection")
SELFIOU: Path = get("selfiou")
SPLITHALF: Path = get("retest_reliability")
SPLITHALF_HUBS: Path = get("retest_hubs")
SPLITHALF_METRIC_CORR: Path = get("retest_reliability_metric_corr")
ICC_RELIABILITY: Path = get("icc_reliability")
HEDGES_G_RELIABILITY: Path = get("hedges_g_reliability")
ICC_RELIABILITY_HUB: Path = get("icc_reliability_hub")
HEDGES_G_RELIABILITY_HUB: Path = get("hedges_g_reliability_hub")
SUBSAMPLESIMILARITY: Path = get("subsamplesimilarity")
SUBSAMPLESIMILARITY_HUB: Path = get("subsamplesimilarity_hub")
PAIRWISE_IOU: Path = get("pairwise_iou")
LISTSPLIT: Path = get("listsplit")

# Intermediate caches.
INTERMEDIATE_SUBJ_MAT: Path = get("intermediate_subj_mat")
INTERMEDIATE_SESS_MAT: Path = get("intermediate_sess_mat")
INTERMEDIATE_SPLITHALF: Path = get("intermediate_retest_reliability")
INTERMEDIATE_ICC: Path = get("intermediate_icc_reliability")
INTERMEDIATE_HEDGES_G: Path = get("intermediate_hedges_g_reliability")
INTERMEDIATE_ICC_HUB: Path = get("intermediate_icc_reliability_hub")
INTERMEDIATE_HEDGES_G_HUB: Path = get("intermediate_hedges_g_reliability_hub")
INTERMEDIATE_IOU_CURVE: Path = get("intermediate_iou_curve")
INTERMEDIATE_SIMILARITY: Path = get("intermediate_similarity")
INTERMEDIATE_HUB_IOU: Path = get("intermediate_hub_iou")
INTERMEDIATE_AVG_REGION: Path = get("intermediate_avg_region")
INTERMEDIATE_HUB_SPLITHALF: Path = get("intermediate_hub_retest_reliability")
INTERMEDIATE_PAIRWISE_IOU: Path = get("intermediate_pairwise_iou")
INTERMEDIATE_CASE_STUDY_POOLED: Path = get("intermediate_case_study_pooled")
INTERMEDIATE_SESS_MAT_CHANNEL: Path = get("intermediate_sess_mat_channel")
INTERMEDIATE_SPLITHALF_CHANNEL: Path = get("intermediate_retest_channel")
INTERMEDIATE_HUB_SPLITHALF_CHANNEL: Path = get("intermediate_hub_retest_channel")

# Standardized statistical-result summary JSONs (results_io.py).
RESULT_SUMMARY_DIR: Path = get("result_summary")

# Build-tracking artifacts.
SNAKEMAKE_SENTINELS: Path = get("snakemake_sentinels")

# Curated tabular data shipped with the repo.
UNRECOVERABLE_SESSIONS_CSV: Path = get("unrecoverable_sessions_csv")
REPFR1_PRESENTATION_TIMES_CSV: Path = get("repfr1_presentation_times_csv")


# ---------------------------------------------------------------------------
# Python environment path
# ---------------------------------------------------------------------------

def _load_python_env_path() -> Path:
    """Resolve the canonical python env path from config.yaml.

    The value supports two template vars: ``${REPO_ROOT}`` (this file's
    parent) and ``${USER}``. Raises KeyError if the entry is missing — the
    pipeline cannot bootstrap without a python interpreter, so silent
    fallback is more dangerous than a clear error.
    """
    cfg = _safe_load_yaml(_CONFIG_PATH)
    if "python_env_path" not in cfg:
        raise KeyError(
            "config/config.yaml is missing required top-level key "
            "'python_env_path' (absolute path to the project's python env)."
        )
    raw = str(cfg["python_env_path"])
    expanded = Template(raw).safe_substitute({
        "REPO_ROOT": str(REPO_ROOT),
        "USER": _user(),
    })
    return Path(expanded)


PYTHON_ENV_PATH: Path = _load_python_env_path()
