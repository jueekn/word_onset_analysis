"""
project_paths.py

Loads `config/config.yaml` and exposes it as module-level constants.

Every entry under the `paths:` block becomes a `pathlib.Path`. Values support
`${VAR}` template expansion (string.Template.safe_substitute) over
`${USER}`, `${REPO_ROOT}` and `${SCRATCH_DIR}` (the resolved `paths.scratch_dir`);
relative paths are joined with the repo root (the directory containing
config/config.yaml). Every other top-level key is read through `get_default`,
which raises on unknown keys -- there is no silent in-code fallback.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from string import Template
from typing import Any, Dict, cast

import yaml

# Repo root = directory containing this file.
REPO_ROOT: Path = Path(__file__).resolve().parent
_CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        loaded: Any = yaml.safe_load(f)
    return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else {}


_CFG: dict[str, Any] = _safe_load_yaml(_CONFIG_PATH)


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_paths(raw: Dict[str, Any]) -> Dict[str, Path]:
    """Two-pass ${VAR} expansion of the `paths:` block into absolute Paths."""
    base = {"USER": os.environ.get("USER") or getpass.getuser(), "REPO_ROOT": str(REPO_ROOT)}
    if "scratch_dir" not in raw:
        raise KeyError("config/config.yaml `paths:` block is missing `scratch_dir`")
    scratch = Template(str(raw["scratch_dir"])).safe_substitute(base)
    base["SCRATCH_DIR"] = str(_abs(scratch))
    return {k: _abs(Template(str(v)).safe_substitute(base)) for k, v in raw.items()}


_PATHS: Dict[str, Path] = _resolve_paths(cast(dict[str, Any], _CFG.get("paths") or {}))


def get(key: str) -> Path:
    """Look up a `paths:` entry by key."""
    if key not in _PATHS:
        raise KeyError(f"Unknown path key {key!r}; available: {sorted(_PATHS)}")
    return _PATHS[key]


def get_default(key: str) -> Any:
    """A top-level config value, for `default=` in argparse and module constants."""
    if key not in _CFG or key == "paths":
        raise KeyError(f"Unknown config key {key!r}; available: {sorted(k for k in _CFG if k != 'paths')}")
    return _CFG[key]


# ---------------------------------------------------------------------------
# Spectral bands — single project-wide source for all FC compute paths.
#     from project_paths import BANDS
#     fmin, fmax = BANDS[band]
# ---------------------------------------------------------------------------
BANDS: dict[str, tuple[float, float]] = {
    str(name): (float(lo), float(hi)) for name, (lo, hi) in get_default("bands").items()}


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# Metrics permanently retired from the pipeline. `dpli` was never one of the
# thesis measures (exploratory only); banning it here makes any reintroduction
# a hard error at import time rather than silently flowing through.
BANNED_METRICS: frozenset[str] = frozenset({"dpli"})
COMPUTATION_METRICS: list[str] = list(get_default("computation_metrics"))
if BANNED_METRICS.intersection(COMPUTATION_METRICS):
    raise ValueError(
        f"config.yaml computation_metrics lists permanently-retired metric(s) "
        f"{sorted(BANNED_METRICS.intersection(COMPUTATION_METRICS))}; remove them.")


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
# `behaviors_main` are the contrastive behaviors that drive group-level analyses
# (en, rm, word_on, voc); `behaviors_noncontrast` the all-events robustness
# variants (en_all, rm_all).
BEHAVIORS_MAIN: tuple[str, ...] = tuple(get_default("behaviors_main"))
BEHAVIORS_NONCONTRAST: tuple[str, ...] = tuple(get_default("behaviors_noncontrast"))
BEHAVIORS_ALL: tuple[str, ...] = BEHAVIORS_MAIN + BEHAVIORS_NONCONTRAST


SCRATCH_DIR: Path = get("scratch_dir")
# Event/session exclusion-count logs + the config name stamped into them.
EXCLUSION_COUNTS_DIR: Path = get("exclusion_counts")
CONFIG_NAME: str = str(get_default("config_name"))
# Curated tabular data shipped with the repo.
UNRECOVERABLE_SESSIONS_CSV: Path = get("unrecoverable_sessions_csv")
