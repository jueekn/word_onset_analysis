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
    suffix = ((".longetal" if os.environ.get("WOA_LONGETAL") else "")      # both set by the Snakefile
             + (".smokescreen" if os.environ.get("WOA_SMOKESCREEN") else ""))
    scratch = Template(str(raw["scratch_dir"])).safe_substitute(base) + suffix   # ${SCRATCH_DIR} users follow
    base["SCRATCH_DIR"] = str(_abs(scratch))
    out = {k: _abs(Template(str(v)).safe_substitute(base)) for k, v in raw.items()}
    for k in ("scratch_dir", "exclusion_counts"):
        if k in out:
            out[k] = Path(str(out[k]) + suffix)
    return out


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
COMPUTATION_METRICS: list[str] = list(get_default("computation_metrics"))
SUBTRACT_ERP: bool = bool(get_default("subtract_erp"))
# synchrony metric of each band's run (config `runs`), for figure assembly
RUN_METRIC: dict[str, str] = {r["band"]: r["metric"] for r in get_default("runs")}
REAL_DATA_BUFFER_MS: float = float(get_default("real_data_buffer_ms"))
# Long et al. 2020 replication settings (config `longetal_params:`), or None.
LONGETAL: dict[str, Any] | None = get_default("longetal_params") if os.environ.get("WOA_LONGETAL") else None
# None = keep the native rate (longetal: Hilbert at the native rate).
RESAMPLE_HZ: float | None = None if LONGETAL else float(get_default("resample_hz"))
MIN_SAMPLE_RATE_HZ: float = float(get_default("min_sample_rate_hz"))
MAX_LINE_HARMONIC_RATIO: float = float(get_default("max_line_harmonic_ratio"))
# notch_harmonics_up_to_hz: None = fundamental-only notch (historical default,
# leaves every existing result bit-identical). A float switches on harmonic
# notching up to that frequency — required for bands that reach the line
# harmonics, e.g. high_gamma [70,150]. See config.yaml for the full rationale.
_notch_harm_raw = get_default("notch_harmonics_up_to_hz")
NOTCH_HARMONICS_UP_TO_HZ: float | None = (
    None if _notch_harm_raw is None else float(_notch_harm_raw))
# pair_distance_threshold_mm: scientific bipolar-pair distance cutoff (data_check).
PAIR_DISTANCE_THRESHOLD_MM: float = float(get_default("pair_distance_threshold_mm"))
_mt_bw_raw = get_default("mt_bandwidth")
MT_BANDWIDTH: float | None = None if _mt_bw_raw is None else float(_mt_bw_raw)
# Power time course (gamma only): bin step and sliding multitaper window.
TIME_BIN_MS: int = int(get_default("time_bin_ms"))
MT_WINDOW_MS: int = int(get_default("mt_window_ms"))


SCRATCH_DIR: Path = get("scratch_dir")
# cml_data reads the download cache location from CML_BIDS_CACHE.
os.environ.setdefault("CML_BIDS_CACHE", str(get("bids_cache")))
# Event/session exclusion-count logs + the config name stamped into them.
EXCLUSION_COUNTS_DIR: Path = get("exclusion_counts")
CONFIG_NAME: str = str(get_default("config_name"))
# Curated tabular data shipped with the repo.
UNRECOVERABLE_SESSIONS_CSV: Path = get("unrecoverable_sessions_csv")
