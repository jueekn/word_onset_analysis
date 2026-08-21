"""ptsa_patches.py — project-level monkey-patches for PTSA defects.

Imported at the top of helper.py so any FC pipeline entry point that
imports helper (i.e. every entry point) gets the patches applied early,
before the offending code paths run. Idempotent: re-importing is a no-op.

Currently patches:
  1. `ptsa.data.readers.params.ParamsReader.locate_params_file` — for
     sessions on the explicit verified list in `config/patch_pyFR.csv`,
     force-use the per-recording-session timestamped
     `{dirname}/{basename}.params.txt` file. The verified list is built
     by `sim/build_patch_pyFR.py` which cross-checks each session's
     filename-encoded timestamp against the first event's mstime; only
     sessions where the alignment falls in a plausible setup window are
     listed, so the timestamped file is provably bound to that session
     rather than to some other recording in the same subject directory.

     For sessions NOT in the verified list, the original upstream search
     order (`{dataroot}.params`, then `{dirname}/params.txt`) is left
     untouched — both to avoid accidentally picking up a per-session
     params.txt that wasn't verified, and to preserve the original
     behavior for CML R1xxx subjects that don't have per-session params
     at all (their sources come from `/protocols/.../sources.json`).
"""
from __future__ import annotations

from os.path import abspath, basename, dirname, isfile, join
from pathlib import Path

_APPLIED = False
# Lazily populated set of EEG-file basenames (filename timestamps) approved
# for the timestamped-params override. Set is loaded once at first call
# from config/patch_pyFR.csv.
_VERIFIED_BASENAMES: set[str] | None = None
_VERIFIED_LOCK_TRIED = False


def _load_verified_basenames() -> set[str]:
    """Read config/patch_pyFR.csv (decisions == 'use_timestamped') once.

    Returns the empty set if the CSV is missing — the patch then becomes
    a no-op and the upstream default search order applies everywhere.
    """
    global _VERIFIED_BASENAMES, _VERIFIED_LOCK_TRIED
    if _VERIFIED_BASENAMES is not None:
        return _VERIFIED_BASENAMES
    if _VERIFIED_LOCK_TRIED:
        return set()
    _VERIFIED_LOCK_TRIED = True
    csv_path = Path(__file__).resolve().parent / "config" / "patch_pyFR.csv"
    if not csv_path.exists():
        _VERIFIED_BASENAMES = set()
        return _VERIFIED_BASENAMES
    import csv as _csv
    out: set[str] = set()
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("decision") == "use_timestamped" and row.get("eegfile_basename"):
                out.add(row["eegfile_basename"])
    _VERIFIED_BASENAMES = out
    return out


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _patch_numpy_removed_aliases()
    _patch_ptsa_params_reader()
    _APPLIED = True


def _patch_numpy_removed_aliases() -> None:
    """Restore the numpy scalar aliases PTSA still uses.

    numpy 1.24 REMOVED `np.float` / `np.int` / `np.bool` / `np.object` /
    `np.complex` / `np.str` (deprecated since 1.20). PTSA predates that and
    still references them on some read paths, so any session routed through
    those paths dies with:

        AttributeError: module 'numpy' has no attribute 'float'

    This is the largest single cause of compute-stage failures here -- 69 of the
    errors in the last roi_power run -- and it is not random: it hits the older
    pyFR readers, so it silently removes a SITE-CORRELATED subset of sessions
    (the TJ_pyFR sessions absent from earlier runs). A site-correlated dropout
    is exactly the kind of thing that biases a group comparison, so this is
    worth fixing rather than tolerating.

    Each alias is restored to the builtin it aliased, which is what the
    deprecation notice itself recommends and is behaviour-preserving: `np.float`
    WAS `float`. Guarded with hasattr so a future numpy that reinstates them (or
    an older one that still has them) is untouched.
    """
    import numpy as _np

    for _name, _builtin in (("float", float), ("int", int), ("bool", bool),
                            ("object", object), ("str", str),
                            ("complex", complex)):
        if not hasattr(_np, _name):
            setattr(_np, _name, _builtin)


def _patch_ptsa_params_reader() -> None:
    """Replace ParamsReader.locate_params_file with a version that also
    checks {basename(dataroot)}.params.txt in the dataroot's directory."""
    try:
        from ptsa.data.readers.params import ParamsReader
    except ImportError:
        # PTSA not installed in this env — nothing to patch.
        return

    original = ParamsReader.locate_params_file

    @staticmethod
    def locate_params_file_patched(dataroot):
        # Override only for sessions explicitly verified in patch_pyFR.csv.
        # The verification step in sim/build_patch_pyFR.py confirms the
        # filename timestamp matches the first event's mstime (within a
        # plausible recording-to-event setup window), so the timestamped
        # params.txt is provably bound to this recording. For non-verified
        # sessions we defer to the original upstream search order to
        # avoid silently flipping which params.txt is used.
        verified = _load_verified_basenames()
        base = basename(dataroot)
        if base in verified:
            ts_path = abspath(join(dirname(dataroot), base + ".params.txt"))
            if isfile(ts_path):
                return ts_path
            # Try the eeg.noreref sibling — events.eegfile often points at
            # eeg.reref/ but the params.txt lives in eeg.noreref/.
            alt_dir = dirname(dataroot).replace("eeg.reref", "eeg.noreref")
            alt = abspath(join(alt_dir, base + ".params.txt"))
            if isfile(alt):
                return alt
        # Not on the verified list (or not findable) — defer to upstream.
        return original(dataroot)

    ParamsReader.locate_params_file = locate_params_file_patched


# Auto-apply on import so the calling site doesn't need to remember.
apply()
