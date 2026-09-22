"""ptsa_patches.py — project-level monkey-patches for PTSA defects.

Imported at the top of helper.py so any pipeline entry point that imports
helper (i.e. every entry point) gets the patches applied early, before the
offending code paths run. Idempotent: re-importing is a no-op.

Currently patches:
  1. numpy scalar aliases (`np.float` etc.) that PTSA still references.
"""
from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _patch_numpy_removed_aliases()
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


# Auto-apply on import so the calling site doesn't need to remember.
apply()
