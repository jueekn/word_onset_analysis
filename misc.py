"""misc.py — session-label and pickle helpers shared across the pipeline."""
from __future__ import annotations

import pickle
from typing import Any, Sequence

import numpy.typing as npt
import pandas as pd

NDArrayAny = npt.NDArray[Any]


def ftag(dfrow: pd.Series) -> str:
    """Filename tag for a session: '{sub}_{exp}_{sess}'."""
    return f"{dfrow['sub']}_{dfrow['exp']}_{dfrow['sess']}"


def get_dfrow(dfrow: Sequence[Any] | NDArrayAny) -> pd.Series:
    """Coerce a (sub, exp, sess) tuple/list/array to a Series."""
    sub, exp, sess = dfrow
    return pd.Series({'sub': str(sub), 'exp': str(exp), 'sess': int(sess)})


def load_pickle(path: str) -> Any:
    """Load any pickle file. Return type is genuinely Any (caller-known)."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path: str, obj: Any) -> None:
    """Pickle obj to path (binary, default protocol)."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)



