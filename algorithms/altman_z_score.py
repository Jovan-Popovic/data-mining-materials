from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# koeficient for 3 variants
_COEF: Dict[str, Dict[str, float]] = {
    "original": dict(x1=1.2, x2=1.4, x3=3.3, x4=0.6, x5=0.999),
    "private": dict(x1=0.717, x2=0.847, x3=3.1, x4=0.42, x5=0.998),
    "non_sales": dict(x1=6.65, x2=3.26, x3=6.72, x4=1.05, x5=0.0),  # x5 избачен
}

_THRESH: Dict[str, Tuple[float, float]] = {
    "original": (1.81, 2.99),  # distress <1.81, grey 1.81-2.99, safe >2.99
    "private": (1.21, 2.90),
    "non_sales": (1.10, 2.60),
}


def _label(z: float, low: float, high: float) -> str:
    if z < low:
        return "Distress"
    elif z < high:
        return "Grey"
    else:
        return "Safe"


def z_score(
    x: Dict[str, float] | Tuple[float, float, float, float, float],
    model: str = "original",
) -> Tuple[float, str]:
    """x може бити dict(x1=…,x2=…,…) или 5-елем. тјјпла."""
    if isinstance(x, (list, tuple)):
        x = {f"x{i+1}": v for i, v in enumerate(x)}
    coef = _COEF[model]
    z = sum(coef[k] * x.get(k, 0) for k in coef)
    label = _label(z, *_THRESH[model])
    return z, label


def batch_z(
    df: pd.DataFrame, cols: Dict[str, str], model: str = "original"
) -> pd.DataFrame:
    """
    cols = {'x1':'CA/TA', 'x2':'RE/TA', …}  → мапира називе колона.
    """
    coef = _COEF[model]
    z = sum(coef[k] * df[cols[k]] for k in coef)
    low, high = _THRESH[model]
    label = np.where(z < low, "Distress", np.where(z < high, "Grey", "Safe"))
    return pd.DataFrame({"Z": z, "Risk": label})
