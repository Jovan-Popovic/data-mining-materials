from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

_COEF = dict(
    const=-1.32,
    x1=-0.407,
    x2=6.03,
    x3=-1.43,
    x4=0.076,
    x5=-2.37,
    x6=-1.83,
    x7=0.285,
    x8=-1.72,
    x9=-0.521,
)


def _logit_to_prob(P: float) -> float:
    return 1 / (1 + np.exp(-P))


def o_score(x: Dict[str, float] | Tuple[float, ...]) -> Tuple[float, str]:
    """
    Прихвата dict са кључевима x1…x9 *или* тјјплу (x1,…,x9).
    Враћа (O_score, ’High-risk’ | ’Low-risk’).
    """
    if isinstance(x, (list, tuple)):
        x = {f"x{i+1}": v for i, v in enumerate(x)}
    P = _COEF["const"] + sum(_COEF[k] * x.get(k, 0) for k in _COEF if k != "const")
    O = _logit_to_prob(P)
    label = "High-risk" if O > 0.5 else "Low-risk"
    return O, label


# ─── batch из DataFrame-а ─────────────────────────────────────────
def batch_o(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    """
    cols = {'x1':'Size', 'x2':'Leverage', … }
    Враћа DataFrame са O_score и Risk label.
    """
    P = _COEF["const"]
    for k, col in cols.items():
        P += _COEF[k] * df[col]
    O = _logit_to_prob(P)
    label = np.where(O > 0.5, "High-risk", "Low-risk")
    return pd.DataFrame({"O_score": O, "Risk": label})
