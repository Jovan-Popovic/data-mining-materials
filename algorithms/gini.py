from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ───────────────────────── Gini core ─────────────────────────
def gini(labels: pd.Series | np.ndarray) -> float:
    """Gini impurity of a label vector (0 … 0.5)."""
    counts = pd.Series(labels).value_counts(normalize=True)
    return 1 - np.sum(counts**2)


def gini_after_split(parent: pd.Series, splits: List[pd.Series]) -> float:
    """Weighted impurity after splitting parent into child sets."""
    m = len(parent)
    return sum(len(s) / m * gini(s) for s in splits)


# ───────────────────────── Best-split finder ─────────────────
def _best_numeric_threshold(col: pd.Series, y: pd.Series) -> Tuple[float, float]:
    df = pd.DataFrame({"x": col, "y": y}).sort_values("x")
    uniq = df["x"].unique()
    best_gain, best_thr = -1, None
    parent_g = gini(y)
    for i in range(len(uniq) - 1):
        thr = (uniq[i] + uniq[i + 1]) / 2
        left = df[df["x"] <= thr]["y"]
        right = df[df["x"] > thr]["y"]
        gain = parent_g - gini_after_split(y, [left, right])
        if gain > best_gain:
            best_gain, best_thr = gain, thr
    return best_gain, best_thr


def best_split(df: pd.DataFrame, target: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Returns dict with:
      { 'attribute', 'threshold'(None if cat), 'gain', 'gini_parent' }
    """
    y = df[target]
    parent_g = gini(y)
    best = {"gain": -1}
    for attr in df.columns.drop(target):
        col = df[attr]
        if pd.api.types.is_numeric_dtype(col):
            gain, thr = _best_numeric_threshold(col, y)
            if verbose:
                print(f"[num] {attr:<12} thr={thr:.3f}  gain={gain:.4f}")
            if gain > best["gain"]:
                best = {
                    "attribute": attr,
                    "threshold": thr,
                    "gain": gain,
                    "gini_parent": parent_g,
                }
        else:  # categorical
            splits = [y[col == v] for v in col.unique()]
            gain = parent_g - gini_after_split(y, splits)
            if verbose:
                print(f"[cat] {attr:<12} gain={gain:.4f}")
            if gain > best["gain"]:
                best = {
                    "attribute": attr,
                    "threshold": None,
                    "gain": gain,
                    "gini_parent": parent_g,
                }
    return best
