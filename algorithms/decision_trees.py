"""
decision_tree_helper.py
───────────────────────
Pure-Python ID3 / C4.5-style Decision Tree that:

• works on a Pandas DataFrame
• supports categorical + numeric attributes
• logs every split (entropy, gain) when verbose=True
• exposes:  fit  |  predict  |  print_tree  |  plot_tree

Dependencies: pandas, numpy, matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- utilities -------------------------------------------------
def _entropy(series: pd.Series) -> float:
    counts = series.value_counts()
    probs = counts / counts.sum()
    return -(probs * np.log2(probs)).sum()


def _info_gain(
    df: pd.DataFrame, target: str, attr: str, threshold: Any | None = None
) -> float:
    """Information gain of splitting df on attr.
    If threshold is given → numeric split ≤ / >."""
    base = _entropy(df[target])
    if threshold is None:  # categorical
        parts = [df[df[attr] == v] for v in df[attr].unique()]
    else:  # numeric
        parts = [df[df[attr] <= threshold], df[df[attr] > threshold]]
    rem = 0.0
    n = len(df)
    for part in parts:
        rem += (len(part) / n) * _entropy(part[target])
    return base - rem


def _best_numeric_threshold(
    df: pd.DataFrame, target: str, attr: str
) -> tuple[float, float]:
    """Returns (best_gain, best_threshold)."""
    df_sorted = df.sort_values(attr)
    values = df_sorted[attr].unique()
    best_gain, best_thr = -1, None
    for i in range(len(values) - 1):  # mid-points
        thr = (values[i] + values[i + 1]) / 2
        gain = _info_gain(df_sorted, target, attr, thr)
        if gain > best_gain:
            best_gain, best_thr = gain, thr
    return best_gain, best_thr


# ---------- tree structure --------------------------------------------
@dataclass
class DTNode:
    is_leaf: bool
    prediction: Any = None  # for leaves
    attr: str | None = None  # splitting attribute
    threshold: float | None = None  # for numeric
    branches: Dict[Any, "DTNode"] = field(default_factory=dict)


# ---------- training --------------------------------------------------
def fit_decision_tree(
    df: pd.DataFrame,
    target: str,
    max_depth: int | None = None,
    min_samples: int = 1,
    verbose: bool = False,
    depth: int = 0,
) -> DTNode:
    """Recursive ID3 / C4.5."""
    # stop conditions ---------------------------------------------------
    if len(df[target].unique()) == 1:
        return DTNode(True, prediction=df[target].iloc[0])
    if max_depth is not None and depth >= max_depth:
        return DTNode(True, prediction=df[target].mode()[0])
    if len(df) < min_samples or len(df.columns) == 1:
        return DTNode(True, prediction=df[target].mode()[0])

    # choose best attribute --------------------------------------------
    best_gain, best_attr, best_thr = -1, None, None
    for attr in df.columns.drop(target):
        if pd.api.types.is_numeric_dtype(df[attr]):
            gain, thr = _best_numeric_threshold(df, target, attr)
        else:
            gain, thr = _info_gain(df, target, attr), None
        if gain > best_gain:
            best_gain, best_attr, best_thr = gain, attr, thr

    if verbose:
        print(
            f"{'│   '*depth}➜ split on '{best_attr}' "
            f"{'≤ {:.2f}'.format(best_thr) if best_thr is not None else ''} "
            f"(gain={best_gain:.4f})"
        )

    if best_gain <= 0:
        return DTNode(True, prediction=df[target].mode()[0])

    # split -------------------------------------------------------------
    node = DTNode(False, attr=best_attr, threshold=best_thr)
    if best_thr is None:  # categorical
        for v, sub in df.groupby(best_attr):
            node.branches[v] = fit_decision_tree(
                sub.drop(columns=[best_attr]),
                target,
                max_depth,
                min_samples,
                verbose,
                depth + 1,
            )
    else:  # numeric
        le = df[df[best_attr] <= best_thr]
        gt = df[df[best_attr] > best_thr]
        node.branches["<="] = fit_decision_tree(
            le, target, max_depth, min_samples, verbose, depth + 1
        )
        node.branches[">"] = fit_decision_tree(
            gt, target, max_depth, min_samples, verbose, depth + 1
        )
    return node


# ---------- prediction ------------------------------------------------
def predict_instance(node: DTNode, row: pd.Series) -> Any:
    while not node.is_leaf:
        if node.threshold is None:  # categorical
            v = row[node.attr]
            node = node.branches.get(v, None)
            if node is None:  # unseen category → majority
                return None
        else:  # numeric
            node = (
                node.branches["<="]
                if row[node.attr] <= node.threshold
                else node.branches[">"]
            )
    return node.prediction


def predict(node: DTNode, df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: predict_instance(node, row), axis=1)


# ---------- pretty-print ----------------------------------------------
def print_tree(node: DTNode, depth: int = 0, value: str | None = None):
    indent = "│   " * depth
    prefix = f"{indent}{value} → " if value is not None else indent
    if node.is_leaf:
        print(f"{prefix}★ {node.prediction}")
    else:
        label = (
            f"{node.attr} ≤ {node.threshold:.2f}"
            if node.threshold is not None
            else node.attr
        )
        print(f"{prefix}{label}")
        for br, child in node.branches.items():
            child_val = (
                f"{br}" if node.threshold is None else ("≤" if br == "<=" else ">")
            )
            print_tree(child, depth + 1, child_val)


# ---------- tiny matplotlib plot (optional) ---------------------------
def plot_tree(
    node: DTNode,
    x: float = 0.5,
    y: float = 1.0,
    x_span: float = 0.5,
    level_gap: float = 0.1,
    ax=None,
):
    """Quick schematic (good for small trees)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
    ax.text(
        x,
        y,
        str(node.prediction if node.is_leaf else node.attr),
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", fc="w"),
    )
    if not node.is_leaf:
        n = len(node.branches)
        dx = x_span / max(n - 1, 1)
        child_x = x - x_span / 2
        for br, child in node.branches.items():
            child_x += dx
            ax.plot([x, child_x], [y - 0.02, y - level_gap + 0.02], "k-")
            ax.text((x + child_x) / 2, y - level_gap / 2, str(br), fontsize=8)
            plot_tree(
                child,
                x=child_x,
                y=y - level_gap,
                x_span=x_span / 2,
                level_gap=level_gap,
                ax=ax,
            )
    if ax.figure:
        plt.tight_layout()
