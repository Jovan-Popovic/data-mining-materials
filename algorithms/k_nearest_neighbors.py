from __future__ import annotations

from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


# ───────────────── distance helpers ─────────────────────────────────────────
def _distance(
    a: np.ndarray,
    b: np.ndarray,
    metric: Literal["euclidean", "manhattan", "minkowski"] = "euclidean",
    p: int = 3,
) -> float:
    if metric == "euclidean":
        return np.linalg.norm(a - b)
    if metric == "manhattan":
        return np.abs(a - b).sum()
    # minkowski
    return (np.abs(a - b) ** p).sum() ** (1 / p)


# ─────────────────── KNN class ──────────────────────────────────────────────
class KNN:
    def __init__(
        self,
        k: int = 3,
        metric: str = "euclidean",
        weighted: bool = False,
        p: int = 3,
        verbose: bool = False,
    ):
        """
        k         – number of neighbours
        metric    – euclidean | manhattan | minkowski
        weighted  – distance-weighted vote (1/d)
        p         – Minkowski power (if chosen)
        verbose   – prints neighbour list on predict
        """
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.p = p
        self.verbose = verbose

    # ------------- training (store data) -----------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)

    # ------------- prediction ----------------------------------------------
    def _get_neighbours(self, x: np.ndarray):
        dists = np.array([_distance(x, xi, self.metric, self.p) for xi in self.X])
        idx = np.argsort(dists)[: self.k]
        return idx, dists[idx]

    def predict(self, Xq: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in Xq])

    def predict_proba(self, Xq: np.ndarray) -> np.ndarray:
        return np.vstack([self._predict_one_proba(x) for x in Xq])

    # ---------- internal helpers -------------------------------------------
    def _predict_one(self, x: np.ndarray):
        probs = self._predict_one_proba(x)
        # highest probability (ties → first come)
        return self.classes_[np.argmax(probs)]

    def _predict_one_proba(self, x: np.ndarray):
        idx, d = self._get_neighbours(x)
        votes = Counter()
        for i, dist in zip(idx, d):
            weight = 1.0 / (dist + 1e-9) if self.weighted else 1.0
            votes[self.y[i]] += weight

        if self.verbose:
            print(f"\nQuery {x} – neighbours:")
            for i, dist in zip(idx, d):
                print(f"  idx={i:<3d}  dist={dist:.3f}  class={self.y[i]}")
            print("  votes:", dict(votes))

        total = sum(votes.values())
        return np.array([votes[c] / total for c in self.classes_])


# ─────────────────── tiny utilities ────────────────────────────────────────
def train_test_split(X, y, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


# ─────────────────── 2-D decision surface plot (for toy data) ──────────────
def plot_2d_knn(
    model: KNN, X: np.ndarray, y: np.ndarray, h: float = 0.1, cmap="Pastel2"
):
    if X.shape[1] != 2:
        raise ValueError("plot_2d_knn works only for 2-D data")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    # training points
    classes = np.unique(y)
    for c in classes:
        pts = X[y == c]
        plt.scatter(pts[:, 0], pts[:, 1], label=f"class {c}", edgecolor="k")
    plt.legend()
    plt.title(f"k={model.k}, metric={model.metric}")
    plt.grid(True)
    plt.show()
