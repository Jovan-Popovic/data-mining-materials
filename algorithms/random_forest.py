from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ────────────────────────────────────────────────────────────────────
def train_test_split(X, y, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


# ───────────────────────── DecisionTree (very small) ───────────────
class _DTNode:  # internal
    def __init__(self, *, gini, klass=None, feat=None, thr=None, left=None, right=None):
        self.gini = gini
        self.klass = klass
        self.feat = feat
        self.thr = thr
        self.left = left
        self.right = right


class _DecisionTree:
    def __init__(self, max_depth: int = 10, min_samples: int = 2):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y):
        def build(idx, depth):
            g = 1 - sum((np.sum(y[idx] == c) / len(idx)) ** 2 for c in (0, 1))
            if depth >= self.max_depth or len(idx) < self.min_samples or g == 0:
                klass = Counter(y[idx]).most_common(1)[0][0]
                return _DTNode(gini=g, klass=klass)
            best_feat, best_thr, best_g, left, right = None, None, 1.0, None, None
            for f in range(X.shape[1]):
                xs = X[idx, f]
                thresholds = np.unique(xs)
                for t in thresholds:
                    l = idx[xs <= t]
                    r = idx[xs > t]
                    if len(l) == 0 or len(r) == 0:
                        continue
                    g_l = 1 - sum((np.sum(y[l] == c) / len(l)) ** 2 for c in (0, 1))
                    g_r = 1 - sum((np.sum(y[r] == c) / len(r)) ** 2 for c in (0, 1))
                    g_split = (len(l) * g_l + len(r) * g_r) / len(idx)
                    if g_split < best_g:
                        best_feat, best_thr, best_g, left, right = f, t, g_split, l, r
            if best_feat is None:
                klass = Counter(y[idx]).most_common(1)[0][0]
                return _DTNode(gini=g, klass=klass)
            return _DTNode(
                gini=g,
                feat=best_feat,
                thr=best_thr,
                left=build(left, depth + 1),
                right=build(right, depth + 1),
            )

        self.root_ = build(np.arange(len(X)), 0)
        return self

    def _predict_one(self, x):
        node = self.root_
        while node.klass is None:
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.klass

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


# ───────────────────────── RandomForestClassifier ──────────────────
class RandomForestClassifier:
    def __init__(
        self, n_estimators=30, max_depth=10, min_samples=2, verbose=False, oob=True
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.verbose = verbose
        self.oob = oob

    def fit(self, X, y):
        m, d = X.shape
        rng = np.random.default_rng()
        self.trees_: List[_DecisionTree] = []
        self.features_: List[np.ndarray] = []
        self.oob_mask_ = np.zeros((self.n_estimators, m), dtype=bool)

        for i in range(self.n_estimators):
            samp_idx = rng.choice(m, m, replace=True)
            oob_idx = np.setdiff1d(np.arange(m), samp_idx, assume_unique=True)
            feat_idx = rng.choice(d, int(np.sqrt(d)) or 1, replace=False)
            tree = _DecisionTree(self.max_depth, self.min_samples)
            tree.fit(X[samp_idx][:, feat_idx], y[samp_idx])
            self.trees_.append(tree)
            self.features_.append(feat_idx)
            self.oob_mask_[i, oob_idx] = True
            if self.verbose:
                print(
                    f"Tree {i}: depth≈{self._tree_depth(tree.root_)}  "
                    f"features={feat_idx}"
                )
        return self

    def _tree_depth(self, node):
        return (
            1
            if node.klass is not None
            else 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))
        )

    def predict(self, X):
        votes = np.zeros((len(X), self.n_estimators), dtype=int)
        for i, (tr, feat) in enumerate(zip(self.trees_, self.features_)):
            votes[:, i] = tr.predict(X[:, feat])
        return np.apply_along_axis(
            lambda row: Counter(row).most_common(1)[0][0], 1, votes
        )

    def oob_score(self, X, y):
        if not self.oob:
            raise ValueError("oob=True not enabled")
        m = len(X)
        votes = np.full((m, 0), 0)
        for i, (tr, feat) in enumerate(zip(self.trees_, self.features_)):
            mask = self.oob_mask_[i]
            pred = tr.predict(X[mask][:, feat])
            col = np.full(m, np.nan)
            col[mask] = pred
            votes = np.column_stack([votes, col])
        final = np.apply_along_axis(
            lambda row: (
                Counter(row[~np.isnan(row)]).most_common(1)[0][0]
                if np.any(~np.isnan(row))
                else -1
            ),
            1,
            votes,
        )
        return accuracy(y[final != -1], final[final != -1])


# ─────────────────────────── 2-D plot ───────────────────────────────
def plot_decision_boundary(model, X, y, h=0.1, title=""):
    if X.shape[1] != 2:
        raise ValueError("2-D only")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Pastel2")
    for cls in np.unique(y):
        pts = X[y == cls]
        plt.scatter(pts[:, 0], pts[:, 1], edgecolor="k", label=f"class {cls}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
