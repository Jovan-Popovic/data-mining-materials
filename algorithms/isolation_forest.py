from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ───────────── Isolation Tree ─────────────
class _ITNode:
    def __init__(self, depth, size, feat=None, thr=None, left=None, right=None):
        self.depth = depth
        self.size = size
        self.feat = feat
        self.thr = thr
        self.left = left
        self.right = right


class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X):
        rng = np.random.default_rng()

        def build(data, depth):
            if depth >= self.max_depth or len(data) <= 1:
                return _ITNode(depth, len(data))
            feat = rng.integers(data.shape[1])
            min_, max_ = data[:, feat].min(), data[:, feat].max()
            thr = rng.uniform(min_, max_)
            left = data[data[:, feat] <= thr]
            right = data[data[:, feat] > thr]
            return _ITNode(
                depth,
                len(data),
                feat,
                thr,
                build(left, depth + 1),
                build(right, depth + 1),
            )

        self.root_ = build(X, 0)
        return self

    def path_length(self, x):
        node = self.root_
        length = 0
        while node.left and node.right:
            length += 1
            node = node.left if x[node.feat] <= node.thr else node.right
        return length + c_factor(node.size)


# ───────────── Forest ─────────────
def c_factor(n):  # expected path length of unsuccessful search BST
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256, max_depth=None, verbose=False):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.verbose = verbose

    def fit(self, X):
        rng = np.random.default_rng()
        m = len(X)
        self.trees = []
        depth = self.max_depth or int(np.ceil(np.log2(self.sample_size)))
        for i in range(self.n_trees):
            sample = X[rng.choice(m, self.sample_size, replace=False)]
            tree = IsolationTree(depth).fit(sample)
            self.trees.append(tree)
            if self.verbose:
                print(f"Tree {i} built.")
        self.c = c_factor(self.sample_size)
        return self

    def score_samples(self, X):
        paths = np.array([[t.path_length(x) for t in self.trees] for x in X])
        E = paths.mean(axis=1)
        return 2 ** (-E / self.c)  # anomaly score ∈ (0,1]

    def predict(self, X, thresh=0.6):
        return np.where(self.score_samples(X) >= thresh, -1, 1)


# ───────────── helper plot ─────────────
def plot_iforest_scores(X, scores, thresh=0.6, title=""):
    assert X.shape[1] == 2, "2-D only"
    normal = X[scores < thresh]
    out = X[scores >= thresh]
    plt.figure(figsize=(4, 4))
    plt.scatter(*normal.T, s=25, label="normal")
    plt.scatter(*out.T, s=25, c="r", marker="x", label="outlier")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
