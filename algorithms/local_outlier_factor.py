from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _knn_distances(X, k):
    m = len(X)
    d = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    sort_idx = np.argsort(d, axis=1)
    knn_idx = sort_idx[:, 1 : k + 1]  # skip self (0)
    knn_dist = np.take_along_axis(d, knn_idx, axis=1)
    k_dist = d[np.arange(m), knn_idx[:, -1]]
    return knn_idx, knn_dist, k_dist


def lof_scores(X, k=5):
    knn_idx, knn_dist, k_dist = _knn_distances(X, k)
    reach_dist = np.maximum(knn_dist, k_dist[knn_idx])
    lrd = k / reach_dist.sum(axis=1)
    lof = []
    for i in range(len(X)):
        lof.append((lrd[knn_idx[i]].sum() / k) / lrd[i])
    return np.array(lof)


def plot_lof_scores(X, lof, thresh=1.4, title=""):
    assert X.shape[1] == 2, "2-D only"
    normal = X[lof < thresh]
    out = X[lof >= thresh]
    plt.figure(figsize=(4, 4))
    plt.scatter(*normal.T, s=25, label="normal")
    plt.scatter(*out.T, s=25, c="r", marker="x", label="outlier")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
