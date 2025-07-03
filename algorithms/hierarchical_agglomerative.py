from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ───────────── HAC core ─────────────
class HAC:
    def __init__(self, linkage: str = "single"):
        self.linkage = linkage

    def _dist(self, X, i, j):
        if self.linkage == "single":
            return np.min(np.linalg.norm(X[i][:, None] - X[j], axis=0))
        if self.linkage == "complete":
            return np.max(np.linalg.norm(X[i][:, None] - X[j], axis=0))
        # average
        return np.mean(np.linalg.norm(X[i][:, None] - X[j], axis=0))

    def fit(self, X: np.ndarray):
        m = len(X)
        clusters = {i: [i] for i in range(m)}
        Z = []
        for step in range(m - 1):
            keys = list(clusters.keys())
            best = (None, None, 1e9)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ci, cj = keys[i], keys[j]
                    d = self._dist(X, clusters[ci], clusters[cj])
                    if d < best[2]:
                        best = (ci, cj, d)
            ci, cj, d = best
            new_id = max(clusters) + 1
            size = len(clusters[ci]) + len(clusters[cj])
            Z.append([ci, cj, d, size])
            clusters[new_id] = clusters.pop(ci) + clusters.pop(cj)
        self.Z_ = np.array(Z)
        return self


# ───────────── flat cut ─────────────
def cluster_cut(Z, n_clusters: int | None = None, distance: float | None = None):
    m = Z.shape[0] + 1
    parent = {i: i for i in range(m)}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    if n_clusters:
        thresh_step = m - n_clusters
    else:
        thresh_step = np.sum(Z[:, 2] <= distance)
    for i in range(Z.shape[0] - thresh_step):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        parent[find(a)] = find(b)
    labels = np.array([find(i) for i in range(m)])
    unique = {v: k for k, v in enumerate(np.unique(labels))}
    return np.array([unique[l] for l in labels])


# ───────────── dendrogram (simple) ─────────────
def plot_dendrogram(Z, labels=None, **kw):
    from scipy.cluster.hierarchy import dendrogram

    plt.figure(figsize=(6, 3))
    dendrogram(Z, labels=labels, **kw)
    plt.title("HAC dendrogram")
    plt.show()
