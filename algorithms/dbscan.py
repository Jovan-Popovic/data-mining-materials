from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── DBSCAN core ────────────────────────────
class DBSCAN:
    def __init__(self, eps: float, min_pts: int = 5, verbose: bool = False):
        self.eps = eps
        self.min_pts = min_pts
        self.verbose = verbose

    def _region_query(self, X, idx):
        d = np.linalg.norm(X - X[idx], axis=1)
        return np.where(d <= self.eps)[0]

    def fit(self, X: np.ndarray):
        n = len(X)
        labels = np.full(n, -1)  # -1 = unvisited/noise
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:  # already processed
                continue
            N_i = self._region_query(X, i)
            if len(N_i) < self.min_pts:  # noise
                labels[i] = -1
                continue

            # start new cluster
            if self.verbose:
                print(f"\nSeed #{i} → cluster {cluster_id}")
            labels[i] = cluster_id
            seed_set = list(N_i)

            j = 0
            while j < len(seed_set):
                p = seed_set[j]
                if labels[p] == -1:  # previously marked noise → now border
                    labels[p] = cluster_id
                if labels[p] == cluster_id:
                    j += 1
                    continue  # already expanded
                labels[p] = cluster_id
                N_p = self._region_query(X, p)
                if len(N_p) >= self.min_pts:
                    seed_set.extend(list(N_p))
                    if self.verbose:
                        print(f"  expand point {p}: added {len(N_p)} neighbours")
                j += 1
            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        return self


# ──────────────────────── helper plots ──────────────────────────────
def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, title=""):
    if X.shape[1] != 2:
        raise ValueError("2-D only")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    plt.figure(figsize=(4, 4))
    for cid in range(n_clusters):
        pts = X[labels == cid]
        plt.scatter(pts[:, 0], pts[:, 1], s=30, color=cmap(cid), label=f"cluster {cid}")
    noise = X[labels == -1]
    if len(noise):
        plt.scatter(noise[:, 0], noise[:, 1], s=20, c="k", marker="x", label="noise")
    plt.legend()
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def k_distance_plot(X: np.ndarray, k: int = 4):
    """Elbow method: plot sorted k-th NN distance."""
    from scipy.spatial import KDTree

    tree = KDTree(X)
    dists, _ = tree.query(X, k=k + 1)  # k+1 because first NN = itself
    kth = np.sort(dists[:, k])  # k-th distance per point
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(len(X)), kth[::-1])  # descending
    plt.xlabel("points (descending)")
    plt.ylabel(f"{k}-NN distance")
    plt.title("k-distance graph")
    plt.grid(True)
    plt.show()
