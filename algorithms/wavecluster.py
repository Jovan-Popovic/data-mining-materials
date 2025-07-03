from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ─────────── pomoćne funkcije ───────────
def _build_grid(X, grid):
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    gx = np.floor((X[:, 0] - xmin) / (xmax - xmin + 1e-9) * grid).astype(int)
    gy = np.floor((X[:, 1] - ymin) / (ymax - ymin + 1e-9) * grid).astype(int)
    gx[gx == grid] = grid - 1
    gy[gy == grid] = grid - 1
    dens = np.zeros((grid, grid), dtype=int)
    for i, j in zip(gx, gy):
        dens[i, j] += 1
    return dens, (xmin, xmax, ymin, ymax), (gx, gy)


def _haar2d(mat):
    """jedan korak diskretne 2-D Haar transformacije (periodic pad)."""
    # po kolonama
    a = (mat[:, 0::2] + mat[:, 1::2]) / 2
    d = (mat[:, 0::2] - mat[:, 1::2]) / 2
    # po redovima
    LL = (a[0::2] + a[1::2]) / 2
    LH = (a[0::2] - a[1::2]) / 2
    HL = (d[0::2] + d[1::2]) / 2
    HH = (d[0::2] - d[1::2]) / 2
    # rekonstruiši samo LL (smooth) nazad na originalni oblik
    s = np.zeros_like(mat, dtype=float)
    s[0::2, 0::2] = LL
    s[1::2, 0::2] = LL
    s[0::2, 1::2] = LL
    s[1::2, 1::2] = LL
    return s, (LH, HL, HH)


# ─────────── glavna klasa ───────────
class WaveCluster:
    def __init__(
        self, grid: int = 64, levels: int = 2, tau: float = 0.2, min_pts: int = 10
    ):
        """
        grid    – veličina kvadratnog rastera
        levels  – koliko puta primeniti Haar + threshold
        tau     – prag (procenat max density) za “nuliranje” detalja
        min_pts – minimalni smoothed density da ćelija bude core
        """
        self.grid = grid
        self.levels = levels
        self.tau = tau
        self.min_pts = min_pts

    def fit(self, X: np.ndarray):
        dens, bbox, (gx, gy) = _build_grid(X, self.grid)
        smooth = dens.astype(float)
        for _ in range(self.levels):
            smooth, _ = _haar2d(smooth)
            # threshold detalja (set to 0 čime se zadržava samo “talas”)
            smooth[np.abs(smooth) < self.tau * smooth.max()] = 0
        self.density_ = smooth
        # pronalaženje gustih ćelija
        core = np.where(smooth >= self.min_pts, 1, 0)
        labels_grid = -np.ones_like(core, int)
        cid = 0
        for i in range(self.grid):
            for j in range(self.grid):
                if core[i, j] and labels_grid[i, j] == -1:
                    # flood fill 4-neighbour
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        if not (0 <= x < self.grid and 0 <= y < self.grid):
                            continue
                        if core[x, y] == 0 or labels_grid[x, y] != -1:
                            continue
                        labels_grid[x, y] = cid
                        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
                    cid += 1
        # map back to points
        labels = labels_grid[gx, gy]
        self.labels_ = labels
        self.n_clusters_ = cid
        self.bbox_ = bbox
        return self


# ─────────── plot ───────────
def plot_wavecluster(model: WaveCluster, X):
    labels = model.labels_
    k = model.n_clusters_
    cmap = plt.cm.get_cmap("tab10", k)
    plt.figure(figsize=(5, 5))
    for cid in range(k):
        pts = X[labels == cid]
        plt.scatter(*pts.T, s=25, color=cmap(cid), label=f"cl {cid}")
    plt.scatter(*X[labels == -1].T, s=20, c="k", marker="x", label="noise")
    xmin, xmax, ymin, ymax = model.bbox_
    plt.title(f"WaveCluster  (k={k})")
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.axis("equal")
    plt.show()
