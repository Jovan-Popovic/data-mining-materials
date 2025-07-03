from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np


# ───────────── struktura ćelije ─────────────
class _Cell:
    def __init__(self, bounds, depth: int):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.points: np.ndarray | None = None  # indeksi tačaka
        self.children: List["_Cell"] = []
        self.depth = depth

    # geometrija
    def contains(self, X):
        x, y = X[:, 0], X[:, 1]
        mask = (x >= self.xmin) & (x < self.xmax) & (y >= self.ymin) & (y < self.ymax)
        return np.where(mask)[0]

    def split(self, m):
        dx = (self.xmax - self.xmin) / m
        dy = (self.ymax - self.ymin) / m
        for ix in range(m):
            for iy in range(m):
                bounds = (
                    self.xmin + ix * dx,
                    self.xmin + (ix + 1) * dx,
                    self.ymin + iy * dy,
                    self.ymin + (iy + 1) * dy,
                )
                self.children.append(_Cell(bounds, self.depth + 1))


# ───────────── STING klasa ─────────────
class STING:
    def __init__(self, max_depth: int = 4, m: int = 4, cell_cap: int = 10):
        self.max_depth = max_depth
        self.m = m
        self.cell_cap = cell_cap

    def fit(self, X: np.ndarray):
        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()
        root = _Cell((xmin, xmax, ymin, ymax), 0)
        root.points = np.arange(len(X))
        # DFS gradnja
        stack = [root]
        while stack:
            cell = stack.pop()
            if cell.depth >= self.max_depth or len(cell.points) <= self.cell_cap:
                continue
            cell.split(self.m)
            for ch in cell.children:
                ch.points = cell.contains(X[cell.points])
                ch.points = cell.points[ch.points]  # globalni indeksi
            stack.extend(cell.children)
        self.root_ = root
        self.X_ = X
        return self

    # ── spajanje gustih ćelija na najnižem nivou ──
    def _collect_leaves(self):
        leaves = []
        stack = [self.root_]
        while stack:
            c = stack.pop()
            if c.children:
                stack.extend(c.children)
            else:
                leaves.append(c)
        return leaves

    def cluster(self, min_pts: int = 15):
        leaves = self._collect_leaves()
        dense = [c for c in leaves if len(c.points) >= min_pts]
        labels = np.full(len(self.X_), -1)
        cl_id = 0
        for c in dense:
            # BFS kroz susedne “dense” ćelije (4-susedstvo)
            stack = [c]
            current = []
            while stack:
                cell = stack.pop()
                if cell in current:
                    continue
                current.append(cell)
                for nb in dense:
                    if nb in current:
                        continue
                    touch = (
                        (
                            abs(nb.xmin - cell.xmax) < 1e-9
                            or abs(nb.xmax - cell.xmin) < 1e-9
                        )
                        and (nb.ymin < cell.ymax and nb.ymax > cell.ymin)
                        or (
                            abs(nb.ymin - cell.ymax) < 1e-9
                            or abs(nb.ymax - cell.ymin) < 1e-9
                        )
                        and (nb.xmin < cell.xmax and nb.xmax > cell.xmin)
                    )
                    if touch:
                        stack.append(nb)
            # dodijeli labelu svim tačkama iz “current”
            pts = np.concatenate([cb.points for cb in current])
            labels[pts] = cl_id
            cl_id += 1
        self.labels_ = labels
        self.n_clusters_ = cl_id
        return labels


# ───────────── plot 2-D grid + klasteri ─────────────
def plot_sting_grid(model: STING, X: np.ndarray, labels: np.ndarray, title=""):
    cmap = plt.cm.get_cmap("tab10", np.max(labels) + 1 or 1)
    plt.figure(figsize=(5, 5))
    # tačke
    for cid in range(model.n_clusters_):
        pts = X[labels == cid]
        plt.scatter(pts[:, 0], pts[:, 1], s=25, color=cmap(cid), label=f"cl {cid}")
    plt.scatter(*X[labels == -1].T, s=20, c="k", marker="x", label="noise")
    # ćelije
    stack = [model.root_]
    for c in stack:
        plt.gca().add_patch(
            plt.Rectangle(
                (c.xmin, c.ymin),
                c.xmax - c.xmin,
                c.ymax - c.ymin,
                fill=False,
                edgecolor="gray",
                lw=0.3,
            )
        )
        stack.extend(c.children)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
