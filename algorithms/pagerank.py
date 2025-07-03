from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def pagerank(adj: np.ndarray, d: float = 0.85, tol: float = 1e-6, max_iter: int = 100):
    """
    adj – square adjacency matrix (0/1); dangling nodes allowed.
    Returns stationary distribution π.
    """
    n = adj.shape[0]
    # column-stochastic P
    out = adj.sum(axis=0)
    P = np.where(out > 0, adj / out, 1 / n)  # handle dangling → uniform
    π = np.full(n, 1 / n)
    v = np.full(n, 1 / n)
    for it in range(max_iter):
        π_new = d * (P @ π) + (1 - d) * v
        if np.linalg.norm(π_new - π, 1) < tol:
            break
        π = π_new
    return π_new


# ─────────── tiny draw helper (matplotlib only) ───────────
def draw_graph(adj: np.ndarray, labels=None):
    n = adj.shape[0]
    θ = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([np.cos(θ), np.sin(θ)])
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                plt.arrow(
                    *xy[j],
                    *(xy[i] - xy[j]) * 0.8,
                    head_width=0.05,
                    length_includes_head=True,
                    fc="0.3",
                    ec="0.3",
                    alpha=0.8,
                )
    for i, (x, y) in enumerate(xy):
        plt.scatter(x, y, s=500, c="w", edgecolors="k")
        txt = labels[i] if labels else str(i)
        plt.text(x, y, txt, ha="center", va="center")
    plt.show()
