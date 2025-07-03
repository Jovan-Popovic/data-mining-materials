from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ────────────────────────────────────────────────────────────────────
# internal helpers
# ────────────────────────────────────────────────────────────────────
def _init_centroids(X: np.ndarray, k: int, method: str) -> np.ndarray:
    rng = np.random.default_rng()
    if method == "random":
        return X[rng.choice(len(X), k, replace=False)]
    # -- k-means++ init
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None] - centroids[-1]) ** 2).sum(axis=2), axis=1)
        probs = d2 / d2.sum()
        centroids.append(X[rng.choice(len(X), p=probs)])
    return np.array(centroids)


def _assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(X[:, None] - C[None, :], axis=2)
    return d.argmin(axis=1)


def _update(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.vstack([X[labels == j].mean(axis=0) for j in range(k)])


# ────────────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────────────
def kmeans_fit(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    init: str = "k++",  # "k++" or "random"
    verbose: bool = False,
    return_history: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, List[float], List[Tuple[np.ndarray, np.ndarray]] | None
]:
    """
    Returns: centroids, labels, sse_history, history (or None)
    history = [(centroids_0, labels_0), ...] snapshot *before* every update
    """
    C = _init_centroids(X, k, init)
    sse_hist: List[float] = []
    history: List[Tuple[np.ndarray, np.ndarray]] = []

    for it in range(1, max_iter + 1):
        labels = _assign(X, C)
        if return_history:
            history.append((C.copy(), labels.copy()))

        sse = ((X - C[labels]) ** 2).sum()
        sse_hist.append(sse)
        if verbose:
            print(f"iter {it:<3d}  SSE={sse:.4f}")

        new_C = _update(X, labels, k)
        shift = np.linalg.norm(new_C - C, axis=1).max()
        if verbose:
            print(f"          max centroid shift = {shift:.6f}")

        if shift <= tol:
            if verbose:
                print("⇣  converged (shift ≤ tol)")
            C = new_C
            break
        C = new_C

    if return_history:
        history.append((C.copy(), _assign(X, C)))
        return C, labels, sse_hist, history
    return C, labels, sse_hist, None


def predict(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return _assign(X, centroids)


# ───────────────── plotting helpers ─────────────────────────────────
def plot_clusters(
    X: np.ndarray, centroids: np.ndarray, labels: np.ndarray, *, title: str = ""
):
    k = len(centroids)
    cmap = plt.cm.get_cmap("tab10", k)
    plt.figure(figsize=(4, 4))
    for j in range(k):
        pts = X[labels == j]
        plt.scatter(pts[:, 0], pts[:, 1], s=25, color=cmap(j), label=f"C{j}")
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=120,
        color="black",
        label="centroids",
    )
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def elbow_plot(X: np.ndarray, k_range: range, **fit_kw):
    sse = [kmeans_fit(X, k, verbose=False)[2][-1] for k in k_range]
    plt.figure()
    plt.plot(list(k_range), sse, "o-")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.title("Elbow method")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_history(
    X: np.ndarray, history: List[Tuple[np.ndarray, np.ndarray]], *, every: int = 1
):
    """Quick multi-frame visualisation of all snapshots"""
    k = len(history[0][0])
    cmap = plt.cm.get_cmap("tab10", k)
    for step, (C, labels) in enumerate(history):
        if step % every:  # skip frames if wanted
            continue
        plt.figure(figsize=(4, 4))
        for j in range(k):
            pts = X[labels == j]
            plt.scatter(pts[:, 0], pts[:, 1], s=20, color=cmap(j))
        plt.scatter(C[:, 0], C[:, 1], marker="X", s=120, color="black")
        plt.title(f"Iteration {step+1}")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
