from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
def _init_centroids(X: np.ndarray, k: int, method: str = "k++") -> np.ndarray:
    if method == "random":
        idx = np.random.choice(len(X), k, replace=False)
        return X[idx]
    # k-means++ initializer
    centroids = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X - centroids[-1]) ** 2).sum(axis=1, keepdims=True), axis=1)
        probs = d2 / d2.sum()
        cum = np.cumsum(probs)
        r = np.random.rand()
        centroids.append(X[np.searchsorted(cum, r)])
    return np.array(centroids)


def _assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return dists.argmin(axis=1)  # label za svaku tačku


def _update(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.vstack([X[labels == j].mean(axis=0) for j in range(k)])


# ------------------------------------------------------------
def kmeans_fit(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    init: str = "k++",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Vraća (centroids, labels, sse_history)
    """
    C = _init_centroids(X, k, init)
    sse_hist = []
    for it in range(1, max_iter + 1):
        labels = _assign(X, C)
        sse = ((X - C[labels]) ** 2).sum()
        sse_hist.append(sse)

        if verbose:
            print(f"iter {it:<3d}  SSE={sse:.2f}")

        new_C = _update(X, labels, k)
        shift = np.linalg.norm(C - new_C, axis=1).max()

        if verbose:
            print(f"          max centroid shift = {shift:.4f}")

        if shift <= tol:
            if verbose:
                print("⇣  konvergirano (shift ≤ tol)")
            break
        C = new_C
    return C, labels, sse_hist


def predict(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return _assign(X, centroids)


# ------------------------------------------------------------
def plot_clusters(
    X: np.ndarray, centroids: np.ndarray, labels: np.ndarray, title: str = ""
):
    k = len(centroids)
    colors = plt.cm.get_cmap("tab10", k)
    plt.figure(figsize=(4, 4))
    for j in range(k):
        pts = X[labels == j]
        plt.scatter(pts[:, 0], pts[:, 1], s=30, color=colors(j), label=f"Cluster {j}")
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="black",
        marker="X",
        s=120,
        label="Centroids",
    )
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def elbow_plot(X: np.ndarray, k_range: range, **fit_kw):
    sses = []
    for k in k_range:
        _, _, hist = kmeans_fit(X, k, verbose=False, **fit_kw)
        sses.append(hist[-1])
    plt.figure()
    plt.plot(list(k_range), sses, "o-")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.title("Elbow method")
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


# --- Tiny helper: k‑means that keeps history --------------------------------
def kmeans_with_history(X, k, max_iter=10, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    history = []  # list of (centroids, labels)

    for _ in range(max_iter):
        # assignment
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = dists.argmin(axis=1)
        history.append((centroids.copy(), labels.copy()))  # snapshot *before* update

        # update
        new_centroids = np.vstack([X[labels == j].mean(axis=0) for j in range(k)])
        if np.linalg.norm(new_centroids - centroids) <= tol:
            centroids = new_centroids
            history.append((centroids.copy(), labels.copy()))  # final snapshot
            break
        centroids = new_centroids
    return history


# --- Sample 2‑D dataset ------------------------------------------------------
np.random.seed(42)
A = np.random.randn(40, 2) * 0.6 + [1, 1]
B = np.random.randn(40, 2) * 0.5 + [5, 4]
C = np.random.randn(40, 2) * 0.4 + [1, 5]
X = np.vstack([A, B, C])

# ---------------------------------------------------------------------------
history = kmeans_with_history(X, k=3)

# --- Visualise every iteration ---------------------------------------------
for step, (C, labels) in enumerate(history, 1):
    plt.figure(figsize=(4, 4))
    for j in range(3):
        pts = X[labels == j]
        plt.scatter(pts[:, 0], pts[:, 1], s=20, label=f"Cluster {j}")
    plt.scatter(C[:, 0], C[:, 1], marker="X", s=100, label="Centroids", edgecolor="k")
    plt.title(f"Iteration {step}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
