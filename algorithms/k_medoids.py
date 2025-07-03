from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np


# ───────────────── helper za distance matrice ─────────────────
def _pairwise_dists(X: np.ndarray):
    m = len(X)
    D = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    return D + np.eye(m) * 1e-9  # da izbjegnemo nule na dijagonali


# ───────────────── KMedoids klasa ─────────────────────────────
class KMedoids:
    def __init__(
        self,
        k: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = "k++",
        verbose: bool = False,
    ):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.verbose = verbose

    # ---------- fit (PAM) -------------------------------------
    def fit(self, X: np.ndarray):
        m = len(X)
        rng = np.random.default_rng()
        D = _pairwise_dists(X)
        # --- inicijalni medoidi ---
        if self.init == "random":
            meds = rng.choice(m, self.k, replace=False)
        else:  # k-means++ stil – daleki medoid
            meds = [rng.integers(m)]
            for _ in range(1, self.k):
                dist_to_closest = np.min(D[:, meds], axis=1)
                probs = dist_to_closest / dist_to_closest.sum()
                meds.append(rng.choice(m, p=probs))
        meds = np.array(meds)

        # --- Build+Swap petlja ---
        def _sse(labels):
            return np.sum(D[np.arange(m), meds[labels]])

        labels = np.argmin(D[:, meds], axis=1)
        best_sse = _sse(labels)
        if self.verbose:
            print(f"start SSE={best_sse:.2f}")

        improved = True
        it = 0
        while improved and it < self.max_iter:
            improved = False
            it += 1
            for mi in range(self.k):
                for h in range(m):
                    if h in meds:
                        continue
                    trial = meds.copy()
                    trial[mi] = h
                    lab = np.argmin(D[:, trial], axis=1)
                    sse = _sse(lab)
                    if sse + 1e-6 < best_sse:  # bolji
                        if self.verbose:
                            print(f"swap m{meds[mi]}→{h}: {best_sse:.2f}→{sse:.2f}")
                        meds, labels, best_sse = trial, lab, sse
                        improved = True
            if not improved or best_sse < self.tol:
                break

        self.medoids_ = meds
        self.labels_ = labels
        self.inertia_ = best_sse
        return self

    # ---------- predikcija -----------------------------------
    def predict(self, X: np.ndarray):
        D = np.linalg.norm(X[:, None] - X[self.medoids_], axis=2)
        return np.argmin(D, axis=1)


# ───────────────── vizuelni helperi ──────────────────────────
def plot_clusters_2d(model: KMedoids, X):
    assert X.shape[1] == 2, "2-D only"
    cmap = plt.cm.get_cmap("tab10", model.k)
    plt.figure(figsize=(4, 4))
    for cid in range(model.k):
        pts = X[model.labels_ == cid]
        plt.scatter(*pts.T, s=25, color=cmap(cid), label=f"cl {cid}")
        m = X[model.medoids_[cid]]
        plt.scatter(m[0], m[1], marker="X", s=120, c="black")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def elbow_plot(X, k_range=range(1, 8), **fit_kw):
    inert = []
    for k in k_range:
        model = KMedoids(k, init="k++", **fit_kw).fit(X)
        inert.append(model.inertia_)
    plt.figure()
    plt.plot(list(k_range), inert, "o-")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.title("Elbow")
    plt.grid(True)
    plt.show()
