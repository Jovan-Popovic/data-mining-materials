from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class PCA:
    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray):
        # center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt
        self.var_ = (S**2) / (len(X) - 1)
        self.explained_variance_ratio_ = self.var_ / self.var_.sum()
        if self.n_components:
            self.components_ = self.components_[: self.n_components]
            self.var_ = self.var_[: self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[
                : self.n_components
            ]
        return self

    def transform(self, X):  # forward projection
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Xp):  # back to original space
        return Xp @ self.components_ + self.mean_


# ─────────── plots ───────────
def scree_plot(pca: PCA):
    plt.figure(figsize=(4, 3))
    plt.bar(range(1, len(pca.var_) + 1), pca.explained_variance_ratio_)
    plt.xlabel("principal component")
    plt.ylabel("explained var ratio")
    plt.title("Scree plot")
    plt.grid(True)
    plt.show()


def scatter_pc(pca: PCA, X: np.ndarray, pc0: int = 0, pc1: int = 1, title=""):
    Xp = pca.transform(X)
    plt.figure(figsize=(4, 4))
    plt.scatter(Xp[:, pc0], Xp[:, pc1], s=25, edgecolor="k")
    plt.xlabel(f"PC{pc0+1}")
    plt.ylabel(f"PC{pc1+1}")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()
