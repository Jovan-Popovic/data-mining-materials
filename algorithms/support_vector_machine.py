from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


# ───────────────────── helpers ──────────────────────────────────────
def train_test_split(X, y, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def _hinge_loss(w, b, X, y, C):
    margins = 1 - y * (X @ w + b)
    return 0.5 * np.dot(w, w) + C * np.maximum(0, margins).mean()


# ───────────────────── SVM class ────────────────────────────────────
class SVM:
    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        epochs: int = 500,
        batch: int | None = None,
        verbose: bool = False,
    ):
        """
        C       – regularisation (soft-margin).  large C → low regularisation
        lr      – learning rate
        epochs  – passes over data
        batch   – batch size (None = full batch) for SGD
        """
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.verbose = verbose

    # ----------------- fit -----------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """y must be +1 / −1"""
        assert set(np.unique(y)) <= {1, -1}, "labels must be ±1"
        m, n = X.shape
        rng = np.random.default_rng()
        self.w_ = rng.normal(scale=0.01, size=n)
        self.b_ = 0.0
        self.loss_history_ = []
        batch = self.batch or m
        lambda_ = 1 / (self.C * m)  # as in Pegasos

        for epoch in range(1, self.epochs + 1):
            idx = rng.permutation(m)
            for i in range(0, m, batch):
                j = idx[i : i + batch]
                Xb, yb = X[j], y[j]
                margin = yb * (Xb @ self.w_ + self.b_)
                mask = margin < 1
                grad_w = lambda_ * self.w_ - (self.C / len(j)) * (
                    (mask * yb)[:, None] * Xb
                ).sum(axis=0)
                grad_b = -(self.C / len(j)) * (mask * yb).sum()

                self.w_ -= self.lr * grad_w
                self.b_ -= self.lr * grad_b

            # track loss
            loss = _hinge_loss(self.w_, self.b_, X, y, self.C)
            self.loss_history_.append(loss)
            if self.verbose and epoch % max(1, self.epochs // 10) == 0:
                print(
                    f"epoch {epoch:<4d}  loss={loss:.4f}  |Δw|={np.linalg.norm(grad_w):.2e}"
                )

        return self

    # ----------------- inference -----------------------------------
    def decision_function(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score_accuracy(self, X, y):
        return (self.predict(X) == y).mean()


# ───────────────────── plots ───────────────────────────────────────
def plot_loss_curve(model: SVM):
    plt.figure(figsize=(4, 3))
    plt.plot(model.loss_history_)
    plt.xlabel("epoch")
    plt.ylabel("hinge loss")
    plt.title("SVM SGD loss")
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model: SVM, X: np.ndarray, y: np.ndarray, h: float = 0.1):
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary needs 2-D X")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Pastel2")
    markers = {1: "o", -1: "s"}
    for cls in (-1, 1):
        pts = X[y == cls]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            marker=markers[cls],
            edgecolor="k",
            label=f"class {cls}",
        )
    # margin lines
    w, b = model.w_, model.b_
    xs = np.linspace(x_min, x_max, 200)
    ys = -(w[0] * xs + b) / w[1]
    margin = 1 / np.linalg.norm(w)
    plt.plot(xs, ys, "k-", label="decision")
    plt.plot(xs, ys + margin, "k--", lw=0.8)
    plt.plot(xs, ys - margin, "k--", lw=0.8)
    plt.legend()
    plt.title("SVM decision boundary")
    plt.grid(True)
    plt.show()
