from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── helpers ──────────────────────────────
def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((len(X), 1)), X]


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_test_split(X, y, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


# ───────────────────────── LogReg class ───────────────────────────
class LogReg:
    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 500,
        batch: int | None = None,
        l2: float = 0.0,  # λ (0 = no reg)
        verbose: bool = False,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.l2 = l2
        self.verbose = verbose

    # -------------------- fit --------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = _add_bias(X)
        rng = np.random.default_rng()
        m, n = Xb.shape
        self.theta_ = rng.normal(scale=0.01, size=n)
        self.loss_history_ = []
        batch = self.batch or m

        for epoch in range(1, self.epochs + 1):
            idx = rng.permutation(m)
            for i in range(0, m, batch):
                j = idx[i : i + batch]
                X_batch, y_batch = Xb[j], y[j]
                p = _sigmoid(X_batch @ self.theta_)
                grad = X_batch.T @ (p - y_batch) / len(j) + self.l2 * self.theta_
                self.theta_ -= self.lr * grad

            # log
            p_all = _sigmoid(Xb @ self.theta_)
            loss = (-y * np.log(p_all) - (1 - y) * np.log(1 - p_all)).mean() + (
                self.l2 / 2
            ) * (self.theta_ @ self.theta_)
            self.loss_history_.append(loss)
            if self.verbose and epoch % max(1, self.epochs // 10) == 0:
                print(
                    f"epoch {epoch:<4d}  loss={loss:.4f}  |Δθ|={np.linalg.norm(grad):.3e}"
                )
        return self

    # -------------------- inference --------------------------------
    def predict_proba(self, X):
        return _sigmoid(_add_bias(X) @ self.theta_)

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)

    def score_accuracy(self, X, y, thresh=0.5):
        return (self.predict(X, thresh) == y).mean()


# ──────────────────────────── plots ───────────────────────────────
def plot_loss_curve(model: LogReg):
    plt.figure(figsize=(4, 3))
    plt.plot(model.loss_history_)
    plt.xlabel("epoch")
    plt.ylabel("log-loss")
    plt.title("Gradient-descent loss")
    plt.grid(True)
    plt.show()


def plot_decision_boundary(
    model: LogReg, X: np.ndarray, y: np.ndarray, h: float = 0.1, cmap="Pastel2"
):
    """Only for 2-D feature space."""
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary requires exactly 2 features")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    for cls in (0, 1):
        pts = X[y == cls]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            edgecolor="k",
            label=f"class {cls}",
            marker="o" if cls == 0 else "s",
        )
    plt.legend()
    plt.title("Decision boundary")
    plt.grid(True)
    plt.show()
