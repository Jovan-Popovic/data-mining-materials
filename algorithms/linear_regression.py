from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── helpers ────────────────────────────────
def _add_bias(X: np.ndarray) -> np.ndarray:
    """Dodaje kolonu jedinica (bias) i vraća novu matricu."""
    return np.c_[np.ones((len(X), 1)), X]


def train_test_split(X, y, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


# ─────────────────────────── LinearReg ──────────────────────────────
class LinearReg:
    def __init__(
        self,
        method: str = "normal",  # "normal" | "gd"
        lr: float = 0.01,
        epochs: int = 200,
        batch: int | None = None,  # None = full GD
        verbose: bool = False,
    ):
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.verbose = verbose

    # --------------- fit ---------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = _add_bias(X)
        if self.method == "normal":
            self.theta_ = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
            self.loss_history_ = []
        else:
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
                    grad = 2 / len(j) * X_batch.T @ (X_batch @ self.theta_ - y_batch)
                    self.theta_ -= self.lr * grad
                loss = np.mean((Xb @ self.theta_ - y) ** 2)
                self.loss_history_.append(loss)
                if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                    print(f"epoch {epoch:<4d}  loss={loss:.4f}")
        return self

    # --------------- predict / score --------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return _add_bias(X) @ self.theta_

    def score_R2(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot


# ─────────────────────────── plots ──────────────────────────────────
def plot_2d_regression(X, y, model: LinearReg, title=""):
    """Radi SAMO kad X ima JEDAN feature kolonu."""
    if X.shape[1] != 1:
        raise ValueError("plot_2d_regression expects exactly one feature")
    plt.figure(figsize=(4, 3))
    plt.scatter(X[:, 0], y, edgecolor="k", label="data")
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
    plt.plot(xs, model.predict(xs), c="red", label="fit")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_slice(
    model: LinearReg,
    X: np.ndarray,
    y: np.ndarray,
    vary_index: int = 0,
    const_vals: Dict[int, float] | None = None,
    title: str = "",
):
    """
    Crta 'presjek' višedimenzionalnog modela:
      • vary_index – feature koji se mijenja (0-based)
      • const_vals – dict {feature_idx: konstanta}; ostale na prosjeku
    """
    d = X.shape[1]
    if vary_index >= d:
        raise ValueError("vary_index out of range")
    base = X.mean(axis=0)
    if const_vals:
        for i, v in const_vals.items():
            base[i] = v

    xs = np.linspace(X[:, vary_index].min(), X[:, vary_index].max(), 200)
    grid = np.tile(base, (len(xs), 1))
    grid[:, vary_index] = xs
    y_hat = model.predict(grid)

    plt.figure(figsize=(4, 3))
    plt.scatter(X[:, vary_index], y, edgecolor="k", label="data")
    plt.plot(xs, y_hat, "r", label="model slice")
    plt.xlabel(f"Feature {vary_index}")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_curve(model: LinearReg):
    if not model.loss_history_:
        print("No loss history (likely trained with normal equation).")
        return
    plt.figure(figsize=(4, 3))
    plt.plot(model.loss_history_)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Gradient-descent loss")
    plt.grid(True)
    plt.show()
