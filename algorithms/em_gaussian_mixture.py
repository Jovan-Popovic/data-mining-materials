from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ───────────── utils ─────────────
def train_test_split(X, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = int(len(X) * test_ratio)
    return X[idx[n:]], X[idx[:n]]


# ───────────── GMM class ─────────
class GMM:
    def __init__(self, k: int, max_iter: int = 100, tol: float = 1e-4, verbose=False):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _init_params(self, X):
        rng = np.random.default_rng()
        m, d = X.shape
        μ = X[rng.choice(m, self.k, replace=False)]
        σ2 = np.full(self.k, X.var())
        π = np.full(self.k, 1 / self.k)
        return π, μ, σ2

    def _e_step(self, X, π, μ, σ2):
        m, d = X.shape
        resp = np.zeros((m, self.k))
        for j in range(self.k):
            coef = (2 * np.pi * σ2[j]) ** (-d / 2)
            exp = np.exp(-np.sum((X - μ[j]) ** 2, axis=1) / (2 * σ2[j]))
            resp[:, j] = π[j] * coef * exp
        ll = np.log(resp.sum(axis=1)).sum()
        resp /= resp.sum(axis=1, keepdims=True)
        return resp, ll

    def _m_step(self, X, resp):
        m, d = X.shape
        Nk = resp.sum(axis=0)
        π = Nk / m
        μ = (resp.T @ X) / Nk[:, None]
        σ2 = np.array(
            [
                ((resp[:, j, None] * (X - μ[j]) ** 2).sum()) / (Nk[j] * d)
                for j in range(self.k)
            ]
        )
        return π, μ, σ2

    def fit(self, X: np.ndarray):
        π, μ, σ2 = self._init_params(X)
        self.log_likelihood_history_ = []
        for it in range(1, self.max_iter + 1):
            resp, ll = self._e_step(X, π, μ, σ2)
            π, μ, σ2 = self._m_step(X, resp)
            self.log_likelihood_history_.append(ll)
            if self.verbose:
                print(f"iter {it:<3d}  log-L={ll:.2f}")
            if it > 1 and abs(ll - self.log_likelihood_history_[-2]) < self.tol:
                break
        self.π_, self.μ_, self.σ2_ = π, μ, σ2
        self.resp_ = resp
        return self

    def predict(self, X):
        return self.resp_.argmax(axis=1)

    def predict_prob(self, X):
        π, μ, σ2 = self.π_, self.μ_, self.σ2_
        resp, _ = self._e_step(X, π, μ, σ2)
        return resp


# ───────────── plots ────────────
def plot_clusters_2d(X, labels, title=""):
    k = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", k)
    plt.figure(figsize=(4, 4))
    for j in range(k):
        plt.scatter(*X[labels == j].T, s=30, color=cmap(j), label=f"comp {j}")
    plt.legend()
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_ll_curve(model: GMM):
    plt.figure(figsize=(4, 3))
    plt.plot(model.log_likelihood_history_)
    plt.xlabel("iter")
    plt.ylabel("log-likelihood")
    plt.title("EM convergence")
    plt.grid(True)
    plt.show()
