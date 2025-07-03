from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# ───────────────────────── helpers ─────────────────────────────────
def train_test_split(X, y, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


# ───────────────────────── Decision Stump ──────────────────────────
class DecisionStump:
    def __init__(self):
        self.feat = None
        self.thr = None
        self.pol = 1

    def fit(self, X, y, w):
        """y must be ±1, w positive and sum=1."""
        m, d = X.shape
        best_err = 1.0
        for f in range(d):
            xs = X[:, f]
            thresholds = np.unique(xs)
            for t in thresholds:
                for pol in (1, -1):  # polarity: left<=thr => pol, else -pol
                    pred = np.where(xs <= t, pol, -pol)
                    err = w[(pred != y)].sum()
                    if err < best_err:
                        best_err, self.feat, self.thr, self.pol = err, f, t, pol
        return best_err

    def predict(self, X):
        return np.where(X[:, self.feat] <= self.thr, self.pol, -self.pol)


# ───────────────────────── AdaBoost ────────────────────────────────
class AdaBoostClassifier:
    def __init__(
        self, n_estimators: int = 50, learning_rate: float = 1.0, verbose: bool = False
    ):
        self.M = n_estimators
        self.lr = learning_rate
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert set(np.unique(y)) <= {1, -1}, "labels must be ±1"
        m = len(X)
        w = np.full(m, 1 / m)
        self.stumps_: list[DecisionStump] = []
        self.alpha_: list[float] = []

        for t in range(1, self.M + 1):
            stump = DecisionStump()
            err = stump.fit(X, y, w)
            err = max(err, 1e-9)  # avoid div/0
            alpha = 0.5 * np.log((1 - err) / err) * self.lr
            pred = stump.predict(X)
            # update weights
            w *= np.exp(-alpha * y * pred)
            w /= w.sum()

            self.stumps_.append(stump)
            self.alpha_.append(alpha)

            if self.verbose:
                print(
                    f"Iter {t:<3d}  feat={stump.feat}  thr={stump.thr:.3f}  "
                    f"pol={stump.pol:+}  err={err:.3f}  α={alpha:.3f}"
                )
        return self

    # inference
    def _agg_score(self, X):
        return sum(a * st.predict(X) for a, st in zip(self.alpha_, self.stumps_))

    def predict(self, X):
        return np.sign(self._agg_score(X)).astype(int)

    def predict_proba(self, X):
        """Return P(y=1) using logistic mapping of aggregate score."""
        score = self._agg_score(X)
        return 1 / (1 + np.exp(-2 * score))  # as in SAMME.R

    def score_accuracy(self, X, y):
        return accuracy(y, self.predict(X))


# ───────────────────────── visual helpers ──────────────────────────
def plot_alpha_curve(model: AdaBoostClassifier):
    plt.figure(figsize=(4, 3))
    plt.stem(range(1, len(model.alpha_) + 1), model.alpha_, basefmt=" ")
    plt.xlabel("iteration")
    plt.ylabel("alpha")
    plt.title("Weak-learner weights")
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, X, y, h=0.1, title=""):
    if X.shape[1] != 2:
        raise ValueError("2-D only")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Pastel2")
    markers = {1: "o", -1: "s"}
    for cls in (-1, 1):
        pts = X[y == cls]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            edgecolor="k",
            marker=markers[cls],
            label=f"class {cls}",
        )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
