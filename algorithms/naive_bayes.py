import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def naive_bayes_train(
    df: pd.DataFrame,
    target: str,
    categorical: List[str],
    numeric: List[str],
    laplace: float = 1.0,
) -> Dict:
    model = {"priors": {}, "cat": {}, "num": {}}
    classes = df[target].unique()
    n_total = len(df)
    for c in classes:
        subset = df[df[target] == c]
        model["priors"][c] = len(subset) / n_total
        for col in categorical:
            counts = subset[col].value_counts().to_dict()
            all_vals = df[col].unique()
            denom = len(subset) + laplace * len(all_vals)
            probs = {}
            for val in all_vals:
                probs[val] = (counts.get(val, 0) + laplace) / denom
            model["cat"].setdefault(col, {})[c] = probs
        for col in numeric:
            mu = subset[col].mean()
            var = subset[col].var(ddof=0)
            model["num"].setdefault(col, {})[c] = (mu, var)
    return model


def gaussian_pdf(x: float, mu: float, var: float) -> float:
    if var == 0:
        return 1.0 if x == mu else 1e-9
    coeff = 1 / math.sqrt(2 * math.pi * var)
    exponent = math.exp(-((x - mu) ** 2) / (2 * var))
    return coeff * exponent


def naive_bayes_predict(
    model: Dict, instance: Dict, categorical: List[str], numeric: List[str]
) -> Tuple[str, Dict[str, float]]:
    posteriors = {}
    for c, prior in model["priors"].items():
        log_prob = math.log(prior)
        for col in categorical:
            val = instance[col]
            log_prob += math.log(model["cat"][col][c].get(val, 1e-9))
        for col in numeric:
            mu, var = model["num"][col][c]
            log_prob += math.log(gaussian_pdf(instance[col], mu, var) + 1e-12)
        posteriors[c] = log_prob
    max_log = max(posteriors.values())
    exp_vals = {c: math.exp(lp - max_log) for c, lp in posteriors.items()}
    total = sum(exp_vals.values())
    probs = {c: v / total for c, v in exp_vals.items()}
    pred = max(probs, key=probs.get)
    return pred, probs


def plot_priors(model: Dict):
    plt.figure()
    classes = list(model["priors"].keys())
    priors = [model["priors"][c] for c in classes]
    plt.bar(classes, priors)
    plt.title("Prior probabilities")
    plt.ylabel("P(Class)")
    plt.show()
