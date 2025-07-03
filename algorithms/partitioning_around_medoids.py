from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Core utilities ----------
def manhattan(p: np.ndarray, q: np.ndarray) -> float:
    """L1 distanca između dvodimenzionih tačaka."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def total_cost(points: np.ndarray, medoid_idx: List[int]) -> float:
    """Ukupni trošak (sumu distanci do najbližeg medoida)."""
    cost = 0.0
    for i, p in enumerate(points):
        dists = [manhattan(p, points[m]) for m in medoid_idx]
        cost += min(dists)
    return cost


def assign_clusters(points: np.ndarray, medoid_idx: List[int]) -> List[int]:
    """Vraća indeks najbližeg medoida za svaku tačku."""
    clusters = []
    for p in points:
        dists = [manhattan(p, points[m]) for m in medoid_idx]
        clusters.append(int(np.argmin(dists)))
    return clusters


# ---------- PAM algorithm ----------
def pam_once(points: np.ndarray, medoid_idx: List[int]) -> Tuple[List[int], bool]:
    """
    Izvrši **jedan** swap-krug:
    • prolazi kroz sve nemedoidne tačke,
    • pokušava zamjenu sa svakim aktuelnim medoidom,
    • prihvata prvi swap koji smanjuje ukupni trošak.
    Vraća (novi_medoid_idx, was_improved).
    """
    best_cost = total_cost(points, medoid_idx)
    n = len(points)
    for mi, m in enumerate(medoid_idx):
        for h in range(n):
            if h in medoid_idx:  # već je medoid
                continue
            trial = medoid_idx.copy()
            trial[mi] = h  # zamijeni m → h
            c = total_cost(points, trial)
            if c < best_cost:  # pronađen bolji
                return trial, True
    return medoid_idx, False  # ništa bolje


def pam(
    points: np.ndarray, initial_medoid_idx: List[int], verbose: bool = True
) -> Tuple[List[int], float, List[List[int]]]:
    """Potpuni PAM do konvergencije (bez poboljšanja)."""
    medoids = initial_medoid_idx.copy()
    history = [medoids.copy()]
    if verbose:
        print(f"Start medoids: {medoids}, cost={total_cost(points, medoids):.2f}")
    improved = True
    while improved:
        medoids, improved = pam_once(points, medoids)
        if improved:
            history.append(medoids.copy())
            if verbose:
                print(f"  ↳ swap → {medoids}, cost={total_cost(points, medoids):.2f}")
    final_cost = total_cost(points, medoids)
    if verbose:
        print(f"Converged. Final cost = {final_cost:.2f}")
    return medoids, final_cost, history


# ---------- Visualisation ----------
def plot_clusters(points: np.ndarray, medoid_idx: List[int], title: str = "") -> None:
    clusters = assign_clusters(points, medoid_idx)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(4, 4))
    for i, p in enumerate(points):
        plt.scatter(
            p[0], p[1], color=colors[clusters[i]], marker="o", s=60, edgecolor="k"
        )
        plt.text(p[0] + 0.1, p[1] + 0.1, str(i + 1), fontsize=8)
    for m in medoid_idx:
        plt.scatter(
            points[m, 0], points[m, 1], color="black", marker="D", s=120, label="medoid"
        )
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
