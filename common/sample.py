"""Stratified sampling."""
import numpy as np


def stratified_indices(y, n, seed=0):
    y = np.asarray(y)
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        k = max(1, round(n * len(rows) / len(y)))
        picked.append(rng.choice(rows, size=min(k, len(rows)), replace=False))
    return np.sort(np.concatenate(picked))


def stratified_sample(X, y, n, seed=0):
    idx = stratified_indices(y, n, seed)
    return np.asarray(X[idx]), np.asarray(y[idx])
