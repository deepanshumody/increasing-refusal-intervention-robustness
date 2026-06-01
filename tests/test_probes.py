"""Unit tests for the linear and MLP probes."""
import numpy as np

from intervention_robust_refusal.shared.probes import linear_probe, mlp_probe


def _separable(seed, n=200, d=10, shift=4.0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    X = rng.standard_normal((2 * n, d))
    X[y == 1, 0] += shift
    return X, y


def test_linear_probe_separates_clean_signal():
    Xtr, ytr = _separable(0)
    Xte, yte = _separable(1)
    assert linear_probe(Xtr, ytr, Xte, yte) > 0.95


def test_linear_probe_near_chance_on_noise():
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.standard_normal((300, 10)), rng.standard_normal((300, 10))
    ytr = rng.integers(0, 2, 300)
    yte = rng.integers(0, 2, 300)
    acc = linear_probe(Xtr, ytr, Xte, yte)
    assert 0.35 < acc < 0.65


def test_mlp_probe_returns_valid_accuracy():
    Xtr, ytr = _separable(0)
    Xte, yte = _separable(1)
    acc = mlp_probe(Xtr, ytr, Xte, yte)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.9  # clean signal should be easily learnable
