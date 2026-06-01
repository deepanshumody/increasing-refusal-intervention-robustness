"""Unit tests for the LEACE and INLP post-hoc erasure baselines.

These use only NumPy + scikit-learn (no torch), and verify the defining
guarantees: LEACE equalises class-conditional means (so a linear probe falls
to chance), and INLP drives probe accuracy down from a separable baseline.
"""
import numpy as np

from intervention_robust_refusal.shared.erasure import (
    inlp_apply,
    inlp_fit,
    leace_apply,
    leace_fit,
)
from intervention_robust_refusal.shared.probes import linear_probe


def _separable_data(seed=0, n=400, d=20, shift=3.0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    X = rng.standard_normal((2 * n, d))
    X[y == 1, 0] += shift  # class signal lives on axis 0
    return X, y


def test_leace_equalises_class_means():
    X, y = _separable_data()
    mu, P = leace_fit(X, y)
    Xp = leace_apply(X, mu, P)
    gap = np.linalg.norm(Xp[y == 1].mean(0) - Xp[y == 0].mean(0))
    assert gap < 1e-6


def test_leace_collapses_linear_probe_to_chance():
    Xtr, ytr = _separable_data(seed=0)
    Xte, yte = _separable_data(seed=1)
    assert linear_probe(Xtr, ytr, Xte, yte) > 0.9  # separable before erasure
    mu, P = leace_fit(Xtr, ytr)
    acc = linear_probe(leace_apply(Xtr, mu, P), ytr, leace_apply(Xte, mu, P), yte)
    assert acc < 0.6  # near chance after erasure


def test_leace_preserves_shape():
    X, y = _separable_data()
    mu, P = leace_fit(X, y)
    assert leace_apply(X, mu, P).shape == X.shape


def test_inlp_reduces_probe_accuracy():
    Xtr, ytr = _separable_data(seed=0)
    Xte, yte = _separable_data(seed=1)
    before = linear_probe(Xtr, ytr, Xte, yte)
    P = inlp_fit(Xtr, ytr)
    after = linear_probe(inlp_apply(Xtr, P), ytr, inlp_apply(Xte, P), yte)
    assert before > 0.9
    assert after < 0.6  # driven near chance, not just marginally reduced
    # The projector genuinely removed directions rather than returning identity.
    assert np.linalg.matrix_rank(P, tol=1e-6) < Xtr.shape[1]
