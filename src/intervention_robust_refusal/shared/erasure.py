"""Post-hoc linear concept-erasure baselines: LEACE and INLP.

These are not part of the proposed training-time objective — they are *upper
bounds on linear erasure of frozen representations*, used in the sentiment
proof-of-concept to contextualise how much of the linear signal a training-
time loss leaves on the table relative to a projection that has full access
to the activations (paper §4.1).

  - LEACE (Belrose et al. 2023) is the closed-form oblique projection that
    minimally perturbs ``X`` (in the whitened metric) while equalising the
    class-conditional means. Theorem 3.1 of that paper guarantees a linear
    probe cannot exceed chance on the projected representations.
  - INLP (Ravfogel et al. 2020) iteratively trains a linear classifier on
    the current representations and projects out its normal until the
    classifier collapses to chance. Greedy and looser than LEACE.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def leace_fit(X: np.ndarray, y: np.ndarray):
    """Fit the LEACE projector. Returns ``(mu, P)`` such that
    ``X' = (X - mu) @ P.T + mu`` has equal class-conditional means.

    Binary ``y`` assumed. A small ridge (``1e-4 * trace(Σ)/d``) is added to
    the covariance before whitening for numerical stability.
    """
    classes = np.unique(y)
    assert len(classes) == 2
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    n = max(X.shape[0] - 1, 1)
    cov = (Xc.T @ Xc) / n
    eps = 1e-4 * np.trace(cov) / cov.shape[0]
    cov = cov + eps * np.eye(cov.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S)) @ Vt    # whitening
    Wp = U @ np.diag(np.sqrt(S)) @ Vt         # de-whitening (inverse of W)
    Y = np.zeros((X.shape[0], len(classes)))
    for i, c in enumerate(classes):
        Y[y == c, i] = 1.0
    Yc = Y - Y.mean(0, keepdims=True)
    Sxy = (Xc.T @ Yc) / n
    Uxy, _, _ = np.linalg.svd(W @ Sxy, full_matrices=False)
    P = np.eye(X.shape[1]) - Wp @ Uxy @ Uxy.T @ W
    return mu, P


def leace_apply(X: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
    return (X - mu) @ P.T + mu


def inlp_fit(X_train: np.ndarray, y_train: np.ndarray,
             n_iter: int = 100, min_acc: float = 0.55) -> np.ndarray:
    """Iteratively project out the linear classifier's normal until train accuracy
    drops below ``min_acc`` (chance-ish). Returns the composed projector ``P``.
    """
    d = X_train.shape[1]
    P = np.eye(d)
    Xc = X_train.copy()
    for _ in range(n_iter):
        clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1).fit(Xc, y_train)
        if clf.score(Xc, y_train) < min_acc:
            break
        w = clf.coef_[0]
        w = w / (np.linalg.norm(w) + 1e-12)
        Pw = np.eye(d) - np.outer(w, w)
        P = Pw @ P
        Xc = Xc @ Pw.T
    return P


def inlp_apply(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    return X @ P.T
