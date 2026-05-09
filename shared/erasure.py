"""LEACE (closed-form oblique projection) and INLP (iterated nullspace projection)."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def leace_fit(X, y):
    """
    Belrose et al. 2023. Returns (mu, P) such that X' = (X - mu) @ P.T + mu has equal class-conditional means.
    Binary y assumed.
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
    W = U @ np.diag(1.0 / np.sqrt(S)) @ Vt
    Wp = U @ np.diag(np.sqrt(S)) @ Vt
    Y = np.zeros((X.shape[0], len(classes)))
    for i, c in enumerate(classes):
        Y[y == c, i] = 1.0
    Yc = Y - Y.mean(0, keepdims=True)
    Sxy = (Xc.T @ Yc) / n
    Uxy, _, _ = np.linalg.svd(W @ Sxy, full_matrices=False)
    P = np.eye(X.shape[1]) - Wp @ Uxy @ Uxy.T @ W
    return mu, P


def leace_apply(X, mu, P):
    return (X - mu) @ P.T + mu


def inlp_fit(X_train, y_train, n_iter=100, min_acc=0.55):
    """Iteratively project out the linear classifier's normal until train acc < min_acc."""
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


def inlp_apply(X, P):
    return X @ P.T
