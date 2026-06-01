"""Linear and MLP probes used to read off class-conditional information.

A probe's held-out accuracy is the quantitative analogue of "how easy is it
to recover this concept from hidden states?" — the lower the probe accuracy
the more thoroughly the matching loss has redistributed the class signal.
Both probes standardise features first; the linear probe is L2-regularised
logistic regression with balanced class weights, and the MLP is a small
two-hidden-layer net with early stopping.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def linear_probe(X_train, y_train, X_test, y_test) -> float:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")),
    ]).fit(X_train, y_train)
    return float(pipe.score(X_test, y_test))


def mlp_probe(X_train, y_train, X_test, y_test) -> float:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                              max_iter=500, early_stopping=True,
                              validation_fraction=0.15, random_state=42)),
    ]).fit(X_train, y_train)
    return float(pipe.score(X_test, y_test))
