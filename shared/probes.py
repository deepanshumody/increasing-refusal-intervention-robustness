"""Linear and MLP probes — sklearn Pipelines matching the existing pipeline."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def linear_probe(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")),
    ]).fit(X_train, y_train)
    return float(pipe.score(X_test, y_test))


def mlp_probe(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                              max_iter=500, early_stopping=True,
                              validation_fraction=0.15, random_state=42)),
    ]).fit(X_train, y_train)
    return float(pipe.score(X_test, y_test))
