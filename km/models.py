"""Detection and type models, and the baselines they are compared with."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import MODEL_FEATURES
from .labelling import TYPES


def fit_detector(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int = 0) -> XGBClassifier:
    """Gradient-boosted trees on the engineered features, with the positive class weighted up."""
    pos = max(int(y.sum()), 1)
    model = XGBClassifier(**cfg["model"]["xgb"], scale_pos_weight=(len(y) - pos) / pos, random_state=seed,
                          eval_metric="logloss", n_jobs=4, tree_method="hist")
    model.fit(X[MODEL_FEATURES], y)
    return model


class TypeModel:
    """Multi-class model for the moment type, trained on the positive windows only."""

    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.classes_: list[str] = []
        self.model = None

    def fit(self, X: pd.DataFrame, types: pd.Series) -> "TypeModel":
        known = types.isin(TYPES).to_numpy()
        X, types = X[known], types[known]
        self.classes_ = [t for t in TYPES if (types == t).any()]
        if len(self.classes_) > 1:
            y = types.map({t: i for i, t in enumerate(self.classes_)}).to_numpy()
            weights = (len(y) / (len(self.classes_) * np.bincount(y)))[y]
            self.model = XGBClassifier(**self.cfg["model"]["xgb"], random_state=self.seed, n_jobs=4,
                                       tree_method="hist", objective="multi:softprob")
            self.model.fit(X[MODEL_FEATURES], y, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.classes_:
            return np.full(len(X), "none", dtype=object)
        if self.model is None:
            return np.full(len(X), self.classes_[0], dtype=object)
        return np.array(self.classes_, dtype=object)[self.model.predict(X[MODEL_FEATURES])]


# Baselines. Each returns a score per window, higher meaning more likely a key moment.

def score_time_loss(windows: pd.DataFrame) -> np.ndarray:
    """Rank windows by the time lost against the reference lap."""
    return windows["time_loss"].fillna(0).to_numpy(float)


def score_labelling_functions(labels: pd.DataFrame) -> np.ndarray:
    """Use the labelling functions directly: the number of functions that fire."""
    return labels["n_fired"].to_numpy(float)


def score_random(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)
