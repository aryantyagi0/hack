from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(self, estimator: BaseEstimator, features: List[str], target: str):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(
            df_train[self.features].values,
            df_train[self.target].values
        )

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )

        X_test = df_test[self.features].values
        y_true = df_test[self.target].values

        # Predictions
        y_pred = self.clf.predict(X_test)

        # Probabilities (for ROC-AUC)
        try:
            y_proba = self.clf.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_true, y_proba)
        except Exception:
            roc = -1

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc,
        }

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]
