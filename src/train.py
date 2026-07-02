"""Model training for the KOI exoplanet classification project.

Trains three models of increasing capacity:
  1. Logistic Regression — linear baseline
  2. Random Forest — bagged trees
  3. XGBoost — gradient-boosted trees (primary model)

Class imbalance (FALSE POSITIVE ~51%, CONFIRMED ~29%, CANDIDATE ~21%) is
handled with class weighting rather than resampling: weighting keeps every
real observation, adds no synthetic points to a feature space full of
heavy-tailed physical quantities (where SMOTE-style interpolation can create
physically impossible objects), and is directly supported by all three models.

All models are evaluated with stratified 5-fold cross-validation on the
training set, scored on macro F1 (the primary metric, since accuracy would
reward simply predicting the majority FALSE POSITIVE class).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def encode_labels(y_train: pd.Series, y_test: pd.Series):
    """Encode string class labels to integers (needed by XGBoost).

    Returns (y_train_enc, y_test_enc, label_encoder).
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


def make_logistic_regression() -> Pipeline:
    """Linear baseline. Scaled inputs (regularized linear models need it),
    balanced class weights to counter the FALSE POSITIVE majority."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def xgb_sample_weights(y_enc: np.ndarray) -> np.ndarray:
    """XGBClassifier has no class_weight parameter for multi-class; emulate
    'balanced' weighting via per-sample weights."""
    return compute_sample_weight("balanced", y_enc)


def cross_validate_model(model, X, y, sample_weight=None) -> np.ndarray:
    """Stratified 5-fold macro-F1 scores for a model on the training set."""
    params = {}
    if sample_weight is not None:
        params = {"sample_weight": sample_weight}
    return cross_val_score(
        model, X, y, cv=CV, scoring="f1_macro", n_jobs=-1, params=params
    )


def tune_xgboost(X, y, sample_weight, n_iter: int = 20) -> RandomizedSearchCV:
    """Randomized hyperparameter search for XGBoost, optimizing macro F1.

    Randomized (not exhaustive grid) search: with 5 hyperparameters a full
    grid is combinatorially expensive, and random search finds comparable
    optima at a fraction of the cost.
    """
    param_distributions = {
        "n_estimators": [200, 300, 400, 600],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        make_xgboost(),
        param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=CV,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X, y, sample_weight=sample_weight)
    return search


def train_all(X_train, y_train_enc):
    """Fit all three models on the full training set.

    Returns dict name -> fitted model.
    """
    weights = xgb_sample_weights(y_train_enc)

    logreg = make_logistic_regression()
    logreg.fit(X_train, y_train_enc)

    rf = make_random_forest()
    rf.fit(X_train, y_train_enc)

    xgb = make_xgboost()
    xgb.fit(X_train, y_train_enc, sample_weight=weights)

    return {"Logistic Regression": logreg, "Random Forest": rf, "XGBoost": xgb}
