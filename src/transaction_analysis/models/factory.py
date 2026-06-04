from enum import Enum

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


class ModelType(Enum):
    XGB = "xgb"
    RF = "rf"


def build_model(name: ModelType, cfg):
    """Factory — returns an unfitted sklearn-compatible estimator."""
    if name == ModelType.XGB:
        return xgb.XGBClassifier(**vars(cfg))
    if name == ModelType.RF:
        return RandomForestClassifier(**vars(cfg))
    raise ValueError(f"Unknown model: {name}")


def train_model(
    name: ModelType,
    cfg,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> object:
    """Train a model and return the fitted estimator."""
    model = build_model(name, cfg)

    fit_kwargs = {}
    if name == ModelType.XGB and X_val is not None and y_val is not None:
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}

    model.fit(X_train, y_train, **fit_kwargs)

    return model


def evaluate_model(model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Binary classification

    return {"y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}


def cross_validate_model(
    name: ModelType,
    cfg,
    cv_cfg,
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict]:
    cv = StratifiedKFold(n_splits=cv_cfg.n_splits, shuffle=True, random_state=cv_cfg.random_state)
    fold_results = []

    with tqdm(total=cv_cfg.n_splits, desc=f"CV {name.name}", unit="fold", leave=True) as pbar:
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = build_model(name, cfg)

            fit_kwargs = {}
            if name == ModelType.XGB:
                fit_kwargs = {"eval_set": [(X_test, y_test)], "verbose": False}

            model.fit(X_train, y_train, **fit_kwargs)

            fold_results.append(
                {
                    "fold": fold,
                    "model": model,
                    "y_test": y_test,
                    "y_pred": model.predict(X_test),
                    "y_proba": model.predict_proba(X_test)[:, 1],
                }
            )

            pbar.set_postfix(fold=fold + 1)
            pbar.update(1)

    return fold_results
