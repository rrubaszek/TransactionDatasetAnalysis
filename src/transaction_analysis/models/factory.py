import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def build_model(name: str, cfg):
    """Factory — returns an unfitted sklearn-compatible estimator."""
    if name == "xgb":
        return xgb.XGBClassifier(**vars(cfg))
    if name == "rf":
        return RandomForestClassifier(**vars(cfg))
    raise ValueError(f"Unknown model: {name}")


def train_model(
    name: str, cfg, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None
) -> object:
    """Train a model and return the fitted estimator."""
    model = build_model(name, cfg)

    fit_kwargs = {}
    if name == "xgb" and X_val is not None and y_val is not None:
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def cross_validate_model(
    name: str,
    cfg,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(name, cfg)

        # XGBoost supports early stopping via eval_set; others ignore it cleanly
        fit_kwargs = {}
        if name == "xgb":
            fit_kwargs = {"eval_set": [(X_test, y_test)], "verbose": False}

        model.fit(X_train, y_train, **fit_kwargs)

        fold_results.append(
            {
                "fold": fold,
                "model": model,  # keep for feature importance later
                "y_test": y_test,
                "y_pred": model.predict(X_test),
                "y_proba": model.predict_proba(X_test)[:, 1],
            }
        )

    return fold_results
