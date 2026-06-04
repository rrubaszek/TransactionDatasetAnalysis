import logging

import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from transaction_analysis.config.logger import setup_logging
from transaction_analysis.config.paths import TRAINED_MODELS_DIR
from transaction_analysis.models.compare import compare_models
from transaction_analysis.models.config import CrossValidationConfig, RFConfig, XGBConfig
from transaction_analysis.models.evaluate import print_report, summarise_cv
from transaction_analysis.models.factory import ModelType, cross_validate_model, train_model
from transaction_analysis.models.features import build_features

logger = logging.getLogger(__name__)


def run(force_cross_validate: bool = False, use_resample: bool = True):
    X, y = build_features()

    if use_resample:
        X_cv, y_cv = resample(X, y, n_samples=2_000_000, stratify=y, random_state=42)

    if force_cross_validate:
        xgb_results = cross_validate_model(
            ModelType.XGB, XGBConfig(max_depth=5), CrossValidationConfig(), X_cv.values, y_cv.values
        )
        rf_results = cross_validate_model(
            ModelType.RF, RFConfig(max_depth=5), CrossValidationConfig(), X_cv.values, y_cv.values
        )

        logger.info(summarise_cv(xgb_results).to_string())
        print_report(xgb_results, ModelType.XGB)
        logger.info(summarise_cv(rf_results).to_string())
        print_report(rf_results, ModelType.RF)

        logger.info(compare_models(xgb_results, rf_results).to_string())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = train_model(ModelType.XGB, XGBConfig(max_depth=5), X_train, y_train, X_test, y_test)
    rf_model = train_model(ModelType.RF, RFConfig(max_depth=5), X_train, y_train, X_test, y_test)

    # For xgb I used native version - apparently it loads faster and is smaller
    xgb_model.save_model(TRAINED_MODELS_DIR / "xgb_model.ubj")
    # Standard sklearn compatibile saving method
    joblib.dump(rf_model, TRAINED_MODELS_DIR / "rf_model.joblib")
    logger.info("Models trained and saved to disk.")


if __name__ == "__main__":
    setup_logging()
    run(force_cross_validate=True, use_resample=True)
