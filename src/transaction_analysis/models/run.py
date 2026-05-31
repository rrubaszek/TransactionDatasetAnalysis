import logging

from transaction_analysis.config.logger import setup_logging
from transaction_analysis.models.compare import compare_models
from transaction_analysis.models.config import RFConfig, XGBConfig
from transaction_analysis.models.evaluate import print_report, summarise_cv
from transaction_analysis.models.factory import ModelType, cross_validate_model
from transaction_analysis.models.features import build_features

logger = logging.getLogger(__name__)


def run():
    X, y = build_features()

    xgb_results = cross_validate_model(ModelType.XGB, XGBConfig(), X.values, y.values)
    rf_results = cross_validate_model(ModelType.RF, RFConfig(), X.values, y.values)

    logger.info(summarise_cv(xgb_results).to_string())
    print_report(xgb_results, ModelType.XGB)
    logger.info(summarise_cv(rf_results).to_string())
    print_report(rf_results, ModelType.RF)

    logger.info(compare_models(xgb_results, rf_results).to_string())


if __name__ == "__main__":
    setup_logging()
    run()
