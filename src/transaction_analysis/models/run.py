from transaction_analysis.models.compare import compare_models
from transaction_analysis.models.config import RFConfig, XGBConfig
from transaction_analysis.models.evaluate import print_report, summarise_cv
from transaction_analysis.models.factory import cross_validate_model
from transaction_analysis.models.features import build_features


def run():
    # 1. Data loading + feature engineering (your existing pipeline, unchanged)
    X, y = build_features()  # extract from your current main()

    # 2. CV for both models
    xgb_results = cross_validate_model("xgb", XGBConfig(), X.values, y.values)
    rf_results = cross_validate_model("rf", RFConfig(), X.values, y.values)

    # 3. Per-model summaries
    print(summarise_cv(xgb_results).to_string())
    print_report(xgb_results, "XGBoost")
    print(summarise_cv(rf_results).to_string())
    print_report(rf_results, "RandomForest")

    # 4. Comparison
    print(compare_models(xgb_results, rf_results).to_string())


if __name__ == "__main__":
    run()
