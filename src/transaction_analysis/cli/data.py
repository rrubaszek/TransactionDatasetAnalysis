from transaction_analysis.data import bootstrap, cleanup, preprocess
from transaction_analysis.eda.analysis import TransactionAnalysis
from transaction_analysis.paths import FRAUD_DATASET_DIR, PLOTS_DIR


def main() -> None:
    print("Downloading datasets...")
    bootstrap.run(force=False)

    print("Preprocessing data...")
    preprocess.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "raw", dataset_out_dir=FRAUD_DATASET_DIR / "preprocessed", force=True
    )
    print("Preprocessing complete.")

    print("Cleaning data...")
    cleanup.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "preprocessed",
        dataset_out_dir=FRAUD_DATASET_DIR / "cleaned",
        force=True,
    )
    print("Cleaning complete.")

    print("Running analysis...")
    TransactionAnalysis(dataset_dir=FRAUD_DATASET_DIR / "cleaned", output_dir=PLOTS_DIR).run(force=True)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
