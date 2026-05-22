from transaction_analysis.data.step import bootstrap, cleanup, preprocess
from transaction_analysis.eda.step import analysis
from transaction_analysis.paths import FRAUD_DATASET_DIR, PLOTS_DIR


def main() -> None:
    print("Downloading datasets...")
    bootstrap.run(force=False)

    print("Preprocessing data...")
    preprocess.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "raw", dataset_out_dir=FRAUD_DATASET_DIR / "preprocessed", force=False
    )
    print("Preprocessing complete.")

    print("Cleaning data...")
    cleanup.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "preprocessed",
        dataset_out_dir=FRAUD_DATASET_DIR / "cleaned",
        force=False,
    )
    print("Cleaning complete.")

    print("Running analysis...")
    analysis.run(dataset_in_dir=FRAUD_DATASET_DIR / "cleaned", plots_out_dir=PLOTS_DIR, force=True)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
