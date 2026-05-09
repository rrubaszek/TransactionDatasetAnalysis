from transaction_analysis import eda
from transaction_analysis.data import bootstrap, cleanup, preprocess
from transaction_analysis.paths import FRAUD_DATASET_DIR, PLOTS_DIR


def main() -> None:

    print("Downloading datasets...")
    bootstrap.run(force=False)

    print("Preprocessing data...")
    preprocess.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "raw", dataset_out_dir=FRAUD_DATASET_DIR / "preprocessed", force=False
    )

    print("Cleaning data...")
    cleanup.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "preprocessed",
        dataset_out_dir=FRAUD_DATASET_DIR / "preprocessed",
        force=False,
    )

    print("Preprocessing complete.")

    eda.main.run(dataset_dir=FRAUD_DATASET_DIR / "preprocessed", output_dir=PLOTS_DIR, force=False)


if __name__ == "__main__":
    main()
