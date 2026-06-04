from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset"

FRAUD_DATASET_DIR: Final[Path] = DATASET_DIR / "fraud-transactions"
PLOTS_DIR: Final[Path] = PROJECT_ROOT / "plots"

TRAINED_MODELS_DIR: Final[Path] = PROJECT_ROOT / "trained_models"
