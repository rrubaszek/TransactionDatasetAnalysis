import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

_FALLBACK_STATE_CENTERS: dict[str, tuple[float, float]] = {
    "AL": (32.8, -86.8),
    "AK": (64.2, -153.4),
    "AZ": (34.3, -111.1),
    "AR": (34.8, -92.2),
    "CA": (36.8, -119.4),
    "CO": (39.0, -105.5),
    "CT": (41.6, -72.7),
    "DE": (39.0, -75.5),
    "FL": (27.8, -81.6),
    "GA": (32.2, -83.4),
    "HI": (20.3, -156.4),
    "ID": (44.4, -114.6),
    "IL": (40.0, -89.2),
    "IN": (40.3, -86.1),
    "IA": (42.0, -93.2),
    "KS": (38.5, -98.4),
    "KY": (37.5, -85.3),
    "LA": (31.2, -91.8),
    "ME": (45.4, -69.0),
    "MD": (39.0, -76.8),
    "MA": (42.3, -71.8),
    "MI": (44.3, -85.6),
    "MN": (46.4, -93.1),
    "MS": (32.7, -89.7),
    "MO": (38.5, -92.5),
    "MT": (47.0, -110.0),
    "NE": (41.5, -99.9),
    "NV": (39.5, -116.9),
    "NH": (43.7, -71.6),
    "NJ": (40.1, -74.5),
    "NM": (34.8, -106.2),
    "NY": (42.9, -75.5),
    "NC": (35.6, -79.8),
    "ND": (47.5, -100.5),
    "OH": (40.4, -82.8),
    "OK": (35.6, -96.9),
    "OR": (43.9, -120.6),
    "PA": (40.9, -77.8),
    "RI": (41.7, -71.5),
    "SC": (33.9, -80.9),
    "SD": (44.4, -100.2),
    "TN": (35.9, -86.4),
    "TX": (31.5, -99.3),
    "UT": (39.3, -111.1),
    "VT": (44.0, -72.7),
    "VA": (37.8, -78.2),
    "WA": (47.4, -120.6),
    "WV": (38.6, -80.6),
    "WI": (44.3, -89.8),
    "WY": (43.0, -107.6),
    "DC": (38.9, -77.0),
}


def load_cleaned_data(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions_path = dataset_dir / "transactions.parquet"

    transactions = pd.read_parquet(transactions_path)

    users_path = dataset_dir / "users.parquet"
    cards_path = dataset_dir / "cards.parquet"

    users = pd.read_parquet(users_path) if users_path.exists() else pd.DataFrame()
    cards = pd.read_parquet(cards_path) if cards_path.exists() else pd.DataFrame()

    return transactions, users, cards


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10


def save_figure(filename: str, output_dir: Path = Path("plots")) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filepath}")


def download_and_extract_zip(
    url: str,
    extract_dir: Path,
    force: bool = False,
    expected_file: str | None = None,
    timeout: int = 60,
) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)

    expected_path = extract_dir / expected_file if expected_file else extract_dir

    if expected_path.exists() and not force:
        print("Using existing shapefiles, skipping download. Use `force=True` to re-download.")
        return expected_path

    zip_path = extract_dir / "tmp_download.zip"

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    zip_path.unlink(missing_ok=True)

    return expected_path
