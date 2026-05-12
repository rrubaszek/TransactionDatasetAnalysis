from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from transaction_analysis.eda.utils import (
    _FALLBACK_STATE_CENTERS,
    download_and_extract_zip,
)
from transaction_analysis.paths import FRAUD_DATASET_DIR

_US_CITIES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"

_NATURAL_EARTH_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

_DATA_DIR = FRAUD_DATASET_DIR / "geo" / "zip"

_SHAPEFILE_NAME = "ne_110m_admin_0_countries.shp"

_FALLBACK_US_CENTER = (39.5, -98.4)


def ensure_us_shapefile(force: bool = False) -> Path:
    return download_and_extract_zip(
        url=_NATURAL_EARTH_URL,
        extract_dir=_DATA_DIR,
        force=force,
        expected_file=_SHAPEFILE_NAME,
    )


def load_us_geometry(force_download: bool = False) -> gpd.GeoDataFrame:
    shapefile = ensure_us_shapefile(force_download)

    world = gpd.read_file(shapefile)

    return world[world["ADMIN"] == "United States of America"]


def load_city_coordinates() -> pd.DataFrame:
    try:
        cities = pd.read_csv(
            _US_CITIES_URL,
            usecols=["City", "State", "lat", "lon"],
        )

        cities["key"] = cities["City"].str.upper().str.strip() + "|" + cities["State"].str.upper().str.strip()

        return cities.set_index("key")[["lat", "lon"]]

    except Exception:
        return pd.DataFrame(columns=["lat", "lon"])


def geocode_us_locations(
    city_series: pd.Series,
    state_series: pd.Series,
    city_coords: pd.DataFrame,
) -> pd.DataFrame:
    keys = city_series.str.upper().str.strip() + "|" + state_series.str.upper().str.strip()

    latitudes = []
    longitudes = []

    for key, state in zip(
        keys,
        state_series.str.upper().str.strip(),
        strict=False,
    ):
        if key in city_coords.index:
            row = city_coords.loc[key]

            latitudes.append(float(row["lat"]))
            longitudes.append(float(row["lon"]))

        else:
            lat, lon = _FALLBACK_STATE_CENTERS.get(
                state,
                _FALLBACK_US_CENTER,
            )

            latitudes.append(lat)
            longitudes.append(lon)

    return pd.DataFrame(
        {
            "lat": latitudes,
            "lon": longitudes,
        },
        index=city_series.index,
    )


def prepare_us_transaction_geo_data(
    transactions: pd.DataFrame,
    amount_col: str,
    city_col: str,
    state_col: str,
    top_n: int = 300,
) -> pd.DataFrame:

    df = transactions.copy()

    df[amount_col] = pd.to_numeric(
        df[amount_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            city_col,
            state_col,
            amount_col,
        ]
    )

    aggregated = (
        df.groupby([city_col, state_col], sort=False)
        .agg(
            txn_count=(amount_col, "count"),
            avg_amount=(amount_col, "mean"),
        )
        .reset_index()
        .sort_values(
            "txn_count",
            ascending=False,
        )
        .head(top_n)
    )

    city_coords = load_city_coordinates()

    geo = geocode_us_locations(aggregated[city_col], aggregated[state_col], city_coords)

    aggregated["lat"] = geo["lat"]
    aggregated["lon"] = geo["lon"]

    aggregated = aggregated[aggregated["lat"].between(24, 50) & aggregated["lon"].between(-125, -66)]

    return aggregated.reset_index(drop=True)


def scale_bubbles(
    values: pd.Series,
    min_size: float = 20,
    max_size: float = 800,
) -> np.ndarray:

    vals = values.astype(float).values

    sqrt_vals = np.sqrt(vals)

    lo = sqrt_vals.min()
    hi = sqrt_vals.max()

    if hi == lo:
        return np.full(len(vals), (min_size + max_size) / 2)

    return min_size + (sqrt_vals - lo) / (hi - lo) * (max_size - min_size)
