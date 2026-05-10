"""US map visualization for merchant transaction data."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transaction_analysis.eda.utils import save_figure

# Bundled US cities lat/lon lookup (avoids any external geocoding API).
# Source: simplemaps.com US Cities (public domain subset)
_US_CITIES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"
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


def _load_city_coords() -> pd.DataFrame:
    try:
        cities = pd.read_csv(_US_CITIES_URL, usecols=["City", "State", "lat", "lon"])
        cities["city_state"] = cities["City"].str.upper().str.strip() + "|" + cities["State"].str.upper().str.strip()
        return cities.set_index("city_state")[["lat", "lon"]]
    except Exception:
        return pd.DataFrame(columns=["lat", "lon"])


def _geocode(
    merchant_city: pd.Series,
    merchant_state: pd.Series,
    city_coords: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:

    key = merchant_city.str.upper().str.strip() + "|" + merchant_state.str.upper().str.strip()

    lats, lons = [], []
    for k, state in zip(key, merchant_state.str.upper().str.strip(), strict=False):
        if k in city_coords.index:
            row = city_coords.loc[k]
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
        else:
            center = _FALLBACK_STATE_CENTERS.get(state, (39.5, -98.4))
            lats.append(center[0])
            lons.append(center[1])

    return pd.Series(lats, index=merchant_city.index), pd.Series(lons, index=merchant_city.index)


def plot_us_map(
    transactions: pd.DataFrame,
    output_dir: Path,
    amount_col: str = "amount_cents_usd",
    city_col: str = "merchant_city",
    state_col: str = "merchant_state",
    top_n: int = 300,
) -> None:

    df = transactions.copy()
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df = df.dropna(subset=[city_col, state_col, amount_col])

    location_agg = (
        df.groupby([city_col, state_col], sort=False)
        .agg(
            txn_count=(amount_col, "count"),
            avg_amount=(amount_col, "mean"),
        )
        .reset_index()
        .sort_values("txn_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    city_coords = _load_city_coords()
    location_agg["lat"], location_agg["lon"] = _geocode(location_agg[city_col], location_agg[state_col], city_coords)

    location_agg = location_agg[(location_agg["lat"].between(24, 50)) & (location_agg["lon"].between(-125, -66))]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor("#f0f4f8")

    _draw_us_base(axes[0])
    _draw_us_base(axes[1])

    # LEFT — transaction volume
    sizes = _scale_bubbles(location_agg["txn_count"], min_size=20, max_size=800)
    sc0 = axes[0].scatter(
        location_agg["lon"],
        location_agg["lat"],
        s=sizes,
        c=location_agg["txn_count"],
        cmap="YlOrRd",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    fig.colorbar(sc0, ax=axes[0], label="Transaction Count", shrink=0.6, pad=0.02)
    axes[0].set_title("Transaction Volume by Merchant Location", fontsize=13, fontweight="bold", pad=12)

    # RIGHT — average amount
    sizes = _scale_bubbles(location_agg["avg_amount"], min_size=20, max_size=800)
    sc1 = axes[1].scatter(
        location_agg["lon"],
        location_agg["lat"],
        s=sizes,
        c=location_agg["avg_amount"],
        cmap="Blues",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    fig.colorbar(sc1, ax=axes[1], label="Avg Transaction Amount ($)", shrink=0.6, pad=0.02)
    axes[1].set_title("Average Transaction Amount by Merchant Location", fontsize=13, fontweight="bold", pad=12)

    # shared legend for bubble size
    for ax, series, label in [
        (axes[0], location_agg["txn_count"], "txns"),
        (axes[1], location_agg["avg_amount"], "$"),
    ]:
        _add_size_legend(ax, series, label)

    plt.tight_layout()
    save_figure("us_merchant_map.png", output_dir)
    plt.close()


def _draw_us_base(ax: plt.Axes) -> None:

    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        us = world[world["name"] == "United States of America"]
        us.plot(ax=ax, color="#e8e8e8", edgecolor="#aaaaaa", linewidth=0.6, zorder=1)
    except Exception:
        # Bare fallback: plain axes with lat/lon limits and a light background
        ax.set_facecolor("#dce9f5")
        ax.patch.set_alpha(0.4)

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, linewidth=0.5, zorder=0)


def _scale_bubbles(
    series: pd.Series,
    min_size: float = 20,
    max_size: float = 800,
) -> np.ndarray:
    vals = series.values.astype(float)
    sqrt_vals = np.sqrt(vals)
    lo, hi = sqrt_vals.min(), sqrt_vals.max()
    if hi == lo:
        return np.full(len(vals), (min_size + max_size) / 2)
    return min_size + (sqrt_vals - lo) / (hi - lo) * (max_size - min_size)


def _add_size_legend(ax: plt.Axes, series: pd.Series, unit: str) -> None:
    lo, hi = series.min(), series.max()
    mid = (lo + hi) / 2
    for val, label in [(lo, f"{lo:,.0f} {unit}"), (mid, f"{mid:,.0f} {unit}"), (hi, f"{hi:,.0f} {unit}")]:
        size = _scale_bubbles(pd.Series([lo, mid, hi, val]), 20, 800)[-1]
        ax.scatter([], [], s=size, color="gray", alpha=0.5, label=label, edgecolors="white")
    ax.legend(
        title="Bubble size",
        loc="lower left",
        fontsize=7,
        title_fontsize=8,
        framealpha=0.7,
        scatterpoints=1,
    )
