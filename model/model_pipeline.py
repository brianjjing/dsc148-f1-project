#DNF PREDICTION WITH LIGHTGBM
#AS WELL AS A FIXED DNF LABEL.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


#Finished status:
FINISH_STATUSES = {"Finished"}
#Everything else (Accident, Engine, Gearbox, Collision, ...) is a DNF.

LAPPED_FINISH_PREFIX = "+" 
# +1 Lap, +2 Laps ... not a dnf, but the notebook right now treats it as a dnf.

# Statuses that will be dropped
NON_RACE_STATUSES = {
    "Did not qualify",
    "Did not prequalify",
    "Withdrew",
    "Not classified",
}


def _resolve_data_dir(start: Path | None = None) -> Path:
    """Find the data/ directory regardless of where the script is run from."""
    here = Path(start or __file__).resolve().parent
    for candidate in (here / "data", here.parent / "data", Path.cwd() / "data"):
        if (candidate / "results.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate data/results.csv")


def load_raw(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    data_dir = data_dir or _resolve_data_dir()
    return {
        "results": pd.read_csv(data_dir / "results.csv"),
        "status": pd.read_csv(data_dir / "status.csv"),
        "races": pd.read_csv(data_dir / "races.csv"),
        "drivers": pd.read_csv(data_dir / "drivers.csv"),
        "constructors": pd.read_csv(data_dir / "constructors.csv"),
    }


def build_labeled_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join results with status/race/driver metadata and attach `dnf` label.

    A row is labeled `dnf = 1` for any race-day non-finish (mechanical,
    accident, disqualification, etc.). Rows whose status is not a race-day
    outcome (DNQ, withdrew, ...) are dropped.
    """
    df = raw["results"].merge(raw["status"], on="statusId", how="left", validate="m:1")
    df = df.merge(
        raw["races"][["raceId", "year", "round", "circuitId", "date"]],
        on="raceId",
        how="left",
        validate="m:1",
    )
    df = df.merge(
        raw["drivers"][["driverId", "dob"]], on="driverId", how="left", validate="m:1"
    )

    df = df[~df["status"].isin(NON_RACE_STATUSES)].copy()

    is_finish = df["status"].isin(FINISH_STATUSES) | df["status"].str.startswith(
        LAPPED_FINISH_PREFIX, na=False
    )
    df["dnf"] = (~is_finish).astype(int)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25

    # `grid == 0` is the pit-lane start sentinel; treat as missing for modeling.
    df["grid"] = df["grid"].replace(0, np.nan)

    return df.sort_values("date").reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Driver- and constructor-level rolling DNF rates over prior races only.

    `.shift(1)` ensures the current race does not contribute to its own
    feature value. Sorting by date (done in `build_labeled_frame`) keeps
    the windows chronological even when raceId order is irregular.
    """
    out = df.copy()
    for group_col, prefix in [("driverId", "driver"), ("constructorId", "constructor")]:
        shifted = out.groupby(group_col)["dnf"].shift(1)
        for w in windows:
            col = f"{prefix}_dnf_rate_last{w}"
            out[col] = (
                shifted.groupby(out[group_col])
                .rolling(w, min_periods=2)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return out


# Pre-race features only — anything measured during/after the race leaks.
FEATURE_COLS: list[str] = [
    "grid",
    "year",
    "round",
    "driver_age",
    "driver_dnf_rate_last5",
    "driver_dnf_rate_last10",
    "constructor_dnf_rate_last5",
    "constructor_dnf_rate_last10",
]
CATEGORICAL_COLS: list[str] = ["circuitId", "constructorId", "driverId"]


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: list[str]
    categorical_cols: list[str]


def build_dataset(test_year_start: int = 2018, data_dir: Path | None = None) -> Dataset:
    raw = load_raw(data_dir)
    df = build_labeled_frame(raw)
    df = add_rolling_features(df)

    feature_cols = FEATURE_COLS + CATEGORICAL_COLS
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    train = df[df["year"] < test_year_start]
    test = df[df["year"] >= test_year_start]

    return Dataset(
        X_train=train[feature_cols],
        y_train=train["dnf"].astype(int),
        X_test=test[feature_cols],
        y_test=test["dnf"].astype(int),
        feature_cols=feature_cols,
        categorical_cols=CATEGORICAL_COLS,
    )


def make_model(y_train: pd.Series | None = None, **overrides) -> lgb.LGBMClassifier:
    """Initialized (untrained) LightGBM classifier with sensible defaults.

    If `y_train` is provided, `scale_pos_weight` is set from the class ratio
    so the model doesn't collapse to predicting the majority class.
    """
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if y_train is not None:
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        if pos > 0:
            params["scale_pos_weight"] = neg / pos
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


if __name__ == "__main__":
    data = build_dataset()
    print(f"Train: {data.X_train.shape}, DNF rate {data.y_train.mean():.3f}")
    print(f"Test:  {data.X_test.shape}, DNF rate {data.y_test.mean():.3f}")
    model = make_model(data.y_train)
    print(model)
