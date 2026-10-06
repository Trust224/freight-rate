"""Shared data cleaning + feature engineering (used by CV and the final run)."""
import numpy as np
import pandas as pd

EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
COORD_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
# Labels outside this $/mile band are treated as corrupt (see README, "Data quality").
RPM_MIN, RPM_MAX = 1.0, 5.0


def date_level_signals(*frames):
    """Per-date median market_index / quote_signal.

    market_index is a date-level market signal (between-day std ~0.17 vs ~0.025
    within a day), so missing values are filled from other loads on the same date.
    """
    allrows = pd.concat(frames)
    g = allrows.groupby("date")
    return g["market_index"].median(), g["quote_signal"].median()


def build_features(df, daily_mi):
    X = pd.DataFrame(index=df.index)
    X["logdist"] = np.log(df["distance"])
    X["weight"] = df["weight"].abs()  # negative weights are sign flips (|w| in 5k-47.5k)
    X["mi"] = df["market_index"].fillna(df["date"].map(daily_mi))
    X["qs"] = df["quote_signal"]
    X["dow"] = df["date"].dt.dayofweek
    X["equip"] = df["equipment"].map(EQUIP)
    for c in COORD_COLS:  # coordinates instead of city IDs -> works for unseen cities
        X[c] = df[c]
    return X


def clean_mask(train):
    rpm = train["posted_rate"] / train["distance"]
    return (rpm > RPM_MIN) & (rpm < RPM_MAX)
