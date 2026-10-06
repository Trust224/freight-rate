"""Rolling-origin (expanding window) cross-validation. Usage: python cv.py"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build_features, clean_mask, date_level_signals

FOLDS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]


def metrics(y, p):
    return np.mean(np.abs(y - p)), np.mean(np.abs(y - p) / y) * 100


def main():
    train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
    val = pd.read_csv("data/validation.csv", parse_dates=["date"])
    daily_mi, _ = date_level_signals(train, val)
    train["clean"] = clean_mask(train)
    rows = []
    for start, end in FOLDS:
        a = train[(train.date < start) & train.clean]  # train only on the past
        b = train[(train.date >= start) & (train.date < end)]  # predict the next 2 months
        m = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=400, learning_rate=0.05,
            max_leaf_nodes=24, min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        ).fit(build_features(a, daily_mi), np.log(a.posted_rate))
        p = pd.Series(np.exp(m.predict(build_features(b, daily_mi))), index=b.index)
        rows.append([*metrics(b.posted_rate, p), *metrics(b.posted_rate[b.clean], p[b.clean])])
        print(f"{start} -> {end}: MAE all ${rows[-1][0]:.0f} | clean ${rows[-1][2]:.0f} (MAPE {rows[-1][3]:.2f}%)")
    r = np.mean(rows, axis=0)
    print(f"MEAN: MAE all ${r[0]:.0f} ({r[1]:.2f}%) | MAE clean ${r[2]:.0f} ({r[3]:.2f}%)")


if __name__ == "__main__":
    main()
