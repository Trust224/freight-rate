"""Train the final model on all of train_test.csv and write both submission files.

Usage: python train_predict.py
Outputs: validation_predictions.csv, december_predictions.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import COORD_COLS, build_features, clean_mask, date_level_signals

SEEDS = [0, 1, 2, 3, 4]


def make_model(seed):
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=24,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )


def main():
    train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
    val = pd.read_csv("data/validation.csv", parse_dates=["date"])
    template = pd.read_csv("data/validation_predictions_template.csv")
    dec = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])

    daily_mi, daily_qs = date_level_signals(train, val)

    # 1) Fit on trimmed labels, log target, absolute-error loss (robust to bad labels).
    fit = train[clean_mask(train)]
    print(f"Training on {len(fit):,} of {len(train):,} rows after dropping corrupt labels")
    Xfit, yfit = build_features(fit, daily_mi), np.log(fit["posted_rate"])
    models = [make_model(s).fit(Xfit, yfit) for s in SEEDS]

    def predict(X):
        return np.exp(np.mean([m.predict(X) for m in models], axis=0))

    # 2) Validation predictions, written in template order.
    val["predicted_rate"] = predict(build_features(val, daily_mi)).round(2)
    out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and len(out) == 12_000
    out.to_csv("validation_predictions.csv", index=False)

    # 3) December chart: fixed route, only the date changes. Look up coordinates from
    #    train, and use that date's market_index / quote_signal medians from validation.csv.
    coords = {}
    for city in ["Lexington", "Fort Wayne"]:
        row = train[train["pickup"] == city].iloc[0]
        coords[city] = (row["pickup_lat"], row["pickup_lon"])
    d = dec.copy()
    d["pickup_lat"], d["pickup_lon"] = zip(*d["pickup"].map(coords))
    d["delivery_lat"], d["delivery_lon"] = zip(*d["delivery"].map(coords))
    d["market_index"] = d["date"].map(daily_mi)
    d["quote_signal"] = d["date"].map(daily_qs)
    assert d[["market_index", "quote_signal"]].notna().all().all()
    dec["predicted_rate"] = predict(build_features(d, daily_mi)).round(2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv("december_predictions.csv", index=False)
    print("Wrote validation_predictions.csv and december_predictions.csv")


if __name__ == "__main__":
    main()
