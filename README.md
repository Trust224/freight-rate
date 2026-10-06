# Freight Rate Prediction

## Run
```bash
python -m pip install -r requirements.txt
python cv.py             # rolling-origin cross-validation
python train_predict.py  # writes validation_predictions.csv + december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```
Data files live in `data/` (train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv).

## Approach
- **Model:** gradient boosting (scikit-learn HistGradientBoostingRegressor), absolute-error loss, target = log(posted_rate), average of 5 seeds.
- **Split:** validation.csv is Nov-Dec 2025 and train is Jan-Oct, so this is a forecast. Rolling-origin CV with 3 expanding-window folds (test windows May-Jun, Jul-Aug, Sep-Oct), never training on the future.
- **Features:** log distance, |weight|, market_index, quote_signal, day of week, equipment, lat/lon of both ends. City names are NOT used, since 8 cities in validation never appear in train.

## Data quality
- Negative weights (292 train / 145 val): sign flips, fixed with abs().
- Missing weight (left as NaN; handled natively by the model) and missing market_index (filled from the same-date median, since it is date-level).
- ~1.3% of labels have an implausible $/mile (<1 or >5): excluded from training.
- Distances clipped at 70 miles; a few distances inconsistent with coordinates were flagged but kept.

## December chart
Chart inputs have no market_index / quote_signal, so they are filled with that date's median from validation.csv.
