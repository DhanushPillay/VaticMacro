"""Holdout benchmark — prints Ridge CV vs holdout vs naive."""

import json
import pathlib

import pandas as pd

from src.train_model import compute_metrics, naive_baseline_mae

metrics = json.loads(pathlib.Path("models/metrics.json").read_text())
hold = pd.read_csv("models/holdout.csv")
# Try target columns
y_true = (
    hold["target_future_inflation"]
    if "target_future_inflation" in hold.columns
    else hold.iloc[:, -2]
)
y_pred = (
    hold["Predicted_Inflation"]
    if "Predicted_Inflation" in hold.columns
    else hold.iloc[:, -1]
)
m = compute_metrics(y_true, y_pred)
naive = naive_baseline_mae(y_true)
ridge_cv = next(
    (x["r2_mean"] for x in metrics.get("metrics", []) if x.get("name") == "Ridge"),
    0.588,
)
print(
    f"Ridge CV R2={ridge_cv:.3f} | Holdout R2={m['r2']:.3f} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} | Naive MAE={naive:.3f}"
)
print("PASS" if m["r2"] > 0.3 and m["mae"] < naive else "FAIL")
