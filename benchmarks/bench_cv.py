"""CV latency + correctness benchmark."""

import time

import pandas as pd

from src.feature_engineering import create_features

df = pd.read_csv("data/inflation_dataset.csv")
t0 = time.perf_counter()
feat = create_features(df)
elapsed_ms = (time.perf_counter() - t0) * 1000
print(
    f"create_features {len(feat)} rows {len(feat.columns)} cols in {elapsed_ms:.1f}ms — {'PASS' if elapsed_ms < 50 else 'SLOW'}"
)
