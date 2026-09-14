import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, RAW_COLS, create_features


def test_feature_columns_pinned_and_clipped() -> None:
    df = pd.read_csv("data/inflation_dataset.csv")
    feat = create_features(df)
    # FEATURE_COLUMNS must be exactly the engineered cols (excluding Date/CPI)
    assert len(FEATURE_COLUMNS) == 30
    assert set(FEATURE_COLUMNS).issubset(set(feat.columns))
    # clipping contract ±50
    pct_like = [c for c in feat.columns if "pct" in c or "lag_" in c or "rolling" in c]
    if pct_like:
        assert float(feat[pct_like].abs().max().max()) <= 50.0 + 1e-9


def test_no_raw_columns_remain() -> None:
    df = pd.read_csv("data/inflation_dataset.csv")
    feat = create_features(df)
    assert not any(c in feat.columns for c in RAW_COLS)


def test_create_features_drops_na_and_sorts() -> None:
    df = pd.read_csv("data/inflation_dataset.csv")
    feat = create_features(df)
    assert not feat.isna().any().any()
    assert pd.to_datetime(feat["Date"]).is_monotonic_increasing
