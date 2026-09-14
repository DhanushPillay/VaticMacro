import pandas as pd

from src.schemas import InflationDatasetSchema


def test_monthly_unique_index_and_no_dup() -> None:
    df = pd.read_csv("data/inflation_dataset.csv", parse_dates=["Date"])
    # pandera validation — coerce Date, check INDCPIALLMINMEI >=0
    InflationDatasetSchema.validate(df)
    assert df["Date"].is_monotonic_increasing
    assert df["Date"].nunique() == len(df)


def test_cpi_not_flatline() -> None:
    df = pd.read_csv("data/inflation_dataset.csv", parse_dates=["Date"])
    # real data has variation
    assert (df["INDCPIALLMINMEI"].diff().abs() > 0).any()
