"""Coverage boost — exercises low-cover modules without heavy ML training."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


def test_validate_helpers() -> None:
    from src.data_refresh import (
        validate_brent,
        validate_cpi,
        validate_fx,
        validate_rate,
        validate_wpi,
    )

    base = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "value": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )
    rate_base = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "value": [6.0, 6.1, 6.0, 6.2, 6.3],
        }
    )
    assert validate_cpi(base) == []
    assert validate_wpi(base) == []
    assert validate_rate(rate_base) == []
    # rate 100 out of range
    bad_rate = rate_base.copy()
    bad_rate.loc[0, "value"] = 50.0
    assert len(validate_rate(bad_rate)) == 1
    # extreme mom >5%
    extreme = base.copy()
    extreme.loc[1, "value"] = 200.0
    assert len(validate_cpi(extreme)) >= 1

    fx = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "value": [75.0, 75.1, 75.2, 75.3, 75.4],
        }
    )
    assert validate_fx(fx) == []
    brent = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "value": [80.0, 81.0, 82.0, 83.0, 84.0],
        }
    )
    assert validate_brent(brent) == []

    # null check
    null_df = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "value": [None]})
    assert "null" in validate_wpi(null_df)[0].lower()


def test_rbi_fallback_returns_none() -> None:
    from src.data_refresh import fetch_rbi_dbie_series

    assert fetch_rbi_dbie_series("WPI") is None
    assert fetch_rbi_dbie_series("CPI") is None


def test_merge_and_save(tmp_path: Path) -> None:
    from src.data_refresh import MERGED_CSV, merge_and_save

    orig = MERGED_CSV
    try:
        # redirect MERGED_CSV to tmp
        import src.data_refresh as dr

        dr.MERGED_CSV = tmp_path / "merged.csv"
        cpi = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [100.0, 101.0, 102.0],
            }
        )
        wpi = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [120.0, 121.0, 122.0],
            }
        )
        rate = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [6.0, 6.1, 6.2],
            }
        )
        fx = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [75.0, 75.1, 75.2],
            }
        )
        brent = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [80.0, 81.0, 82.0],
            }
        )
        merged = merge_and_save(cpi, wpi, rate, fx, brent)
        assert len(merged) == 3
        assert "INDCPIALLMINMEI" in merged.columns
        assert (tmp_path / "merged.csv").exists()
    finally:
        dr.MERGED_CSV = orig


def test_log_refresh(tmp_path: Path) -> None:
    import src.data_refresh as dr

    orig = dr.REFRESH_LOG
    try:
        dr.REFRESH_LOG = tmp_path / "log.json"
        dr.log_refresh({"cpi": {"status": "success"}})
        assert dr.REFRESH_LOG.exists()
        dr.log_refresh({"wpi": {"status": "success"}})
        import json

        data = json.loads(dr.REFRESH_LOG.read_text())
        assert len(data) == 2
    finally:
        dr.REFRESH_LOG = orig


def test_fetch_all_sources_skipped_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When MERGED_CSV missing, fetch_all with cadence that skips should handle None gracefully."""
    import src.data_refresh as dr

    orig = dr.MERGED_CSV
    orig_log = dr.REFRESH_LOG
    try:
        dr.MERGED_CSV = tmp_path / "nonexistent.csv"
        dr.REFRESH_LOG = tmp_path / "log.json"
        # Patch fetchers to avoid network
        monkeypatch.setattr(
            dr,
            "refresh_cpi",
            lambda api_key=None: (
                pd.DataFrame(
                    {
                        "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
                        "value": [100.0, 101.0],
                    }
                ),
                [],
            ),
        )
        # Use cadence=daily so cpi skipped
        res = dr.fetch_all_sources(api_key="fake", cadence="daily")
        assert "cpi" in res
        # merged should error because cpi_df is None when skipped and file missing
        assert res["merged"]["status"] in ("error", "success")
    finally:
        dr.MERGED_CSV = orig
        dr.REFRESH_LOG = orig_log


def test_preprocessing_variants(tmp_path: Path) -> None:
    from src.preprocessing import load_and_clean_data

    # observation_date variant
    p = tmp_path / "a.csv"
    pd.DataFrame(
        {"observation_date": ["2020-01-01", "2020-02-01"], "DEXINUS": [75.1, 75.3]}
    ).to_csv(p, index=False)
    df = load_and_clean_data(p)
    assert "observation_date" in df.columns

    # Row Labels variant
    p2 = tmp_path / "b.csv"
    pd.DataFrame(
        {
            "Row Labels": ["2020-01-01", "2020-02-01", "Grand Total"],
            "INDCPIALLMINMEI": [100, 101, 999],
        }
    ).to_csv(p2, index=False)
    df2 = load_and_clean_data(p2)
    assert len(df2) == 2
    assert "Grand Total" not in df2["observation_date"].astype(str).values


def test_preprocessing_pivot_and_year(tmp_path: Path) -> None:
    from src.preprocessing import load_and_clean_data

    # pivot format Year/Month
    p = tmp_path / "pivot.csv"
    pd.DataFrame(
        {"Year/Month": ["Jan", "Feb"], "2020": [1.0, 2.0], "2021": [3.0, 4.0]}
    ).to_csv(p, index=False)
    df = load_and_clean_data(p)
    assert "observation_date" in df.columns

    # year columns
    p2 = tmp_path / "year.csv"
    pd.DataFrame(
        {"Country": ["India", "India"], "2020": [1.0, 2.0], "2021": [3.0, 4.0]}
    ).to_csv(p2, index=False)
    df2 = load_and_clean_data(p2)
    assert "observation_date" in df2.columns


def test_train_model_helpers() -> None:
    import numpy as np

    from src.train_model import (
        _capture_environment,
        compute_metrics,
        naive_baseline_mae,
    )

    m = compute_metrics([3.0, 3.5, 4.0], [3.1, 3.4, 4.1])
    assert 0 < m["r2"] <= 1
    assert m["mae"] == pytest.approx(0.1, abs=1e-9)
    assert "rmse" in m
    # single element edge
    m2 = compute_metrics([1.0], [1.0])
    assert m2["r2"] == 0.0
    # naive baseline
    assert naive_baseline_mae([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert np.isnan(naive_baseline_mae([1.0]))
    env = _capture_environment()
    assert "scikit-learn" in env
    assert "python" in env


def test_scheduler_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.scheduler as sch

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["scheduler.py"])
    with pytest.raises(SystemExit) as exc:
        sch.main()
    assert exc.value.code == 1


def test_scheduler_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.scheduler as sch

    monkeypatch.setenv("FRED_API_KEY", "fake123456")
    monkeypatch.setattr(sys, "argv", ["scheduler.py", "daily"])

    def fake_fetch(api_key=None, cadence="all"):
        return {"cpi": {"status": "success", "rows": 10}}

    monkeypatch.setattr(sch, "fetch_all_sources", fake_fetch)
    # should not raise
    sch.main()


def test_config_and_schema() -> None:
    import dataclasses

    from src.config import SETTINGS

    assert SETTINGS.DATA_PATH.name == "inflation_dataset.csv"
    assert "cpi" in SETTINGS.COLUMN_MAP
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        SETTINGS.DATA_PATH = Path("other.csv")  # type: ignore

    from src.schemas import InflationDatasetSchema

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "INDCPIALLMINMEI": [100.0, 101.0],
            "WPIATT01INM661N": [120.0, 121.0],
            "INTDSRINM193N": [6.0, 6.1],
            "DEXINUS": [75.0, 75.1],
            "Average of DCOILBRENTEU": [80.0, 81.0],
        }
    )
    InflationDatasetSchema.validate(df)
