"""Full coverage for src/data_refresh — hits fetch happy paths, validates, refreshes, and fetch_all cadences."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.data_refresh as dr


# ------------------------------------------------------------------ fetch_fred_series
def test_fetch_fred_success_filters_null_and_dot() -> None:
    payload = {
        "observations": [
            {"date": "2020-01-01", "value": "100.5"},
            {"date": "2020-02-01", "value": "NULL"},
            {"date": "2020-03-01", "value": "."},
            {"date": "2020-04-01", "value": "not_a_number"},
            {"date": "2020-05-01", "value": "102.3"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None
    with patch("src.data_refresh.requests.get", return_value=mock_resp) as mg:
        df = dr.fetch_fred_series("FAKE", api_key="key123")
    mg.assert_called_once()
    assert len(df) == 2  # only 100.5 and 102.3 survive
    assert list(df["value"]) == [100.5, 102.3]
    # sorted
    assert df.iloc[0]["date"] < df.iloc[1]["date"]


def test_fetch_fred_no_observations_raises() -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"foo": 1}
    mock_resp.raise_for_status.return_value = None
    with (
        patch("src.data_refresh.requests.get", return_value=mock_resp),
        pytest.raises(ValueError, match="No observations"),
    ):
        dr.fetch_fred_series("BAD", api_key="k")


def test_fetch_fred_empty_after_filter_raises() -> None:
    payload = {"observations": [{"date": "2020-01-01", "value": "NULL"}]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None
    with (
        patch("src.data_refresh.requests.get", return_value=mock_resp),
        pytest.raises(ValueError, match="No valid data"),
    ):
        dr.fetch_fred_series("EMPTY", api_key="k")


def test_fetch_fred_request_exception_wraps() -> None:
    import requests

    with (
        patch(
            "src.data_refresh.requests.get",
            side_effect=requests.exceptions.ConnectionError("boom"),
        ),
        pytest.raises(RuntimeError, match="FRED API error"),
    ):
        dr.fetch_fred_series("X", api_key="k")


# ---------------------------------------------------------------- validates
def test_validate_all_branches() -> None:
    # cpi out of range low/high
    low = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
            "value": [10.0, 11.0],
        }
    )
    assert any("range" in s for s in dr.validate_cpi(low))
    high = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
            "value": [300.0, 310.0],
        }
    )
    assert any("range" in s for s in dr.validate_cpi(high))
    # cpi null
    null = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "value": [float("nan")]})
    assert any("null" in s.lower() for s in dr.validate_cpi(null))

    # wpi non-positive
    wpi_bad = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=2, freq="ME"), "value": [0.0, 5.0]}
    )
    assert any("non-positive" in s for s in dr.validate_wpi(wpi_bad))
    assert any("null" in s.lower() for s in dr.validate_wpi(null))
    assert (
        dr.validate_wpi(
            pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
                    "value": [120.0, 121.0],
                }
            )
        )
        == []
    )

    # rate null + range
    assert any("null" in s.lower() for s in dr.validate_rate(null))
    bad_rate = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
            "value": [25.0, 26.0],
        }
    )
    assert any("range" in s for s in dr.validate_rate(bad_rate))

    # fx null + extreme >2%
    assert any("null" in s.lower() for s in dr.validate_fx(null))
    fx_extreme = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "value": [70.0, 80.0],
        }
    )  # ~14%
    assert any("extreme" in s for s in dr.validate_fx(fx_extreme))
    assert (
        dr.validate_fx(
            pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=2, freq="D"),
                    "value": [75.0, 75.1],
                }
            )
        )
        == []
    )

    # brent null + extreme >10%
    assert any("null" in s.lower() for s in dr.validate_brent(null))
    brent_extreme = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "value": [50.0, 80.0],
        }
    )  # 60%
    assert any("extreme" in s for s in dr.validate_brent(brent_extreme))
    assert (
        dr.validate_brent(
            pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=2, freq="D"),
                    "value": [80.0, 81.0],
                }
            )
        )
        == []
    )


# ---------------------------------------------------------------- refresh_cpi
def test_refresh_cpi_appends_new_month(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cpi_path = tmp_path / "data" / "INDCPIALLMINMEI (Consumer Price Index).csv"
    # existing up to Jan 2020
    pd.DataFrame(
        {"observation_date": ["2020-01-31"], "INDCPIALLMINMEI": [100.0]}
    ).to_csv(cpi_path, index=False)

    html = b'<table class="table table-hover"><tr><th>Inflation Rate MoM</th><td>2.0</td><td>x</td><td>y</td><td>Feb 2020</td></tr></table>'

    class FakeResp:
        def read(self) -> bytes:
            return html

    monkeypatch.setattr(urllib.request, "urlopen", lambda req: FakeResp())
    df, _issues = dr.refresh_cpi()
    assert len(df) == 2
    assert df["value"].iloc[-1] == pytest.approx(102.0)
    # file was updated
    saved = pd.read_csv(cpi_path)
    assert len(saved) == 2


def test_refresh_cpi_up_to_date_no_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cpi_path = tmp_path / "data" / "INDCPIALLMINMEI (Consumer Price Index).csv"
    pd.DataFrame(
        {"observation_date": ["2020-05-31"], "INDCPIALLMINMEI": [110.0]}
    ).to_csv(cpi_path, index=False)
    html = b'<table class="table table-hover"><tr><th>Inflation Rate MoM</th><td>0.5</td><td>x</td><td>y</td><td>Apr 2020</td></tr></table>'

    class FakeResp:
        def read(self) -> bytes:
            return html

    monkeypatch.setattr(urllib.request, "urlopen", lambda req: FakeResp())
    df, _ = dr.refresh_cpi()
    assert len(df) == 1  # no append because Apr < May


def test_refresh_cpi_handles_fetch_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cpi_path = tmp_path / "data" / "INDCPIALLMINMEI (Consumer Price Index).csv"
    pd.DataFrame(
        {"observation_date": ["2020-01-31"], "INDCPIALLMINMEI": [100.0]}
    ).to_csv(cpi_path, index=False)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req: (_ for _ in ()).throw(RuntimeError("net down")),
    )
    df, _ = dr.refresh_cpi()
    assert len(df) == 1  # returns original, prints warning


# ---------------------------------------------------------------- refresh_* wrappers
def test_refresh_wpi_success_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
            "value": [120.0, 121.0],
        }
    )
    monkeypatch.setattr(dr, "fetch_fred_series", lambda sid, api_key=None: fake_df)
    df, _issues = dr.refresh_wpi(api_key="k")
    assert len(df) == 2

    # fallback path: FRED fails, RBI returns data
    def failing_fetch(*a, **kw) -> pd.DataFrame:
        raise RuntimeError("fred down")

    monkeypatch.setattr(dr, "fetch_fred_series", failing_fetch)
    monkeypatch.setattr(dr, "fetch_rbi_dbie_series", lambda ind="WPI": fake_df)
    df2, _ = dr.refresh_wpi(api_key="k")
    assert len(df2) == 2

    # both fail -> raise
    monkeypatch.setattr(dr, "fetch_rbi_dbie_series", lambda ind="WPI": None)
    with pytest.raises(RuntimeError):
        dr.refresh_wpi(api_key="k")


def test_refresh_rate_fx_brent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=2, freq="ME"), "value": [6.0, 6.1]}
    )
    monkeypatch.setattr(dr, "fetch_fred_series", lambda sid, api_key=None: fake)
    for fn in (dr.refresh_rate, dr.refresh_fx, dr.refresh_brent):
        df, _ = fn(api_key="k")
        assert len(df) == 2


# ---------------------------------------------------------------- log_refresh branches
def test_log_refresh_handles_dict_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orig = dr.REFRESH_LOG
    try:
        dr.REFRESH_LOG = tmp_path / "log.json"
        # pre-seed as dict (old format)
        import json

        (tmp_path / "log.json").write_text(json.dumps({"old": "entry"}))
        dr.log_refresh({"cpi": {"status": "success"}})
        data = json.loads((tmp_path / "log.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0] == {"old": "entry"}
    finally:
        dr.REFRESH_LOG = orig


# ---------------------------------------------------------------- _load_existing_series via fetch_all_sources
def test_load_existing_series_all_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orig_csv = dr.MERGED_CSV
    orig_log = dr.REFRESH_LOG
    try:
        dr.MERGED_CSV = tmp_path / "merged.csv"
        dr.REFRESH_LOG = tmp_path / "log.json"
        merged = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
                "INDCPIALLMINMEI": [100.0, 101.0, 102.0],
                "WPIATT01INM661N": [120.0, 121.0, 122.0],
                "INTDSRINM193N": [6.0, 6.1, 6.2],
                "DEXINUS": [75.0, 75.1, 75.2],
                "Average of DCOILBRENTEU": [80.0, 81.0, 82.0],
            }
        )
        merged.to_csv(dr.MERGED_CSV, index=False)

        # force skipped cadences to exercise _load_existing_series
        # stub refreshers so they are not called
        monkeypatch.setattr(
            dr,
            "refresh_cpi",
            lambda api_key=None: (_ for _ in ()).throw(
                AssertionError("should not be called")
            ),
        )
        monkeypatch.setattr(
            dr,
            "refresh_wpi",
            lambda api_key=None: (_ for _ in ()).throw(
                AssertionError("should not be called")
            ),
        )
        # use cadence=daily so monthly are skipped and loaded from file
        fake_fx = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [75.0, 75.1, 75.2],
            }
        )
        fake_brent = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [80.0, 81.0, 82.0],
            }
        )
        monkeypatch.setattr(dr, "refresh_fx", lambda api_key=None: (fake_fx, []))
        monkeypatch.setattr(dr, "refresh_brent", lambda api_key=None: (fake_brent, []))

        res = dr.fetch_all_sources(api_key="k", cadence="daily")
        assert res["cpi"]["status"] == "skipped"
        assert res["wpi"]["status"] == "skipped"
        assert res["rate"]["status"] == "skipped"
        assert res["fx"]["status"] == "success"
        assert res["merged"]["status"] == "success"
        assert res["merged"]["rows"] == 3

        # now cadence=monthly so daily are skipped
        monkeypatch.setattr(
            dr,
            "refresh_fx",
            lambda api_key=None: (_ for _ in ()).throw(
                AssertionError("should not be called")
            ),
        )
        monkeypatch.setattr(
            dr,
            "refresh_brent",
            lambda api_key=None: (_ for _ in ()).throw(
                AssertionError("should not be called")
            ),
        )
        fake_cpi = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [100.0, 101.0, 102.0],
            }
        )
        fake_wpi = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [120.0, 121.0, 122.0],
            }
        )
        fake_rate = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="ME"),
                "value": [6.0, 6.1, 6.2],
            }
        )
        monkeypatch.setattr(dr, "refresh_cpi", lambda api_key=None: (fake_cpi, []))
        monkeypatch.setattr(dr, "refresh_wpi", lambda api_key=None: (fake_wpi, []))
        monkeypatch.setattr(dr, "refresh_rate", lambda api_key=None: (fake_rate, []))
        res2 = dr.fetch_all_sources(api_key="k", cadence="monthly")
        assert res2["fx"]["status"] == "skipped"
        assert res2["brent"]["status"] == "skipped"
        assert res2["merged"]["status"] == "success"
    finally:
        dr.MERGED_CSV = orig_csv
        dr.REFRESH_LOG = orig_log


def test_fetch_all_sources_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orig_csv = dr.MERGED_CSV
    orig_log = dr.REFRESH_LOG
    try:
        dr.MERGED_CSV = (
            tmp_path / "merged.csv"
        )  # missing file -> _load returns None but fetch_all still tries
        dr.REFRESH_LOG = tmp_path / "log.json"

        def boom(*a, **kw) -> tuple[pd.DataFrame, list]:
            raise RuntimeError("api down")

        monkeypatch.setattr(dr, "refresh_cpi", boom)
        monkeypatch.setattr(dr, "refresh_wpi", boom)
        monkeypatch.setattr(dr, "refresh_rate", boom)
        monkeypatch.setattr(dr, "refresh_fx", boom)
        monkeypatch.setattr(dr, "refresh_brent", boom)

        res = dr.fetch_all_sources(api_key="k", cadence="all")
        assert res["cpi"]["status"] == "error"
        assert res["wpi"]["status"] == "error"
        assert res["rate"]["status"] == "error"
        assert res["fx"]["status"] == "error"
        assert res["brent"]["status"] == "error"
        assert res["merged"]["status"] == "error"
        assert "missing" in res["merged"]["message"].lower()
    finally:
        dr.MERGED_CSV = orig_csv
        dr.REFRESH_LOG = orig_log


def test_fetch_all_sources_merge_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orig_csv = dr.MERGED_CSV
    orig_log = dr.REFRESH_LOG
    try:
        dr.MERGED_CSV = tmp_path / "merged.csv"
        dr.REFRESH_LOG = tmp_path / "log.json"
        fake = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
                "value": [1.0, 2.0],
            }
        )
        monkeypatch.setattr(dr, "refresh_cpi", lambda api_key=None: (fake, []))
        monkeypatch.setattr(dr, "refresh_wpi", lambda api_key=None: (fake, []))
        monkeypatch.setattr(dr, "refresh_rate", lambda api_key=None: (fake, []))
        monkeypatch.setattr(dr, "refresh_fx", lambda api_key=None: (fake, []))
        monkeypatch.setattr(dr, "refresh_brent", lambda api_key=None: (fake, []))
        monkeypatch.setattr(
            dr,
            "merge_and_save",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("merge boom")),
        )
        res = dr.fetch_all_sources(api_key="k", cadence="all")
        assert res["merged"]["status"] == "error"
        assert "boom" in res["merged"]["message"]
    finally:
        dr.MERGED_CSV = orig_csv
        dr.REFRESH_LOG = orig_log


def test_load_existing_series_handles_corrupt_file(tmp_path: Path) -> None:
    orig = dr.MERGED_CSV
    orig_log = dr.REFRESH_LOG
    try:
        dr.MERGED_CSV = tmp_path / "bad.csv"
        dr.REFRESH_LOG = tmp_path / "log.json"
        dr.MERGED_CSV.write_text("not,a,csv\n,,,")
        # exercise error path inside _load_existing_series
        fake = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="ME"),
                "value": [1.0, 2.0],
            }
        )
        import src.data_refresh as d2

        # patch fetchers to avoid network
        with (
            patch.object(d2, "refresh_fx", return_value=(fake, [])),
            patch.object(d2, "refresh_brent", return_value=(fake, [])),
            patch.object(d2, "refresh_cpi", return_value=(fake, [])),
            patch.object(d2, "refresh_wpi", return_value=(fake, [])),
            patch.object(d2, "refresh_rate", return_value=(fake, [])),
        ):
            res = d2.fetch_all_sources(api_key="k", cadence="daily")
            # cpi skipped but file corrupt -> load returns None, merged fails gracefully
            assert "merged" in res
    finally:
        dr.MERGED_CSV = orig
        dr.REFRESH_LOG = orig_log
