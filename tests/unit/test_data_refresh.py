from unittest.mock import MagicMock, patch

from src.data_refresh import fetch_fred_series, fetch_rbi_dbie_series


def test_rbi_fallback_returns_none_gracefully() -> None:
    assert fetch_rbi_dbie_series("WPI") is None


def test_fred_404_handled_as_runtime_error() -> None:
    with patch("src.data_refresh.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        mock_get.return_value = mock_resp
        try:
            fetch_fred_series("INDCPIALLMINMEI", api_key="bad")
            raise AssertionError("should raise")
        except Exception as e:
            assert "404" in str(e) or "FRED" in str(e)
