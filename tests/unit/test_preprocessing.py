from pathlib import Path

import pandas as pd

from src.preprocessing import load_and_clean_data


def test_load_handles_observation_date_format(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    pd.DataFrame(
        {"observation_date": ["2020-01-01", "2020-02-01"], "DEXINUS": [75.1, 75.3]}
    ).to_csv(p, index=False)
    df = load_and_clean_data(str(p))
    assert "observation_date" in df.columns
    assert len(df) == 2


def test_load_handles_row_labels(tmp_path: Path) -> None:
    p = tmp_path / "b.csv"
    pd.DataFrame(
        {
            "Row Labels": ["2020-01-01", "2020-02-01", "Grand Total"],
            "INDCPIALLMINMEI": [100, 101, 999],
        }
    ).to_csv(p, index=False)
    df = load_and_clean_data(str(p))
    assert "Grand Total" not in df.astype(str).values
