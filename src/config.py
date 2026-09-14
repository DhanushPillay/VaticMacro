"""Centralized typed configuration for VaticMacro."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

COLUMN_MAP: dict[str, str] = {
    "cpi": "INDCPIALLMINMEI",
    "wpi": "WPIATT01INM661N",
    "interest_rate": "INTDSRINM193N",
    "usd_inr": "DEXINUS",
    "brent_crude": "Average of DCOILBRENTEU",
    "industrial_prod": "INDPRINTO01GYSAM",
    "trade_balance": "XTNTVA01INM667N",
}


@dataclass(frozen=True)
class Settings:
    """Typed app settings — single source for paths and column map."""

    COLUMN_MAP: dict[str, str] = field(default_factory=lambda: dict(COLUMN_MAP))
    DATA_PATH: Path = Path("data/inflation_dataset.csv")
    MODEL_PATH: Path = Path("models/best_model.pkl")
    METRICS_PATH: Path = Path("models/metrics.json")
    HOLDOUT_PATH: Path = Path("models/holdout.csv")
    REFRESH_LOG: Path = Path("data/refresh_log.json")
    MODEL_DIR: Path = Path("models")


SETTINGS = Settings()
# Backwards-compat re-exports for legacy imports
MODEL_PATH = str(SETTINGS.MODEL_PATH)
METRICS_PATH = str(SETTINGS.METRICS_PATH)
DATA_PATH = str(SETTINGS.DATA_PATH)
