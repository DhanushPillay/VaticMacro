"""Pandera schemas for VaticMacro datasets."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class InflationDatasetSchema(pa.DataFrameModel):
    """Monthly merged dataset — validates data/inflation_dataset.csv."""

    Date: Series[pa.DateTime] = pa.Field(nullable=False)
    INDCPIALLMINMEI: Series[float] = pa.Field(ge=0, nullable=False)
    WPIATT01INM661N: Series[float] = pa.Field(ge=0, nullable=True)
    INTDSRINM193N: Series[float] = pa.Field(ge=0, nullable=True)
    DEXINUS: Series[float] = pa.Field(ge=0, nullable=True)
    Average_of_DCOILBRENTEU: Series[float] = pa.Field(
        alias="Average of DCOILBRENTEU", ge=0, nullable=True
    )

    class Config:
        strict = False
        coerce = True
        name = "InflationDatasetSchema"
