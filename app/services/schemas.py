"""Pydantic schemas for API validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SandboxRequest(BaseModel):
    """Validated inputs for predictive sandbox — clipped ±50%."""

    WPIATT01INM661N_pct_1m: float = Field(default=0, ge=-50, le=50)
    INTDSRINM193N_pct_1m: float = Field(default=0, ge=-50, le=50)
    DEXINUS_pct_1m: float = Field(default=0, ge=-50, le=50)
    brent_pct_1m: float = Field(
        default=0, ge=-50, le=50, alias="Average of DCOILBRENTEU_pct_1m"
    )

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool = False
