"""Pydantic schemas for hierarchy endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TreeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class NodeCreate(BaseModel):
    parent_id: int
    kind: str  # "region" | "store"
    name: str = Field(default="", max_length=255)
    series_id: Optional[int] = None  # required for kind == "store"


class HierarchyForecastRequest(BaseModel):
    tree_id: int
    horizon: int = 12
    confidence: float = 0.95
    interval_method: str = "analytic"  # analytic | simulate
    trend_kind: str = "add"
    seasonal_kind: str = "add"
    # When "local", unchanged leaves reuse the referenced forecast's stored
    # base columns; only refit nodes and ancestors recompute.  "full" rebuilds
    # everything.  Both paths must agree to 1e-9 (tested).
    recompute_mode: str = "full"
    # Reuse stored base forecasts from this HierarchyForecast (local mode).
    based_on_id: Optional[int] = None
    label: str = ""


class HierarchyBacktestRequest(BaseModel):
    tree_id: int
    origin_start: int
    horizon: int = 8
    stride: int = 1
    confidence: float = 0.95
    interval_method: str = "analytic"
    trend_kind: str = "add"
    seasonal_kind: str = "add"
    label: str = ""
