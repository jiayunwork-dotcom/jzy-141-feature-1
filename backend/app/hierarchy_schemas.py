"""层级（门店—区域—全网）相关请求结构。"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RegionSpec(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    series_ids: List[int]


class HierarchyTreeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    network_name: str = Field(min_length=1, max_length=255, default="全网")
    regions: List[RegionSpec]


class HierarchyForecastRequest(BaseModel):
    tree_id: int
    horizon: int = 12
    confidence: float = 0.95
    interval_method: str = "analytic"  # analytic | simulate
    # 聚合节点（区域/全网）的模型选择；门店层始终用各自被引用的既有拟合。
    auto: bool = True
    trend_kind: Optional[str] = None
    seasonal_kind: Optional[str] = None
    locks: Dict[str, float] = Field(default_factory=dict)
    label: str = ""
    # full = 全部节点重算基础预测；auto = 复用最近一次结果中未变节点的
    # 基础预测，只重算换了拟合的门店及其祖先路径，调和仍对整棵树重跑。
    mode: str = "full"
    # 显式指定每个门店使用的拟合 id；不给则用各门店当前最新拟合。
    base_fit_ids: Optional[Dict[str, int]] = None


class HierarchyBacktestRequest(BaseModel):
    tree_id: int
    origin_start: int
    horizon: int = 8
    stride: int = 1
    confidence: float = 0.95
    interval_method: str = "analytic"
    auto: bool = True
    trend_kind: Optional[str] = None
    seasonal_kind: Optional[str] = None
    locks: Dict[str, float] = Field(default_factory=dict)
    label: str = ""
