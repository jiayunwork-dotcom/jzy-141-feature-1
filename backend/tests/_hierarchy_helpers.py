"""Helpers shared by hierarchy tests: reference data and in-memory trees."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from app.hierarchy.tree import LeafInfo, NodeSnapshot, TreeSnapshot


REF_START = date(2022, 1, 3)  # Monday
REF_N = 164
REF_PERIOD = 52


def reference_store_values() -> List[np.ndarray]:
    """Three noiseless weekly series specified in the acceptance brief."""
    t = np.arange(REF_N)
    y1 = 100 + 0.2 * t + np.where(t % 52 == 48, 50, 0)
    y2 = 150 + 0.3 * t + np.where(t % 52 == 20, 30, 0)
    y3 = (
        80
        + 0.1 * t
        + np.where(t % 52 == 48, 20, 0)
        + np.where(t % 52 == 20, 10, 0)
    )
    return [y1, y2, y3]


def weekly_dates(n: int = REF_N, start: date = REF_START) -> List[str]:
    return [(start + timedelta(weeks=k)).isoformat() for k in range(n)]


@dataclass
class _Row:
    id: int
    tree_id: int
    parent_id: Optional[int]
    kind: str
    name: str
    level: int
    series_id: Optional[int]


@dataclass
class _Series:
    id: int
    name: str
    period: int
    dates: List[str]
    values: List[float]


def make_reference_tree(
    values: Optional[List[np.ndarray]] = None,
    periods=(52, 52, 52),
    date_lists: Optional[List[List[str]]] = None,
) -> TreeSnapshot:
    """Build the reference tree: [s1,s2]->region甲, s3->region乙, root 全网."""
    if values is None:
        values = reference_store_values()
    if date_lists is None:
        date_lists = [weekly_dates(len(v)) for v in values]

    # ids: 1 root, 2 region甲, 3-5 stores 1..3, 6 region乙
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "区域甲", 1, None),
        _Row(3, 1, 2, "store", "门店一", 2, 11),
        _Row(4, 1, 2, "store", "门店二", 2, 12),
        _Row(6, 1, 1, "region", "区域乙", 1, None),
        _Row(5, 1, 6, "store", "门店三", 2, 13),
    ]
    series_by_id: Dict[int, _Series] = {}
    for i, (v, dl, per) in enumerate(zip(values, date_lists, periods)):
        sid = 11 + i
        series_by_id[sid] = _Series(
            id=sid, name=f"门店{['一', '二', '三'][i]}",
            period=per, dates=dl,
            values=[float(x) for x in v],
        )
    from app.hierarchy.tree import build_snapshot
    return build_snapshot(1, "参考树", 1, rows, series_by_id)


def base_fit_all(
    tree: TreeSnapshot,
    horizon: int = 12,
    trend_kind: str = "add",
    seasonal_kind: str = "add",
    confidence: float = 0.95,
    interval_method: str = "analytic",
):
    """Fit every node from its history; return {node_id: NodeBaseForecast}."""
    from app.kernels.reconcile import fit_base_forecast
    base = {}
    for node in tree.nodes.values():
        base[node.id] = fit_base_forecast(
            node, node.history_values, tree.period,
            trend_kind, seasonal_kind, horizon,
            confidence, interval_method,
        )
    return base
