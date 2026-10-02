"""Tree construction / attachment validation tests."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from _hierarchy_helpers import (
    REF_N,
    _Row,
    _Series,
    make_reference_tree,
    reference_store_values,
    weekly_dates,
)
from app.hierarchy.tree import HierarchyError, build_snapshot


def _series_rows(values, periods=(52, 52, 52), date_lists=None):
    if date_lists is None:
        date_lists = [weekly_dates(len(v)) for v in values]
    series = {}
    for i, (v, dl, per) in enumerate(zip(values, date_lists, periods)):
        sid = 11 + i
        series[sid] = _Series(
            id=sid, name=f"s{i+1}", period=per, dates=dl,
            values=[float(x) for x in v],
        )
    return series


def test_valid_reference_tree_builds():
    tree = make_reference_tree()
    assert tree.period == 52
    assert tree.root.kind == "network"
    assert len(tree.leaves()) == 3


def test_mismatched_period_rejected_and_named():
    vals = reference_store_values()
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "区域甲", 1, None),
        _Row(3, 1, 2, "store", "门店一", 2, 11),
        _Row(4, 1, 2, "store", "门店二", 2, 12),
    ]
    series = _series_rows(vals[:2], periods=(52, 12))
    with pytest.raises(HierarchyError) as ei:
        build_snapshot(1, "t", 1, rows, series)
    msg = str(ei.value)
    assert "门店二" in msg and "12" in msg and "52" in msg and "周期" in msg


def test_mismatched_start_week_rejected_and_named():
    vals = reference_store_values()
    dl1 = weekly_dates(REF_N)
    dl2 = weekly_dates(REF_N, start=date(2022, 1, 10))  # one week late
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "区域甲", 1, None),
        _Row(3, 1, 2, "store", "门店一", 2, 11),
        _Row(4, 1, 2, "store", "门店二", 2, 12),
    ]
    series = _series_rows(
        vals[:2], date_lists=[dl1, dl2]
    )
    with pytest.raises(HierarchyError) as ei:
        build_snapshot(1, "t", 1, rows, series)
    msg = str(ei.value)
    assert "门店二" in msg and "起始周" in msg


def test_mismatched_end_week_rejected_and_named():
    vals = reference_store_values()
    dl1 = weekly_dates(REF_N)
    dl2 = weekly_dates(REF_N - 4)
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "区域甲", 1, None),
        _Row(3, 1, 2, "store", "门店一", 2, 11),
        _Row(4, 1, 2, "store", "门店二", 2, 12),
    ]
    series = _series_rows(vals[:2], date_lists=[dl1, dl2])
    with pytest.raises(HierarchyError) as ei:
        build_snapshot(1, "t", 1, rows, series)
    assert "截止周" in str(ei.value) and "门店二" in str(ei.value)


def test_max_three_levels_enforced():
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "区域甲", 1, None),
        _Row(3, 1, 2, "store", "门店一", 2, 11),
        # a fourth level hanging under the store is structurally illegal
        _Row(4, 1, 3, "region", "深层区域", 3, None),
        _Row(5, 1, 4, "store", "深层门店", 4, 12),
    ]
    vals = reference_store_values()
    series = _series_rows(vals[:2])
    with pytest.raises(HierarchyError):
        build_snapshot(1, "t", 1, rows, series)


def test_duplicate_root_rejected():
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, None, "network", "全网2", 0, None),
    ]
    with pytest.raises(HierarchyError, match="根"):
        build_snapshot(1, "t", 1, rows, {})


def test_internal_node_without_leaf_rejected():
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "空区域", 1, None),
        _Row(5, 1, 1, "region", "区域乙", 1, None),
        _Row(3, 1, 5, "store", "门店一", 2, 11),
    ]
    vals = reference_store_values()
    series = _series_rows(vals[:1])
    with pytest.raises(HierarchyError, match="空区域"):
        build_snapshot(1, "t", 1, rows, series)


def test_aggregate_cannot_reference_series():
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "区域甲", 1, 11),
        _Row(5, 1, 1, "region", "区域乙", 1, None),
        _Row(3, 1, 5, "store", "门店一", 2, 12),
    ]
    vals = reference_store_values()
    series = _series_rows(vals[:2])
    with pytest.raises(HierarchyError, match="不允许直接关联上传序列"):
        build_snapshot(1, "t", 1, rows, series)


def test_missing_leaf_series_named():
    rows = [
        _Row(1, 1, None, "network", "全网", 0, None),
        _Row(2, 1, 1, "region", "区域乙", 1, None),
        _Row(3, 1, 2, "store", "门店一", 2, 99),
    ]
    with pytest.raises(HierarchyError, match="门店一"):
        build_snapshot(1, "t", 1, rows, {})
