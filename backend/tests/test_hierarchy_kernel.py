"""层级调和内核与挂接校验的纯数值测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.hierarchy_utils import (
    LeafSpec,
    aggregate_history,
    check_tree_assignment,
)
from app.kernels.hierarchy import (
    NodeBase,
    coherence_residual,
    reconcile,
)


def _leaf(nid, name, point, var):
    return NodeBase(
        node_id=nid, level=0, name=name,
        point=np.asarray(point, float),
        lower=np.asarray(point, float) - 2,
        upper=np.asarray(point, float) + 2,
        residual_var=var,
    )


def _noisy_tree(seed=0, noise=0.3):
    """三家店两区域全网，基础预测故意不相加（需要调和）。"""
    rng = np.random.default_rng(seed)
    h = 10
    l1 = 100 + np.arange(h) * 0.2 + rng.normal(0, noise, h)
    l2 = 150 + np.arange(h) * 0.3 + rng.normal(0, noise, h)
    l3 = 80 + np.arange(h) * 0.1 + rng.normal(0, noise, h)
    rA = l1 + l2 + rng.normal(0, noise * 2, h)
    rB = l3 + rng.normal(0, noise * 2, h)
    net = rA + rB + rng.normal(0, noise * 3, h)
    bases = [
        _leaf(1, "店1", l1, 1.0), _leaf(2, "店2", l2, 1.2),
        _leaf(3, "店3", l3, 0.8),
        NodeBase(10, 1, "区域甲", rA, rA - 4, rA + 4, 3.0),
        NodeBase(11, 1, "区域乙", rB, rB - 4, rB + 4, 3.0),
        NodeBase(20, 2, "全网", net, net - 8, net + 8, 6.0),
    ]
    bottom = [1, 2, 3]
    desc = {1: [1], 2: [2], 3: [3], 10: [1, 2], 11: [3], 20: [1, 2, 3]}
    parent_children = {10: [1, 2], 11: [3], 20: [10, 11]}
    return bases, bottom, desc, parent_children


def test_reconciliation_coherent_all_three_levels():
    bases, bottom, desc, pc = _noisy_tree()
    rec = reconcile(bases, bottom, desc)
    resid = coherence_residual(rec, pc)
    # 任意父节点每期 = 直接下级之和，远严于百万分之一
    assert np.max(np.abs(resid)) < 1e-9
    # 调和确实改动了（本来不一致的）基础预测
    assert np.max(np.abs(rec.rec_point - rec.base_point)) > 1e-6


def test_relative_coherence_tolerance_per_million():
    bases, bottom, desc, pc = _noisy_tree()
    rec = reconcile(bases, bottom, desc)
    for pid, children in pc.items():
        parent = rec.rec_point[rec.row_index[pid]]
        child_sum = sum(
            rec.rec_point[rec.row_index[c]] for c in children
        )
        rel = np.abs(parent - child_sum) / np.maximum(np.abs(parent), 1.0)
        assert np.max(rel) < 1e-6


def test_single_child_region_equals_store_every_date():
    bases, bottom, desc, pc = _noisy_tree()
    rec = reconcile(bases, bottom, desc)
    np.testing.assert_allclose(
        rec.rec_point[rec.row_index[11]],
        rec.rec_point[rec.row_index[3]],
        atol=1e-12,
    )


def test_interval_widths_monotone_non_decreasing():
    bases, bottom, desc, _ = _noisy_tree()
    rec = reconcile(bases, bottom, desc)
    for nid in rec.node_ids:
        i = rec.row_index[nid]
        width = rec.rec_upper[i] - rec.rec_lower[i]
        assert np.all(np.diff(width) >= -1e-9), nid
        assert np.all(rec.rec_upper[i] >= rec.rec_lower[i])


def test_zero_variance_stabilized_and_identity_when_already_coherent():
    h = 6
    l1 = np.arange(h, dtype=float) + 1
    l2 = np.arange(h, dtype=float) * 2 + 3
    rA = l1 + l2
    net = rA.copy()
    bases = [
        _leaf(1, "店1", l1, 0.0), _leaf(2, "店2", l2, 0.0),
        NodeBase(10, 1, "区域", rA, rA, rA, 0.0),
        NodeBase(20, 2, "全网", net, net, net, 0.0),
    ]
    desc = {1: [1], 2: [2], 10: [1, 2], 20: [1, 2]}
    rec = reconcile(bases, [1, 2], desc)
    # 无噪声且本来一致：调和应保持不动（权重地板不得拉动结果）
    np.testing.assert_allclose(rec.rec_point, rec.base_point, atol=1e-9)


# ---------------------------------------------------------------------------
# 挂接校验
# ---------------------------------------------------------------------------

def _datespec(start="2022-01-03", n=10):
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(weeks=k)).isoformat() for k in range(n)]


def test_assignment_accepts_aligned_tree():
    dates = _datespec()
    a = LeafSpec(1, "店1", 52, dates, [1.0] * 10)
    b = LeafSpec(2, "店2", 52, dates, [2.0] * 10)
    c = LeafSpec(3, "店3", 52, dates, [3.0] * 10)
    check = check_tree_assignment([("甲", [a, b]), ("乙", [c])])
    assert check.ok
    assert check.issues == []


def test_assignment_rejects_period_mismatch_with_child_detail():
    a = LeafSpec(1, "店1", 52, _datespec(), [1.0] * 10)
    b = LeafSpec(2, "店2", 12, _datespec(), [2.0] * 10)
    check = check_tree_assignment([("甲", [a, b])])
    assert not check.ok
    assert len(check.issues) == 1
    issue = check.issues[0]
    assert issue.region_name == "甲"
    assert issue.series_id == 2
    assert "周期" in issue.reason


def test_assignment_rejects_start_end_and_length_mismatch():
    a = LeafSpec(1, "店1", 52, _datespec("2022-01-03", 10), [1.0] * 10)
    late = LeafSpec(2, "店2", 52, _datespec("2022-01-10", 10), [2.0] * 10)
    short = LeafSpec(3, "店3", 52, _datespec("2022-01-03", 9), [3.0] * 9)
    check = check_tree_assignment([("甲", [a, late, short])])
    reasons = {(i.series_id, i.reason) for i in check.issues}
    assert any(sid == 2 and "起始周" in r for sid, r in reasons)
    assert any(sid == 3 and "周数" in r for sid, r in reasons)


def test_assignment_rejects_duplicate_store():
    dates = _datespec()
    a = LeafSpec(1, "店1", 52, dates, [1.0] * 10)
    dup = LeafSpec(1, "店1", 52, dates, [1.0] * 10)
    check = check_tree_assignment([("甲", [a]), ("乙", [dup])])
    assert not check.ok
    assert any("重复挂接" in i.reason for i in check.issues)


def test_aggregate_history_sums_week_by_week():
    total = aggregate_history([[1, 2, 3], [4, 5, 6], [-1, 0, 1]])
    np.testing.assert_array_equal(total, [4, 7, 10])
    with pytest.raises(ValueError):
        aggregate_history([[1, 2], [1, 2, 3]])
