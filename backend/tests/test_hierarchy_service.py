"""层级服务层测试：过期判定、局部/整棵重算一致、并发版本记录、失败不留半成品。

这些测试与 test_api_flow 一样跑在进程内 SQLite 上（conftest 设置
DATABASE_URL），因此容器里无需 PostgreSQL 也能跑通；SQLAlchemy 的
``with_for_update`` 在 SQLite 方言下被跳过，并发版本一致性靠「先在单事务
快照整组拟合、之后只用内存值」保证。
"""
from __future__ import annotations

import threading
from datetime import date, timedelta

import numpy as np
import pytest

from app.db import session_scope
from app import hierarchy_service as hs
from app import storage
from app.hierarchy_schemas import (
    HierarchyForecastRequest,
    HierarchyTreeCreate,
    RegionSpec,
)
from app.hierarchy_storage import load_tree
from app.kernels.hw import ModelError, forecast
from app.kernels.selection import fit_one

D0 = date(2022, 1, 3)


def _weekly(n):
    return [(D0 + timedelta(weeks=k)).isoformat() for k in range(n)]


def _add_fit(db, sid, y, m=52, locks=None):
    fr = fit_one(y, "add", "add", m, locks=locks or {})
    fc = forecast(fr.final_state, 12, fr.residuals, level=0.95)
    return storage.create_fit(db, sid, {
        "label": "", "auto": False, "trend_kind": "add",
        "seasonal_kind": "add", "period": m,
        "params": {"alpha": fr.params.alpha, "beta": fr.params.beta,
                   "gamma": fr.params.gamma, "phi": fr.params.phi},
        "locks": locks or {}, "sse": fr.sse, "aic": fr.aic,
        "residuals": [float(x) for x in fr.residuals],
        "fitted": [float(x) for x in fr.all_fitted],
        "forecast": {**fc.to_dict(), "horizon": 12, "future_dates": []},
        "initial_state": {"level": fr.initial_level,
                          "trend": fr.initial_trend,
                          "season": [float(x) for x in fr.initial_season]},
        "final_state": fr.final_state.to_dict(), "scores": [],
    })


@pytest.fixture
def tree_with_fits():
    """三店（含噪声以便真正区分重算）两区域；返回 tree_id/series_ids/数据。"""
    n, m = 164, 52
    rng = np.random.default_rng(7)
    t = np.arange(n)
    ys = [
        100 + 0.2 * t + np.where(t % m == 48, 50, 0) + rng.normal(0, 1.0, n),
        150 + 0.3 * t + np.where(t % m == 20, 30, 0) + rng.normal(0, 1.2, n),
        80 + 0.1 * t + np.where(t % m == 48, 20, 0)
        + np.where(t % m == 20, 10, 0) + rng.normal(0, 0.8, n),
    ]
    dates = _weekly(n)
    sids = []
    with session_scope() as db:
        for name, y in zip(["店甲1", "店甲2", "店乙"], ys):
            sid = storage.create_series(db, name, dates, y.tolist(), m).id
            sids.append(sid)
            _add_fit(db, sid, y, m)
        req = HierarchyTreeCreate(
            name="噪声树",
            regions=[RegionSpec(name="甲", series_ids=sids[:2]),
                     RegionSpec(name="乙", series_ids=sids[2:])],
        )
        tree_id = hs.build_tree(req, db)["tree_id"]
        db.commit()
    return tree_id, sids, ys, m


def _run(tree_id, mode="full", locks=None, **over):
    req = HierarchyForecastRequest(
        tree_id=tree_id, horizon=12, auto=False,
        trend_kind="add", seasonal_kind="add",
        locks=locks or {}, mode=mode, **over,
    )
    return hs.run_hierarchy_forecast_job(req, lambda p, s: None)["created_id"]


def _get(fid):
    with session_scope() as db:
        return hs.get_forecast(db, fid)


def test_fresh_result_not_stale_then_marked_stale_after_refit(tree_with_fits):
    tree_id, sids, ys, m = tree_with_fits
    fid = _run(tree_id)
    f = _get(fid)
    assert f["is_stale"] is False
    assert f["stale_detail"] == []
    with session_scope() as db:
        first_fits = hs.latest_fit_ids_for_series(db, sids)
    assert f["fit_refs"] == {str(sid): first_fits[sid] for sid in sids}

    # 给第一家店锁参重拟，产生新 fit
    with session_scope() as db:
        _add_fit(db, sids[0], ys[0], m, locks={"alpha": 0.5})
        db.commit()

    f = _get(fid)
    assert f["is_stale"] is True
    detail = f["stale_detail"]
    assert len(detail) == 1 and detail[0]["series_id"] == sids[0]
    assert detail[0]["used_fit_id"] != detail[0]["latest_fit_id"]


def test_partial_recompute_matches_full_to_1e_9(tree_with_fits):
    tree_id, sids, ys, m = tree_with_fits
    # 基准全树结果
    _run(tree_id, mode="full")
    # 锁参重拟其中一家店
    with session_scope() as db:
        _add_fit(db, sids[1], ys[1], m, locks={"alpha": 0.4, "gamma": 0.6})
        db.commit()

    fid_partial = _run(tree_id, mode="auto")
    fid_full = _run(tree_id, mode="full")
    fp, ff = _get(fid_partial), _get(fid_full)
    # 两条路对整棵树每个节点、每个日期的调和结果一致
    for a, b in zip(fp["nodes"], ff["nodes"]):
        assert a["node_id"] == b["node_id"]
        np.testing.assert_allclose(
            a["rec"]["point"], b["rec"]["point"], atol=1e-9,
            err_msg=f"node {a['name']}",
        )
        np.testing.assert_allclose(
            a["rec"]["lower"], b["rec"]["lower"], atol=1e-9)
        np.testing.assert_allclose(
            a["rec"]["upper"], b["rec"]["upper"], atol=1e-9)


def test_recorded_fit_refs_match_versions_actually_used(tree_with_fits,
                                                        monkeypatch):
    """任务跑到一半，另一线程给某店重拟：结果记录的必须是它实际用的旧组。"""
    tree_id, sids, ys, m = tree_with_fits
    used_during = {}
    original = hs._compute_all_bases

    def spying_compute(tree, snap, req, report):
        # 记录快照到的那一组 fit 版本
        used_during["snapshot_refs"] = {
            ls["series_id"]: ls["fit_id"] for ls in snap["leaf"].values()
        }
        # 快照完成后、计算进行中，另一位计划员给第一家店锁参重拟
        with session_scope() as db:
            _add_fit(db, sids[0], ys[0], m, locks={"alpha": 0.5})
            db.commit()
        return original(tree, snap, req, report)

    monkeypatch.setattr(hs, "_compute_all_bases", spying_compute)
    fid = _run(tree_id)
    f = _get(fid)
    # 记录的引用版本 == 快照到的那组（旧 fit id），没有混进新 fit
    assert f["fit_refs"][str(sids[0])] == used_during["snapshot_refs"][sids[0]]
    with session_scope() as db:
        latest = hs.latest_fit_ids_for_series(db, sids)
    # 因为重拟已经完成，这次结果读时即应被标记为过期
    assert latest[sids[0]] != f["fit_refs"][str(sids[0])]
    assert f["is_stale"] is True


def test_node_failure_reports_node_and_persists_nothing(tree_with_fits):
    tree_id, sids, ys, m = tree_with_fits
    # 构造一棵含「区域加总后存在负值」的树，指定乘法季节必在聚合节点失败
    n = 164
    neg = (100 + 0.2 * np.arange(n)).copy()
    zero = (150 + 0.3 * np.arange(n)).copy()
    neg[10], zero[10] = -40.0, 0.0  # 区域历史 t=10 为负
    dates = _weekly(n)
    with session_scope() as db:
        i_neg = storage.create_series(db, "负店", dates, neg.tolist(), m).id
        i_zero = storage.create_series(db, "零店", dates, zero.tolist(), m).id
        _add_fit(db, i_neg, neg, m)
        _add_fit(db, i_zero, zero, m)
        bad_id = hs.build_tree(HierarchyTreeCreate(
            name="坏树",
            regions=[RegionSpec(name="问题区域", series_ids=[i_neg, i_zero])],
        ), db)["tree_id"]
        db.commit()

    req = HierarchyForecastRequest(
        tree_id=bad_id, horizon=12, auto=False,
        trend_kind="add", seasonal_kind="mul",
    )
    with pytest.raises(ModelError) as ei:
        hs.run_hierarchy_forecast_job(req, lambda p, s: None)
    msg = str(ei.value)
    # 明确指出节点与原因
    assert "问题区域" in msg or "全网" in msg
    assert "正" in msg
    with session_scope() as db:
        assert hs.list_forecasts(db, bad_id) == []


def test_leaf_without_fit_is_reported(tree_with_fits):
    tree_id, sids, ys, m = tree_with_fits
    n = 164
    with session_scope() as db:
        sid = storage.create_series(
            db, "未拟合", _weekly(n), (50 + 0.1 * np.arange(n)).tolist(), m
        ).id
        tid = hs.build_tree(HierarchyTreeCreate(
            name="缺拟合树",
            regions=[RegionSpec(name="区域", series_ids=[sid])],
        ), db)["tree_id"]
        db.commit()
    req = HierarchyForecastRequest(
        tree_id=tid, horizon=12, auto=False,
        trend_kind="add", seasonal_kind="add",
    )
    with pytest.raises(ModelError) as ei:
        hs.run_hierarchy_forecast_job(req, lambda p, s: None)
    assert "未拟合" in str(ei.value)
    with session_scope() as db:
        assert hs.list_forecasts(db, tid) == []


def test_explicit_fit_ids_pin_versions(tree_with_fits):
    tree_id, sids, ys, m = tree_with_fits
    with session_scope() as db:
        old_fits = hs.latest_fit_ids_for_series(db, sids)
    # 先制造一个新拟合
    with session_scope() as db:
        new_fit = _add_fit(db, sids[2], ys[2], m, locks={"alpha": 0.6})
        new_id = new_fit.id
        db.commit()
    # 显式指定每家店都用各自的旧 fit id 组，结果不得引用新 fit
    f = _get(_run(tree_id, base_fit_ids={str(s): old_fits[s] for s in sids}))
    assert f["fit_refs"][str(sids[2])] != new_id
    assert f["is_stale"] is True  # 引用的不是最新 -> 过期


def test_api_attachment_mismatch_returns_structured_400(client):
    c, _ = client
    n = 60
    dates = _weekly(n)
    r = c.post("/api/series", json={
        "name": "挂接A", "period": 52, "dates": dates,
        "values": [100.0] * n})
    a = r.json()["id"]
    r = c.post("/api/series", json={
        "name": "挂接B", "period": 12, "dates": dates,
        "values": [200.0] * n})
    b = r.json()["id"]
    r = c.post("/api/hierarchy/trees/validate", json={
        "name": "预检树", "network_name": "全网",
        "regions": [{"name": "甲", "series_ids": [a, b]}]})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert any(i["series_id"] == b and "周期" in i["reason"]
               for i in body["issues"])

    r = c.post("/api/hierarchy/trees", json={
        "name": "预检树", "network_name": "全网",
        "regions": [{"name": "甲", "series_ids": [a, b]}]})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, dict) and detail["issues"]
