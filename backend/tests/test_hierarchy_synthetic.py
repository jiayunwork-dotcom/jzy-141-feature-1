"""需求给定合成算例：三家店、两区域、全网，加法趋势+加法季节 12 步。"""
from __future__ import annotations

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
from app.kernels.hw import forecast
from app.kernels.selection import fit_one

D0 = "2022-01-03"
N, M = 164, 52


def synthetic_arrays():
    t = np.arange(N)
    y1 = 100 + 0.2 * t + np.where(t % M == 48, 50, 0)
    y2 = 150 + 0.3 * t + np.where(t % M == 20, 30, 0)
    y3 = (80 + 0.1 * t
          + np.where(t % M == 48, 20, 0)
          + np.where(t % M == 20, 10, 0))
    return y1, y2, y3


@pytest.fixture
def synthetic_tree():
    """建三店两区域树并各做一次加加拟合；返回 tree_id 与系列 id。"""
    from datetime import date, timedelta
    d0 = date.fromisoformat(D0)
    dates = [(d0 + timedelta(weeks=k)).isoformat() for k in range(N)]
    ys = synthetic_arrays()
    names = ["合成店一", "合成店二", "合成店三"]
    series_ids = []
    with session_scope() as db:
        for name, y in zip(names, ys):
            sid = storage.create_series(db, name, dates, y.tolist(), M).id
            series_ids.append(sid)
            fr = fit_one(y, "add", "add", M, locks={})
            fc = forecast(fr.final_state, 12, fr.residuals, level=0.95)
            storage.create_fit(db, sid, {
                "label": "", "auto": False, "trend_kind": "add",
                "seasonal_kind": "add", "period": M,
                "params": {"alpha": fr.params.alpha, "beta": fr.params.beta,
                           "gamma": fr.params.gamma, "phi": fr.params.phi},
                "locks": {}, "sse": fr.sse, "aic": fr.aic,
                "residuals": [float(x) for x in fr.residuals],
                "fitted": [float(x) for x in fr.all_fitted],
                "forecast": {**fc.to_dict(), "horizon": 12, "future_dates": []},
                "initial_state": {
                    "level": fr.initial_level, "trend": fr.initial_trend,
                    "season": [float(x) for x in fr.initial_season]},
                "final_state": fr.final_state.to_dict(), "scores": [],
            })
        req = HierarchyTreeCreate(
            name="合成树", network_name="全网",
            regions=[RegionSpec(name="区域甲", series_ids=series_ids[:2]),
                     RegionSpec(name="区域乙", series_ids=series_ids[2:])],
        )
        tree_id = hs.build_tree(req, db)["tree_id"]
        db.commit()
    return tree_id, series_ids, ys


def _forecast(tree_id, **over):
    req = HierarchyForecastRequest(
        tree_id=tree_id, horizon=12, confidence=0.95,
        interval_method="analytic", auto=False,
        trend_kind="add", seasonal_kind="add", **over,
    )
    out = hs.run_hierarchy_forecast_job(req, lambda p, s: None)
    with session_scope() as db:
        return hs.get_forecast(db, out["created_id"])


def test_network_first_step_is_428_4(synthetic_tree):
    tree_id, _, _ = synthetic_tree
    f = _forecast(tree_id)
    net = [n for n in f["nodes"] if n["level"] == 2][0]
    assert net["rec"]["point"][0] == pytest.approx(428.4, abs=1e-9)
    assert net["base"]["point"][0] == pytest.approx(428.4, abs=1e-9)


def test_base_vs_reconciled_within_0_01_all_nodes_dates(synthetic_tree):
    tree_id, _, _ = synthetic_tree
    f = _forecast(tree_id)
    for n in f["nodes"]:
        diff = np.abs(
            np.array(n["base"]["point"]) - np.array(n["rec"]["point"])
        )
        assert np.max(diff) < 0.01, (n["name"], np.max(diff))


def test_coherence_at_1e_9_and_single_child_region(synthetic_tree):
    tree_id, _, _ = synthetic_tree
    f = _forecast(tree_id)
    nodes = {n["node_id"]: n for n in f["nodes"]}
    with session_scope() as db:
        tree = hs.load_tree(db, tree_id)
        for pid, children in tree.parent_children().items():
            parent = np.array(nodes[pid]["rec"]["point"])
            child_sum = sum(
                np.array(nodes[c]["rec"]["point"]) for c in children
            )
            np.testing.assert_allclose(parent, child_sum, atol=1e-9)
    region_b = [n for n in f["nodes"] if n["name"] == "区域乙"][0]
    store3 = [n for n in f["nodes"] if n["name"] == "合成店三"][0]
    np.testing.assert_allclose(
        region_b["rec"]["point"], store3["rec"]["point"], atol=1e-12
    )


def test_future_dates_are_consecutive_mondays(synthetic_tree):
    tree_id, _, _ = synthetic_tree
    f = _forecast(tree_id)
    assert f["future_dates"][0] == "2025-02-24"  # 2022-01-03 + 164 周
    assert len(f["future_dates"]) == 12
    from datetime import date
    ds = [date.fromisoformat(d) for d in f["future_dates"]]
    assert all((b - a).days == 7 for a, b in zip(ds, ds[1:]))
