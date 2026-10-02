"""层级滚动原点回测：按层 MAE/MASE、调和前后对照、原点隔离。"""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pytest

from app.db import session_scope
from app import hierarchy_service as hs
from app import storage
from app.hierarchy_schemas import (
    HierarchyBacktestRequest,
    HierarchyTreeCreate,
    RegionSpec,
)
from app.hierarchy_storage import get_store_refs, load_tree
from app.kernels.hier_runner import (
    HierarchyShape,
    NodeModelSpec,
    run_hierarchy_backtest,
)
from app.kernels.hw import forecast
from app.kernels.selection import fit_one

N, M = 164, 52
D0 = date(2022, 1, 3)


def _synthetic():
    t = np.arange(N)
    return [
        100 + 0.2 * t + np.where(t % M == 48, 50, 0),
        150 + 0.3 * t + np.where(t % M == 20, 30, 0),
        80 + 0.1 * t + np.where(t % M == 48, 20, 0)
        + np.where(t % M == 20, 10, 0),
    ]


@pytest.fixture
def shape_and_values():
    dates = [(D0 + timedelta(weeks=k)).isoformat() for k in range(N)]
    sids = []
    with session_scope() as db:
        for name, y in zip(["回测店1", "回测店2", "回测店3"], _synthetic()):
            sid = storage.create_series(db, name, dates, y.tolist(), M).id
            sids.append(sid)
            fr = fit_one(y, "add", "add", M, locks={})
            fc = forecast(fr.final_state, 12, fr.residuals)
            storage.create_fit(db, sid, {
                "label": "", "auto": False, "trend_kind": "add",
                "seasonal_kind": "add", "period": M,
                "params": {"alpha": fr.params.alpha, "beta": fr.params.beta,
                           "gamma": fr.params.gamma, "phi": fr.params.phi},
                "locks": {}, "sse": fr.sse, "aic": fr.aic,
                "residuals": [float(x) for x in fr.residuals],
                "fitted": [float(x) for x in fr.all_fitted],
                "forecast": {**fc.to_dict(), "horizon": 12, "future_dates": []},
                "initial_state": {"level": fr.initial_level,
                                  "trend": fr.initial_trend,
                                  "season": [float(x) for x in fr.initial_season]},
                "final_state": fr.final_state.to_dict(), "scores": []})
        tree_id = hs.build_tree(HierarchyTreeCreate(
            name="回测树",
            regions=[RegionSpec(name="区域甲", series_ids=sids[:2]),
                     RegionSpec(name="区域乙", series_ids=sids[2:])],
        ), db)["tree_id"]
        db.commit()

    with session_scope() as db:
        tree = load_tree(db, tree_id)
        refs = get_store_refs(db, [tree.nodes[i].series_id
                                   for i in tree.bottom_ids])
        leaf_values = {
            i: np.asarray(refs[tree.nodes[i].series_id].values, float)
            for i in tree.bottom_ids
        }
        shape = HierarchyShape(
            node_ids=tree.ordered_node_ids(),
            levels={i: n.level for i, n in tree.nodes.items()},
            names={i: n.name for i, n in tree.nodes.items()},
            parent_children=tree.parent_children(),
            descendants_bottom=tree.descendants_bottom(),
            bottom_ids=tree.bottom_ids,
        )
    return tree_id, leaf_values, shape


def test_layerwise_metrics_noiseless(shape_and_values):
    _, leaf_values, shape = shape_and_values
    spec = NodeModelSpec(auto=False, trend_kind="add",
                         seasonal_kind="add", locks={})
    res = run_hierarchy_backtest(
        leaf_values, shape, M, origin_start=120, horizon=8, stride=17,
        spec=spec,
    )
    assert set(res["layers"]) == {"0", "1", "2"}
    for key, label in (("0", "门店"), ("1", "区域"), ("2", "全网")):
        lay = res["layers"][key]
        assert lay["label"] == label
        # 无噪声加加模型：调和前后 MAE 都应几乎为 0
        assert lay["base"]["mae"] < 1e-6
        assert lay["rec"]["mae"] < 1e-6
        # MASE 有限（季节朴素尺度 > 0）
        assert np.isfinite(lay["base"]["mase"])
        assert np.isfinite(lay["rec"]["mase"])
    # 逐原点的每个节点都有调和前/后两套点预测与实际值
    first = res["origins"][0]
    assert first["origin"] == 120
    assert len(first["nodes"]) == len(shape.node_ids)
    for row in first["nodes"]:
        assert len(row["base_point"]) == 8
        assert len(row["rec_point"]) == 8
        assert len(row["actual"]) == 8


def test_origin_isolation_extreme_future_values(shape_and_values):
    _, leaf_values, shape = shape_and_values
    spec = NodeModelSpec(auto=False, trend_kind="add",
                         seasonal_kind="add", locks={})
    origin, horizon = 120, 8
    clean = run_hierarchy_backtest(
        leaf_values, shape, M, origin_start=origin, horizon=horizon,
        stride=99, spec=spec,
    )
    # 在该原点之后注入极端值：污染一个门店的未来与训练远段，
    # 但对原点 k=120，所有节点只应看到 y[:120]。
    contaminated = dict(leaf_values)
    target = shape.bottom_ids[0]
    arr = contaminated[target].copy()
    arr[origin + 1:] += 9e6
    contaminated[target] = arr
    polluted = run_hierarchy_backtest(
        contaminated, shape, M, origin_start=origin, horizon=horizon,
        stride=99, spec=spec,
    )
    for a, b in zip(clean["origins"][0]["nodes"],
                    polluted["origins"][0]["nodes"]):
        np.testing.assert_allclose(a["base_point"], b["base_point"], atol=1e-8)
        np.testing.assert_allclose(a["rec_point"], b["rec_point"], atol=1e-8)


def test_api_hierarchy_backtest_job(client, shape_and_values):
    c, jobs = client
    tree_id, _, _ = shape_and_values
    r = c.post("/api/hierarchy/backtests", json={
        "tree_id": tree_id, "origin_start": 120, "horizon": 8,
        "stride": 34, "confidence": 0.95, "interval_method": "analytic",
        "auto": False, "trend_kind": "add", "seasonal_kind": "add",
        "locks": {},
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    for _ in range(300):
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    bt = c.get(f"/api/hierarchy/backtests/{j['created_id']}").json()
    assert set(bt["result"]["layers"]) == {"0", "1", "2"}
    listed = c.get(f"/api/hierarchy/backtests/by-tree/{tree_id}").json()
    assert any(b["id"] == j["created_id"] for b in listed)
