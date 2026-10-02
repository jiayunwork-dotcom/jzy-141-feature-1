"""End-to-end hierarchy tests through FastAPI + SQLite.

Pins down:
* reference numbers via the real API (428.4, coherence, region 乙 identity);
* attachment rejection with the offending child named;
* full vs local recompute agree date by date to 1e-9;
* staleness flags after refit / structural change;
* version pinning when a refit happens mid-run (concurrent planners);
* node failure reports the node and persists nothing;
* hierarchy backtest runs per-origin and reports by-level MAE/MASE.
"""
from __future__ import annotations

import time
from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from _hierarchy_helpers import reference_store_values, weekly_dates


@pytest.fixture(scope="module")
def client():
    from app.db import init_db, wait_for_db
    from app.hierarchy import models  # noqa: F401
    from app.jobs import job_manager
    from app.main import app

    wait_for_db()
    init_db()
    with TestClient(app) as c:
        yield c, job_manager


def _wait(c, job_id, timeout=120):
    for _ in range(int(timeout / 0.1)):
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _create_series(c, name, values, period=52, start=date(2022, 1, 3)):
    dates = weekly_dates(len(values), start)
    r = c.post("/api/series", json={
        "name": name, "period": period,
        "dates": dates, "values": [float(v) for v in values],
    })
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _fit(c, sid, horizon=12, label=""):
    r = c.post("/api/fits", json={
        "series_id": sid, "horizon": horizon, "confidence": 0.95,
        "interval_method": "analytic", "auto": False,
        "trend_kind": "add", "seasonal_kind": "add",
        "locks": {}, "label": label,
    })
    job = _wait(c, r.json()["id"])
    assert job["status"] == "done", job
    return job["created_id"]


def _build_reference_tree(c, suffix=""):
    r = c.post("/api/hierarchy/trees", json={"name": f"参考树{suffix}"})
    assert r.status_code == 200, r.text
    tree_id = r.json()["id"]
    root_id = r.json()["nodes"][0]["id"]

    vals = reference_store_values()
    sids = [
        _create_series(c, f"门店一{suffix}", vals[0]),
        _create_series(c, f"门店二{suffix}", vals[1]),
        _create_series(c, f"门店三{suffix}", vals[2]),
    ]
    fit_ids = [_fit(c, s) for s in sids]

    def add_region(name):
        r = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
            "parent_id": root_id, "kind": "region", "name": name,
        })
        assert r.status_code == 200, r.text
        return r.json()["id"]

    ra = add_region("区域甲")
    rb = add_region("区域乙")
    stores = []
    for rid, sid, nm in (
        (ra, sids[0], "门店一"), (ra, sids[1], "门店二"),
        (rb, sids[2], "门店三"),
    ):
        r = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
            "parent_id": rid, "kind": "store", "name": nm,
            "series_id": sid,
        })
        assert r.status_code == 200, r.text
        stores.append(r.json()["id"])
    return tree_id, sids, fit_ids, stores


def _hier_forecast(c, tree_id, **over):
    body = {
        "tree_id": tree_id, "horizon": 12, "confidence": 0.95,
        "interval_method": "analytic", "trend_kind": "add",
        "seasonal_kind": "add", "recompute_mode": "full",
    }
    body.update(over)
    r = c.post("/api/hierarchy/forecasts", json=body)
    assert r.status_code == 200, r.text
    return _wait(c, r.json()["id"])


def _nodes_by_name(fc):
    out = {}
    for v in fc["result"]["nodes"].values():
        out.setdefault(v["name"], v)
    return out


def _node_by_kind(fc, kind):
    return next(
        v for v in fc["result"]["nodes"].values() if v["kind"] == kind
    )


def test_reference_flow_numbers_and_coherence(client):
    c, _ = client
    tree_id, sids, fit_ids, store_node_ids = _build_reference_tree(c, "-A")

    job = _hier_forecast(c, tree_id)
    assert job["status"] == "done", job
    fc = c.get(f"/api/hierarchy/forecasts/{job['created_id']}").json()
    assert fc["stale"] is False
    nodes = _nodes_by_name(fc)

    net = _node_by_kind(fc, "network")
    assert net["reconciled"]["point"][0] == pytest.approx(428.4, abs=1e-9)

    for name, v in nodes.items():
        b = np.asarray(v["base"]["point"])
        r = np.asarray(v["reconciled"]["point"])
        assert np.max(np.abs(r - b)) <= 0.01, name
        # interval widths monotone non-decreasing in h
        width = np.asarray(v["reconciled"]["upper"]) - \
            np.asarray(v["reconciled"]["lower"])
        assert np.all(np.diff(width) >= -1e-9), name

    # coherence through the API payload, all three levels
    by_name = nodes
    for name, v in by_name.items():
        if v["kind"] == "store":
            continue
        kids = [x for x in by_name.values()
                if x["parent_id"] == v["node_id"]]
        total = sum(
            (np.asarray(ch["reconciled"]["point"]) for ch in kids),
            start=np.zeros(12),
        )
        np.testing.assert_allclose(
            v["reconciled"]["point"], total, rtol=1e-6, atol=0,
            err_msg=name,
        )
    # region 乙 == store 3 date by date
    np.testing.assert_allclose(
        by_name["区域乙"]["reconciled"]["point"],
        by_name["门店三"]["reconciled"]["point"], atol=1e-12,
    )
    # fit refs recorded per leaf node
    refs = fc["fit_refs"]
    for nid in store_node_ids:
        assert str(nid) in refs


def test_attachment_rejected_with_reason(client):
    c, _ = client
    # mismatched period
    r = c.post("/api/hierarchy/trees", json={"name": "对齐拒绝树"})
    tree_id = r.json()["id"]
    root_id = r.json()["nodes"][0]["id"]
    vals = reference_store_values()
    s1 = _create_series(c, "对齐-门店一", vals[0], period=52)
    s2 = _create_series(c, "对齐-门店二", vals[1], period=12)
    rid = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": root_id, "kind": "region", "name": "区域",
    }).json()["id"]
    r = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": rid, "kind": "store", "name": "对齐-门店一",
        "series_id": s1,
    })
    assert r.status_code == 200
    r = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": rid, "kind": "store", "name": "对齐-门店二",
        "series_id": s2,
    })
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "对齐-门店二" in detail and "周期" in detail
    # nothing was attached
    tree = c.get(f"/api/hierarchy/trees/{tree_id}").json()
    assert sum(1 for n in tree["nodes"] if n["kind"] == "store") == 1

    # duplicate series rejected
    r = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": rid, "kind": "store", "name": "重复",
        "series_id": s1,
    })
    assert r.status_code == 400 and "重复挂接" in r.json()["detail"]

    # mismatched start week
    s3 = _create_series(
        c, "对齐-门店三", vals[2], period=52,
        start=date(2022, 1, 10),
    )
    r = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": rid, "kind": "store", "name": "对齐-门店三",
        "series_id": s3,
    })
    assert r.status_code == 400 and "起始周" in r.json()["detail"]


def test_local_vs_full_recompute_agree_and_staleness(client):
    c, _ = client
    tree_id, sids, fit_ids, _ = _build_reference_tree(c, "-B")

    job1 = _hier_forecast(c, tree_id, label="v1")
    fc1 = c.get(f"/api/hierarchy/forecasts/{job1['created_id']}").json()

    # Refit store 1 with a genuinely different model (no trend): on these
    # noiseless trending data this base forecast genuinely moves, even though
    # the fit itself is valid.  This mirrors a planner locking/changing the
    # model on one store.
    r = c.post("/api/fits", json={
        "series_id": sids[0], "horizon": 12, "confidence": 0.95,
        "interval_method": "analytic", "auto": False,
        "trend_kind": "none", "seasonal_kind": "add",
        "locks": {"alpha": 0.001}, "label": "改无趋势",
    })
    _wait(c, r.json()["id"])

    # old result is now stale
    old = c.get(f"/api/hierarchy/forecasts/{job1['created_id']}").json()
    assert old["stale"] is True
    assert any("门店一" in why for why in old["stale_reasons"])

    # local recompute based on v1
    job_local = _hier_forecast(
        c, tree_id, recompute_mode="local",
        based_on_id=job1["created_id"], label="local",
    )
    assert job_local["status"] == "done", job_local
    # full recompute
    job_full = _hier_forecast(c, tree_id, recompute_mode="full", label="full")
    assert job_full["status"] == "done", job_full

    fcl = c.get(
        f"/api/hierarchy/forecasts/{job_local['created_id']}"
    ).json()
    fcf = c.get(
        f"/api/hierarchy/forecasts/{job_full['created_id']}"
    ).json()

    assert fcl["stale"] is False and fcf["stale"] is False
    nl, nf = _nodes_by_name(fcl), _nodes_by_name(fcf)
    for name in nl:
        for key in ("base", "reconciled"):
            np.testing.assert_allclose(
                nl[name][key]["point"], nf[name][key]["point"],
                atol=1e-9, err_msg=f"{name} {key}",
            )
            # bands must be generated under this run's level by both paths
            np.testing.assert_allclose(
                nl[name][key]["lower"], nf[name][key]["lower"],
                atol=1e-9, err_msg=f"{name} {key} lower",
            )
            np.testing.assert_allclose(
                nl[name][key]["upper"], nf[name][key]["upper"],
                atol=1e-9, err_msg=f"{name} {key} upper",
            )

    # unaffected branch (region 乙 + stores 2,3) keeps the v1 numbers;
    # changed path (region 甲, network, store 1) moves.
    n1 = _nodes_by_name(fc1)
    for name in ("区域乙", "门店二", "门店三"):
        np.testing.assert_allclose(
            nl[name]["reconciled"]["point"],
            n1[name]["reconciled"]["point"], atol=1e-12, err_msg=name,
        )
    # changed path genuinely moves (no-trend fit drifts from the truth);
    # the exact-equivalence requirement is the 1e-9 check above, here we
    # only need a detectable, realistic difference.
    assert not np.allclose(
        nl["门店一"]["reconciled"]["point"],
        n1["门店一"]["reconciled"]["point"], atol=1e-4,
    )


def test_concurrent_refit_uses_pinned_versions(client):
    from app.db import session_scope
    from app.hierarchy import service as hservice
    from app.hierarchy.schemas import HierarchyForecastRequest

    c, _ = client
    tree_id, sids, fit_ids, _ = _build_reference_tree(c, "-C")
    # latest fits before the run
    before = {sid: fid for sid, fid in zip(sids, fit_ids)}

    captured = {}

    def hook(snap, fit_snaps):
        # Snapshot taken: another planner refits store 2 NOW.
        r = c.post("/api/fits", json={
            "series_id": sids[1], "horizon": 12, "confidence": 0.95,
            "interval_method": "analytic", "auto": False,
            "trend_kind": "add", "seasonal_kind": "add",
            "locks": {"alpha": 0.2}, "label": "并发重拟",
        })
        job = _wait(c, r.json()["id"])
        captured["new_fit_id"] = job["created_id"]
        captured["snapped_fit_id"] = fit_snaps[sids[1]].id

    req = HierarchyForecastRequest(tree_id=tree_id)
    out = hservice.run_hierarchy_forecast_job(
        req, report=lambda p, s: None, snapshot_hook=hook
    )
    # the run used the OLD fit, not the one inserted mid-run
    assert captured["new_fit_id"] != captured["snapped_fit_id"]
    fc = c.get(
        f"/api/hierarchy/forecasts/{out['created_id']}"
    ).json()
    nodes = _nodes_by_name(fc)
    store2_fit_ref = next(
        v for v in nodes.values() if v["name"] == "门店二"
    )["base"]["fit_id"]
    assert store2_fit_ref == captured["snapped_fit_id"]
    assert store2_fit_ref != captured["new_fit_id"]
    # and it is flagged stale immediately
    assert fc["stale"] is True
    assert any("门店二" in why for why in fc["stale_reasons"])


def test_node_failure_persists_nothing_and_names_node(client):
    c, _ = client
    r = c.post("/api/hierarchy/trees", json={"name": "失败树"})
    tree_id = r.json()["id"]
    root_id = r.json()["nodes"][0]["id"]

    # leaf series containing a zero: additive fit works, multiplicative
    # aggregate fit must fail.
    vals = reference_store_values()
    v_bad = vals[0].copy()
    v_bad[60] = 0.0
    sid = _create_series(c, "失败-门店", v_bad)
    _fit(c, sid)  # additive pinned fit exists

    rid = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": root_id, "kind": "region", "name": "问题区域",
    }).json()["id"]
    r = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": rid, "kind": "store", "name": "失败-门店",
        "series_id": sid,
    })
    assert r.status_code == 200

    from app.db import session_scope
    from app.hierarchy import storage as hstorage
    with session_scope() as db:
        before_count = len(hstorage.list_forecasts(db, tree_id))

    job = _hier_forecast(c, tree_id, seasonal_kind="mul")
    assert job["status"] == "error"
    assert "问题区域" in (job["error"] or "")

    with session_scope() as db:
        after = hstorage.list_forecasts(db, tree_id)
    assert len(after) == before_count == 0


def test_missing_leaf_fit_reports_node(client):
    c, _ = client
    r = c.post("/api/hierarchy/trees", json={"name": "无拟合树"})
    tree_id = r.json()["id"]
    root_id = r.json()["nodes"][0]["id"]
    vals = reference_store_values()
    sid = _create_series(c, "裸门店", vals[0])  # no fit at all
    rid = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": root_id, "kind": "region", "name": "区域",
    }).json()["id"]
    c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": rid, "kind": "store", "name": "裸门店",
        "series_id": sid,
    })
    job = _hier_forecast(c, tree_id)
    assert job["status"] == "error"
    assert "裸门店" in (job["error"] or "")


def test_hierarchy_backtest_flow_and_structural_staleness(client):
    c, _ = client
    tree_id, sids, fit_ids, _ = _build_reference_tree(c, "-D")

    r = c.post("/api/hierarchy/backtests", json={
        "tree_id": tree_id,
        "origin_start": 2 * 52 + 2, "horizon": 6, "stride": 30,
        "confidence": 0.95, "interval_method": "analytic",
        "trend_kind": "add", "seasonal_kind": "add",
    })
    job = _wait(c, r.json()["id"], timeout=180)
    assert job["status"] == "done", job
    bt = c.get(f"/api/hierarchy/backtests/{job['created_id']}").json()
    levels = bt["result"]["by_level"]
    assert set(levels) == {"network", "region", "store"}
    for lvl in levels.values():
        assert np.isfinite(lvl["base_mae"])
        assert np.isfinite(lvl["reconciled_mase"])
        assert {"base_mae", "reconciled_mae", "base_mase",
                "reconciled_mase"} <= set(lvl)
    # noiseless: reconciled MAE tiny at every level
    for lvl in levels.values():
        assert lvl["reconciled_mae"] < 1.0
    assert len(bt["result"]["origins"]) >= 1

    # structural change marks both forecast and backtest stale
    _hier_forecast(c, tree_id)
    tree_detail = c.get(f"/api/hierarchy/trees/{tree_id}").json()
    new_region = c.post(f"/api/hierarchy/trees/{tree_id}/nodes", json={
        "parent_id": tree_detail["nodes"][0]["id"], "kind": "region",
        "name": "空挂区域",
    }).json()
    new_region_id = new_region["id"]
    detail = c.get(f"/api/hierarchy/trees/{tree_id}").json()
    # empty region: browsable but not forecast-ready
    assert detail["valid"] is True
    assert detail["ready_for_forecast"] is False
    assert "空挂区域" in detail["forecast_error"]
    fc_list = c.get(f"/api/hierarchy/trees/{tree_id}/forecasts").json()
    assert all(f["stale"] for f in fc_list)
    bt2 = c.get(f"/api/hierarchy/backtests/{job['created_id']}").json()
    assert bt2["stale"] is True

    # remove the dangling node -> tree ready again
    r = c.delete(
        f"/api/hierarchy/trees/{tree_id}/nodes/{new_region_id}"
    )
    assert r.status_code == 200
    detail = c.get(f"/api/hierarchy/trees/{tree_id}").json()
    assert detail["valid"] is True
    assert detail["ready_for_forecast"] is True
