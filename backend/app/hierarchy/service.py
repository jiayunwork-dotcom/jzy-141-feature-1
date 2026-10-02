"""Service orchestration for hierarchical forecasts and backtests.

Concurrency contract
~~~~~~~~~~~~~~~~~~~~
A hierarchy job snapshots *everything* it is going to use inside ONE database
transaction at start:

* the tree rows + structure revision,
* every leaf series (immutable rows, but their fit history is not),
* every leaf's *latest* Fit row id, pinned for the whole job.

All base forecasts are then computed detached from the DB; the result row and
its ``fit_refs`` are written in a single final transaction.  Consequently the
persisted fit refs are by construction exactly the fits the numbers came from
— a planner refitting a store mid-run can only insert a new Fit row after the
snapshot, which marks the result stale but cannot mix versions inside it.

Staleness
~~~~~~~~~
A stored result is stale when

* the tree's ``structure_revision`` moved (nodes added/removed), or
* any leaf's *latest* fit id differs from the referenced id
  (someone refit a store, even with the same locked parameters).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

from ..db import session_scope
from ..kernels.hw import ModelError
from ..kernels.reconcile import (
    NodeBaseForecast,
    base_forecast_from_fit,
    fit_base_forecast,
    reconcile_payload,
)
from .. import storage as series_storage
from . import storage as hstorage
from .backtest import run_hierarchy_backtest
from .tree import HierarchyError, TreeSnapshot, build_snapshot


class _SeriesSnap:
    def __init__(self, row):
        self.id = row.id
        self.name = row.name
        self.period = row.period
        self.dates = list(row.dates)
        self.values = list(row.values)


def snapshot_tree(
    db, tree_id: int
) -> Tuple[TreeSnapshot, Dict[int, Any], int]:
    """Load tree + leaf series + latest leaf fits in one session/transaction.

    Returns (snapshot, latest_fit_by_series, structure_revision).
    Caller keeps the returned dict alive; the DB session may close after.
    """
    tree = hstorage.get_tree(db, tree_id)
    if tree is None:
        raise HierarchyError(f"层级树 {tree_id} 不存在。")
    rows = hstorage.list_nodes(db, tree_id)
    series_ids = {r.series_id for r in rows if r.series_id is not None}
    series_by_id = {}
    for sid in series_ids:
        s = series_storage.get_series(db, sid)
        if s is None:
            series_by_id = {}  # force build_snapshot to name the missing node
            break
        series_by_id[sid] = _SeriesSnap(s)
    snap = build_snapshot(
        tree.id, tree.name, tree.structure_revision, rows, series_by_id
    )
    latest_fit: Dict[int, Any] = {}
    for leaf in snap.leaves():
        fits = series_storage.list_fits(db, leaf.series_id)
        latest_fit[leaf.series_id] = fits[0] if fits else None
    return snap, latest_fit, tree.structure_revision


def _future_dates(dates: List[str], horizon: int) -> List[str]:
    last = datetime.fromisoformat(dates[-1]).date()
    return [
        (last + timedelta(weeks=k)).isoformat()
        for k in range(1, horizon + 1)
    ]


def _validate_run_spec(req) -> None:
    if req.horizon < 1:
        raise HierarchyError("horizon 必须 >= 1。")
    if not 0.5 <= req.confidence < 1:
        raise HierarchyError("confidence 必须在 [0.5, 1)。")
    if req.interval_method not in ("analytic", "simulate"):
        raise HierarchyError("interval_method 只能是 analytic/simulate。")
    if req.trend_kind not in ("none", "add", "add_damped"):
        raise HierarchyError("trend_kind 必须是 none/add/add_damped。")
    if req.seasonal_kind not in ("add", "mul"):
        raise HierarchyError("seasonal_kind 必须是 add/mul。")
    if req.recompute_mode not in ("full", "local"):
        raise HierarchyError("recompute_mode 只能是 full/local。")


def run_hierarchy_forecast_job(
    req, report, snapshot_hook=None
) -> Dict[str, Any]:
    """Background job: base forecasts for every node + reconciliation.

    ``snapshot_hook`` is a test-only callable invoked between the snapshot
    transaction and computation, to deterministically interleave a refit.
    """
    _validate_run_spec(req)
    report(0.02, "读取树与最新拟合版本")
    with session_scope() as db:
        snap, latest_fits, revision = snapshot_tree(db, req.tree_id)
        # Detach fit objects: copy the columns we need into lightweight snaps
        # so a concurrent session's expiry/refresh can never mutate them.
        fit_snaps = {
            sid: _detach_fit(f) for sid, f in latest_fits.items() if f is not None
        }
        prior = None
        if req.recompute_mode == "local":
            if req.based_on_id is None:
                raise HierarchyError(
                    "局部重算需要指定 based_on_id（上一次层级预测）。"
                )
            prior = hstorage.get_forecast(db, req.based_on_id)
            if prior is None:
                raise HierarchyError("引用的历史层级预测不存在。")
            if prior.tree_id != req.tree_id:
                raise HierarchyError("based_on_id 不属于这棵树。")
            if prior.horizon != req.horizon:
                raise HierarchyError(
                    "局部重算的步长必须与引用的层级预测一致。"
                )
            prior_result = {
                "revision": prior.structure_revision,
                "result": dict(prior.result),
            }
        else:
            prior_result = None
    revision_at_snapshot = revision

    if snapshot_hook is not None:
        snapshot_hook(snap, fit_snaps)

    missing = [
        leaf.name for leaf in snap.leaves()
        if leaf.series_id not in fit_snaps
    ]
    if missing:
        raise ModelError(
            "以下门店节点还没有任何拟合结果，请先在单序列页完成拟合："
            + "、".join(f"『{n}』" for n in missing)
        )

    report(0.1, "计算门店基础预测")
    base: Dict[int, NodeBaseForecast] = {}
    fit_refs: Dict[str, int] = {}
    for leaf in snap.leaves():
        f = fit_snaps[leaf.series_id]
        fit_refs[str(leaf.id)] = f.id

    reused: set[int] = set()
    if req.recompute_mode == "local":
        old_rev = prior_result["revision"]
        if old_rev != revision_at_snapshot:
            raise HierarchyError(
                "树结构在上次层级预测后发生过变化，不能局部重算，请整棵重来。"
            )
        # Leaf base columns of unchanged stores are reused as-is: the fit id
        # pins both the numbers and the bands.  Aggregate nodes are all
        # refit below — their base fit is deterministic and cheap, and their
        # bands must use THIS run's confidence/method, while their point
        # forecast only depends on the (unchanged) summed history, so it
        # matches a full rerun to 1e-9.  Reconciled values are bottom-up,
        # hence only the changed leaf -> ancestors path can move.
        old_nodes = prior_result["result"]["nodes"]
        changed_leaves = set(_changed_leaf_nodes(
            snap, prior_result["result"], fit_refs
        ))
        for leaf in snap.leaves():
            if leaf.id not in changed_leaves:
                ob = old_nodes[str(leaf.id)]["base"]
                if ob.get("fit_id") == fit_snaps[leaf.series_id].id:
                    base[leaf.id] = _base_from_dict(leaf, ob)
                    reused.add(leaf.id)

    for leaf in snap.leaves():
        if leaf.id in base:
            continue
        f = fit_snaps[leaf.series_id]
        base[leaf.id] = base_forecast_from_fit(
            leaf, f, req.horizon, req.confidence, req.interval_method
        )

    report(0.35, "拟合区域/全网基础预测")
    failures: List[str] = []
    aggregates = [n for n in snap.nodes.values() if n.kind != "store"]
    todo = [n for n in aggregates if n.id not in base]
    for i, node in enumerate(todo):
        try:
            base[node.id] = fit_base_forecast(
                node,
                node.history_values,
                snap.period,
                req.trend_kind,
                req.seasonal_kind,
                req.horizon,
                req.confidence,
                req.interval_method,
            )
        except Exception as exc:  # noqa: BLE001
            kind = "全网" if node.kind == "network" else "区域"
            failures.append(f"{kind}节点『{node.name}』拟合失败：{exc}")
        report(0.35 + 0.45 * (i + 1) / max(len(todo), 1),
               f"聚合节点拟合 {i + 1}/{len(todo)}")
    if failures:
        # Atomicity: nothing is persisted; the job reports every bad node.
        raise ModelError("；".join(failures))

    report(0.85, "自底向上调和")
    future_dates = _future_dates(snap.root.history_dates, req.horizon)
    payload = reconcile_payload(snap, base, future_dates)
    payload["recompute_mode"] = req.recompute_mode
    payload["reused_aggregate_node_ids"] = sorted(reused)

    report(0.95, "写入层级预测")
    with session_scope() as db:
        # Re-check the tree still exists and revision did not move during the
        # run (a structural change mid-run invalidates alignment).
        tree = hstorage.get_tree(db, req.tree_id)
        if tree is None:
            raise HierarchyError("树在计算过程中被删除。")
        if tree.structure_revision != revision_at_snapshot:
            raise HierarchyError(
                "树结构在计算过程中发生了变化，请重新发起层级预测。"
            )
        obj = hstorage.create_forecast(db, req.tree_id, {
            "label": req.label or "",
            "horizon": req.horizon,
            "confidence": req.confidence,
            "interval_method": req.interval_method,
            "trend_kind": req.trend_kind,
            "seasonal_kind": req.seasonal_kind,
            "structure_revision": revision_at_snapshot,
            "recompute_mode": req.recompute_mode,
            "fit_refs": fit_refs,
            "result": payload,
        })
        fc_id = obj.id
    return {"created_id": fc_id, "result": {"hierarchy_forecast_id": fc_id}}


class _FitSnap:
    __slots__ = (
        "id", "trend_kind", "seasonal_kind", "params", "residuals",
        "fitted", "forecast", "final_state", "sse", "aic",
    )

    def __init__(self, fit):
        self.id = fit.id
        self.trend_kind = fit.trend_kind
        self.seasonal_kind = fit.seasonal_kind
        self.params = dict(fit.params or {})
        self.residuals = list(fit.residuals or [])
        self.fitted = list(fit.fitted or [])
        self.forecast = dict(fit.forecast or {})
        self.final_state = dict(fit.final_state or {})
        self.sse = fit.sse
        self.aic = fit.aic


def _detach_fit(fit) -> _FitSnap:
    return _FitSnap(fit)


def _changed_leaf_nodes(
    snap: TreeSnapshot, prior_result: dict, fit_refs: Dict[str, int]
) -> List[int]:
    old = prior_result["nodes"]
    changed = []
    for leaf in snap.leaves():
        old_fit_id = (
            old.get(str(leaf.id), {}).get("base", {}).get("fit_id")
        )
        if old_fit_id != fit_refs[str(leaf.id)]:
            changed.append(leaf.id)
    return changed


def _base_from_dict(node, d: dict) -> NodeBaseForecast:
    return NodeBaseForecast(
        node_id=node.id,
        series_id=node.series_id,
        trend_kind=d["trend_kind"],
        seasonal_kind=d["seasonal_kind"],
        point=np.asarray(d["point"], dtype=float),
        lower=np.asarray(d["lower"], dtype=float),
        upper=np.asarray(d["upper"], dtype=float),
        sse=float(d.get("sse", 0.0)),
        aic=float(d.get("aic", 0.0)),
        residual_std=float(d.get("residual_std", 0.0)),
        fit_id=d.get("fit_id"),
    )


def staleness_info(db, tree_id: int) -> Dict[int, Dict[str, Any]]:
    """Map hierarchy_forecast_id -> {stale, reasons, latest_fit_ids}."""
    tree = hstorage.get_tree(db, tree_id)
    info: Dict[int, Dict[str, Any]] = {}
    if tree is None:
        return info
    rows = hstorage.list_nodes(db, tree_id)
    series_ids = [r.series_id for r in rows if r.series_id is not None]
    latest = {}
    for sid in series_ids:
        fits = series_storage.list_fits(db, sid)
        latest[sid] = fits[0].id if fits else None
    for fc in hstorage.list_forecasts(db, tree_id):
        reasons = []
        if fc.structure_revision != tree.structure_revision:
            reasons.append("树结构在上次层级预测后发生了变化（增删节点）。")
        for row in rows:
            if row.series_id is None:
                continue
            used = (fc.fit_refs or {}).get(str(row.id))
            cur = latest.get(row.series_id)
            if used != cur:
                reasons.append(
                    f"门店节点『{row.name}』有更新的拟合"
                    f"（引用 fit_id={used}，最新 fit_id={cur}）。"
                )
        info[fc.id] = {"stale": bool(reasons), "reasons": reasons}
    return info


def forecast_out(db, fc) -> dict:
    info = staleness_info(db, fc.tree_id)
    flag = info.get(fc.id, {"stale": False, "reasons": []})
    return {
        "id": fc.id,
        "tree_id": fc.tree_id,
        "created_at": fc.created_at.isoformat() if fc.created_at else "",
        "label": fc.label,
        "horizon": fc.horizon,
        "confidence": fc.confidence,
        "interval_method": fc.interval_method,
        "trend_kind": fc.trend_kind,
        "seasonal_kind": fc.seasonal_kind,
        "structure_revision": fc.structure_revision,
        "recompute_mode": fc.recompute_mode,
        "fit_refs": fc.fit_refs or {},
        "stale": flag["stale"],
        "stale_reasons": flag["reasons"],
        "result": fc.result,
    }


def run_hierarchy_backtest_job(req, report) -> Dict[str, Any]:
    report(0.02, "读取树结构")
    with session_scope() as db:
        tree = hstorage.get_tree(db, req.tree_id)
        if tree is None:
            raise HierarchyError(f"层级树 {req.tree_id} 不存在。")
        revision = tree.structure_revision
        rows = hstorage.list_nodes(db, req.tree_id)
        series_by_id = {}
        for r in rows:
            if r.series_id is not None:
                s = series_storage.get_series(db, r.series_id)
                if s is not None:
                    series_by_id[s.id] = _SeriesSnap(s)
        snap = build_snapshot(
            tree.id, tree.name, tree.structure_revision, rows, series_by_id
        )

    report(0.08, "逐原点整树重拟与调和")
    outcome = run_hierarchy_backtest(
        snap,
        origin_start=int(req.origin_start),
        horizon=int(req.horizon),
        stride=int(req.stride),
        trend_kind=req.trend_kind,
        seasonal_kind=req.seasonal_kind,
        confidence=float(req.confidence),
        interval_method=req.interval_method,
        progress_cb=lambda p, stage: report(0.08 + 0.87 * p, stage),
    )
    result = outcome.to_dict()
    report(0.97, "保存层级回测结果")
    with session_scope() as db:
        obj = hstorage.create_backtest(db, req.tree_id, {
            "label": req.label or "",
            "origin_start": int(req.origin_start),
            "horizon": int(req.horizon),
            "stride": int(req.stride),
            "confidence": float(req.confidence),
            "interval_method": req.interval_method,
            "trend_kind": req.trend_kind,
            "seasonal_kind": req.seasonal_kind,
            "structure_revision": revision,
            "result": result,
        })
        bt_id = obj.id
    return {"created_id": bt_id, "result": {"hierarchy_backtest_id": bt_id}}


def backtest_out(db, bt) -> dict:
    tree = hstorage.get_tree(db, bt.tree_id)
    stale = bool(tree and bt.structure_revision != tree.structure_revision)
    return {
        "id": bt.id,
        "tree_id": bt.tree_id,
        "created_at": bt.created_at.isoformat() if bt.created_at else "",
        "label": bt.label,
        "origin_start": bt.origin_start,
        "horizon": bt.horizon,
        "stride": bt.stride,
        "confidence": bt.confidence,
        "interval_method": bt.interval_method,
        "trend_kind": bt.trend_kind,
        "seasonal_kind": bt.seasonal_kind,
        "structure_revision": bt.structure_revision,
        "stale": stale,
        "result": bt.result,
    }
