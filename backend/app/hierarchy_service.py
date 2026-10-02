"""层级服务层：快照并发控制、基础预测、WLS 调和、过期判定、层级回测。

并发版本记录
============
层级任务启动时在 **同一个数据库事务** 里确定并读出每个门店要用到的那一
次拟合（默认各门店当前最新拟合；PostgreSQL 下顺带对这些 fit 行加
``FOR UPDATE``），拿到的是按 **fit id 钉死** 的整行快照，之后全部计算只
使用内存值。关键在于既有 ``fits`` 行是 **只增不改** 的：任务跑到一半时
别的计划员给某店重新拟合只会 **插入新行**，既不改动被快照的旧行，也不
影响旧 id 的解析，因此任务不会出现「前几个节点旧、后几个节点新」，结果
里 ``fit_refs`` 记录的就是它实际使用的那一组 fit id。该新拟合一旦提交，
这次结果在读时即被标记为过期（见过期判定）。

失败不留半成品
==============
任何节点（门店缺拟合、聚合节点模型被拒绝等）抛错都发生在写库之前，
整次结果只有在全部节点基础预测 + 调和都成功后才在单个事务里落库。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import select

from .db import session_scope
from .hierarchy_storage import (
    TreeInfo,
    create_hierarchy_backtest,
    create_hierarchy_forecast,
    create_tree,
    get_hierarchy_backtest,
    get_hierarchy_forecast,
    get_store_refs,
    latest_fit_ids_for_series,
    list_hierarchy_backtests,
    list_hierarchy_forecasts,
    list_trees,
    load_tree,
)
from .hierarchy_utils import (
    LeafSpec,
    aggregate_history,
    check_tree_assignment,
)
from .kernels import hierarchy as hk
from .kernels.hier_runner import (
    HierarchyShape,
    base_forecast_from_fit,
    base_forecast_for_values,
    run_hierarchy_backtest,
)
from .kernels.hw import ModelError
from .models import Fit


class HierarchyError(ModelError):
    """层级挂接/结构类错误（400）。"""


# ---------------------------------------------------------------------------
# 树管理
# ---------------------------------------------------------------------------

def build_tree(req, db) -> Dict[str, Any]:
    if not req.regions:
        raise HierarchyError("至少需要一个区域。")
    all_series_ids = [sid for r in req.regions for sid in r.series_ids]
    if not all_series_ids:
        raise HierarchyError("每个区域至少要选一个门店序列。")
    if len(set(all_series_ids)) != len(all_series_ids):
        raise HierarchyError("同一门店不能在树中重复挂接。")

    refs = get_store_refs(db, all_series_ids)
    missing = sorted(set(all_series_ids) - set(refs))
    if missing:
        raise HierarchyError(f"门店序列不存在：{missing}")

    region_leaves: List[tuple[str, List[LeafSpec]]] = []
    for r in req.regions:
        leaves = []
        for sid in r.series_ids:
            ref = refs[sid]
            leaves.append(LeafSpec(
                series_id=ref.series_id, name=ref.name, period=ref.period,
                dates=ref.dates, values=ref.values,
            ))
        region_leaves.append((r.name, leaves))

    check = check_tree_assignment(region_leaves)
    if not check.ok:
        # 结构化错误：前端逐条展示「哪个区域的哪个子节点、差在哪里」。
        raise HierarchyError({
            "message": "挂接被拒绝，存在对不上的子节点。",
            "issues": [i.to_dict() for i in check.issues],
        })

    period = refs[all_series_ids[0]].period
    dates = refs[all_series_ids[0]].dates
    regions_payload = [
        {"name": name, "stores": [refs[leaf.series_id] for leaf in leaves]}
        for name, leaves in region_leaves
    ]
    tree = create_tree(
        db, name=req.name, period=period, dates=list(dates),
        regions=regions_payload, network_name=req.network_name,
    )
    return {"tree_id": tree.id}


def validate_assignment(req, db) -> Dict[str, Any]:
    """不落库的挂接预检（前端选入门店时即时反馈）。"""
    all_series_ids = [sid for r in req.regions for sid in r.series_ids]
    refs = get_store_refs(db, all_series_ids) if all_series_ids else {}
    missing = sorted(set(all_series_ids) - set(refs))
    region_leaves = []
    for r in req.regions:
        region_leaves.append((r.name, [
            LeafSpec(refs[sid].series_id, refs[sid].name, refs[sid].period,
                     refs[sid].dates, refs[sid].values)
            for sid in r.series_ids if sid in refs
        ]))
    check = check_tree_assignment(region_leaves) if region_leaves else None
    return {
        "ok": (check is not None and check.ok and not missing
               and len(all_series_ids) == len(set(all_series_ids))),
        "missing_series": missing,
        "issues": [i.to_dict() for i in check.issues] if check else [],
    }


def tree_detail(db, tree_id: int) -> Dict[str, Any]:
    tree = load_tree(db, tree_id)
    if tree is None:
        raise HierarchyError(f"层级树 {tree_id} 不存在。")
    refs = get_store_refs(db, [
        n.series_id for n in tree.nodes.values() if n.series_id is not None
    ])
    desc = tree.descendants_bottom()

    histories: Dict[str, List[float]] = {}
    for nid in tree.ordered_node_ids():
        n = tree.nodes[nid]
        if n.kind == "store":
            vals = list(refs[n.series_id].values)
        else:
            vals = aggregate_history([
                list(refs[tree.nodes[b].series_id].values)
                for b in desc[nid]
            ]).tolist()
        histories[str(nid)] = [float(v) for v in vals]

    nodes_out = []
    for nid in tree.ordered_node_ids():
        n = tree.nodes[nid]
        nodes_out.append({
            "node_id": nid,
            "name": n.name,
            "kind": n.kind,
            "level": n.level,
            "parent_id": n.parent_id,
            "series_id": n.series_id,
            "children": list(n.children),
        })
    return {
        "id": tree.tree_id,
        "name": tree.name,
        "period": tree.period,
        "dates": list(tree.dates),
        "root_id": tree.root_id,
        "nodes": nodes_out,
        "histories": histories,
    }


# ---------------------------------------------------------------------------
# 任务开始时的一致性快照
# ---------------------------------------------------------------------------

def _snapshot(db, tree: TreeInfo, req) -> Dict[str, Any]:
    """在单事务内读出每个门店实际要使用的拟合版本与数据。"""
    leaf_nodes = [tree.nodes[nid] for nid in tree.bottom_ids]
    series_ids = [n.series_id for n in leaf_nodes]
    refs = get_store_refs(db, series_ids)

    chosen: Dict[int, int] = {}
    explicit = {int(k): int(v) for k, v in (req.base_fit_ids or {}).items()}
    if explicit:
        for sid in series_ids:
            if sid not in explicit:
                raise ModelError(
                    f"门店序列 {sid} 未指定 base_fit_ids，必须整组指定。")
        chosen = explicit
    else:
        chosen = latest_fit_ids_for_series(db, series_ids)

    fit_rows: Dict[int, Fit] = {}
    if chosen:
        stmt = select(Fit).where(Fit.id.in_(list(chosen.values())))
        if db.bind.dialect.name != "sqlite":
            stmt = stmt.with_for_update()
        for row in db.scalars(stmt):
            fit_rows[row.id] = row

    snaps: Dict[int, Dict[str, Any]] = {}
    for node in leaf_nodes:
        sid = node.series_id
        if sid not in chosen:
            raise ModelError(
                f"门店节点「{node.name}」（序列 {sid}）还没有任何拟合，"
                "无法做层级预测；请先在单序列页完成一次拟合。")
        fit = fit_rows.get(chosen[sid])
        if fit is None or fit.series_id != sid:
            raise ModelError(
                f"门店节点「{node.name}」指定的拟合 {chosen[sid]} 不存在或不"
                f"属于序列 {sid}。")
        snaps[node.node_id] = {
            "series_id": sid,
            "fit_id": fit.id,
            "name": node.name,
            "values": list(refs[sid].values),
            "params": dict(fit.params),
            "final_state": dict(fit.final_state),
            "residuals": list(fit.residuals),
            "trend_kind": fit.trend_kind,
            "seasonal_kind": fit.seasonal_kind,
        }
    return {
        "leaf": snaps,
        "series_ids": series_ids,
        "dates": list(tree.dates),
    }


def _future_dates(dates: List[str], horizon: int) -> List[str]:
    last = datetime.fromisoformat(dates[-1]).date()
    return [(last + timedelta(weeks=k)).isoformat()
            for k in range(1, horizon + 1)]


def _model_spec(req) -> Any:
    from .kernels.hier_runner import NodeModelSpec
    return NodeModelSpec(
        auto=bool(req.auto), trend_kind=req.trend_kind,
        seasonal_kind=req.seasonal_kind, locks=dict(req.locks or {}),
    )


def _compute_all_bases(tree: TreeInfo, snap: Dict[str, Any], req,
                       report) -> Dict[int, Dict[str, Any]]:
    """所有节点的基础预测。叶子用被引用拟合；聚合节点用加总历史现拟合。

    任意节点失败在此抛出（带节点名），尚不落库。
    """
    spec = _model_spec(req)
    bases: Dict[int, Dict[str, Any]] = {}

    # 叶子：与单序列页同一个 forecast()，数字不变。
    for nid, ls in snap["leaf"].items():
        report(0.05 + 0.35 * (len(bases) / max(len(snap["leaf"]), 1)),
               f"门店基础预测：{ls['name']}")
        try:
            b = base_forecast_from_fit(
                ls["params"], ls["final_state"], ls["residuals"],
                horizon=int(req.horizon), confidence=float(req.confidence),
                interval_method=req.interval_method,
            )
        except ModelError as exc:
            raise ModelError(f"门店节点「{ls['name']}」预测失败：{exc}") from exc
        b["fit_id"] = ls["fit_id"]
        b["trend_kind"] = ls["trend_kind"]
        b["seasonal_kind"] = ls["seasonal_kind"]
        bases[nid] = b

    # 聚合节点：自底向上，用下级叶子完整历史加总后各自拟合。
    desc = tree.descendants_bottom()
    agg_ids = [nid for nid in tree.ordered_node_ids()
               if tree.nodes[nid].kind != "store"]
    for j, nid in enumerate(agg_ids):
        node = tree.nodes[nid]
        report(0.4 + 0.4 * ((j + 1) / max(len(agg_ids), 1)),
               f"聚合节点拟合：{node.name}")
        history = aggregate_history([
            snap["leaf"][b]["values"] for b in desc[nid]
        ])
        # base_forecast_for_values 自身已把失败包成「节点「名」…」。
        b = base_forecast_for_values(
            history, tree.period, int(req.horizon), spec,
            confidence=float(req.confidence),
            interval_method=req.interval_method, node_name=node.name,
        )
        bases[nid] = b
    return bases


def _reconcile_bases(tree: TreeInfo,
                     bases: Dict[int, Dict[str, Any]]) -> hk.ReconciledTree:
    ordered = tree.ordered_node_ids()
    node_bases = [
        hk.NodeBase(
            node_id=nid, level=tree.nodes[nid].level, name=tree.nodes[nid].name,
            point=bases[nid]["point"], lower=bases[nid]["lower"],
            upper=bases[nid]["upper"],
            residual_var=float(bases[nid]["residual_var"]),
            fit_id=bases[nid].get("fit_id"),
            trend_kind=bases[nid].get("trend_kind", ""),
            seasonal_kind=bases[nid].get("seasonal_kind", ""),
        )
        for nid in ordered
    ]
    return hk.reconcile(
        node_bases, tree.bottom_ids, tree.descendants_bottom(),
        apply_intervals=True,
    )


def _nodes_payload(tree: TreeInfo, bases, rec) -> List[Dict[str, Any]]:
    out = []
    for nid in tree.ordered_node_ids():
        n = tree.nodes[nid]
        sl = rec.node_slice(nid)
        b = bases[nid]
        fit_ref: Dict[str, Any]
        if n.kind == "store":
            fit_ref = {"fit_id": b["fit_id"]}
        else:
            fit_ref = {"agg_fit": {
                "params": b["params"],
                "sse": b["sse"],
                "aic": b["aic"],
                "residual_std": b["residual_std"],
                "trend_kind": b["trend_kind"],
                "seasonal_kind": b["seasonal_kind"],
            }}
        out.append({
            "node_id": nid,
            "kind": n.kind,
            "level": n.level,
            "name": n.name,
            "series_id": n.series_id,
            "fit_ref": fit_ref,
            "residual_var": float(sl["residual_var"]),
            "base": {
                "point": [float(x) for x in sl["base_point"]],
                "lower": [float(x) for x in sl["base_lower"]],
                "upper": [float(x) for x in sl["base_upper"]],
            },
            "rec": {
                "point": [float(x) for x in sl["rec_point"]],
                "lower": [float(x) for x in sl["rec_lower"]],
                "upper": [float(x) for x in sl["rec_upper"]],
            },
        })
    return out


def run_hierarchy_forecast_job(req, report) -> Dict[str, Any]:
    report(0.01, "读取树与拟合版本快照")
    with session_scope() as db:
        tree = load_tree(db, int(req.tree_id))
        if tree is None:
            raise ModelError(f"层级树 {req.tree_id} 不存在。")
        snap = _snapshot(db, tree, req)
        prior = None
        if req.mode == "auto":
            rows = list_hierarchy_forecasts(db, tree.tree_id)
            prior = rows[0] if rows else None
            prior_payload = _prior_bases(prior) if prior is not None else None
        else:
            prior_payload = None

    report(0.03, "计算各节点基础预测")
    if req.mode == "auto" and prior_payload is not None:
        bases = _recompute_partial(tree, snap, req, prior_payload, report)
    else:
        bases = _compute_all_bases(tree, snap, req, report)

    report(0.85, "WLS 调和（全树线性映射）")
    rec = _reconcile_bases(tree, bases)

    # 调和一致性自检（父=直接下级之和）；不通过就整体失败，绝不落库。
    resid = hk.coherence_residual(rec, tree.parent_children())
    if resid.size and float(np.max(np.abs(resid))) > 1e-9:
        raise ModelError(
            f"调和一致性自检失败：最大偏差 {float(np.max(np.abs(resid))):.3e}")

    report(0.92, "写入层级预测结果")
    nodes_payload = _nodes_payload(tree, bases, rec)
    fit_refs = {str(ls["series_id"]): ls["fit_id"]
                for ls in snap["leaf"].values()}
    with session_scope() as db:
        obj = create_hierarchy_forecast(db, tree.tree_id, {
            "label": req.label or "",
            "horizon": int(req.horizon),
            "confidence": float(req.confidence),
            "interval_method": req.interval_method,
            "auto": bool(req.auto),
            "trend_kind": req.trend_kind,
            "seasonal_kind": req.seasonal_kind,
            "locks": dict(req.locks or {}),
            "reconciliation": "wls",
            "future_dates": _future_dates(snap["dates"], int(req.horizon)),
            "nodes": nodes_payload,
            "fit_refs": fit_refs,
        })
        forecast_id = obj.id
    return {"created_id": forecast_id,
            "result": {"hierarchy_forecast_id": forecast_id}}


def _prior_bases(prior) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for entry in prior.nodes:
        b = entry["base"]
        out[int(entry["node_id"])] = {
            "point": np.asarray(b["point"], dtype=float),
            "lower": np.asarray(b["lower"], dtype=float),
            "upper": np.asarray(b["upper"], dtype=float),
            "residual_var": float(entry.get("residual_var", 0.0)),
            "trend_kind": (entry.get("fit_ref", {}) or {})
                .get("agg_fit", {}).get("trend_kind")
                or _kind_from_entry(entry, "trend_kind"),
            "seasonal_kind": (entry.get("fit_ref", {}) or {})
                .get("agg_fit", {}).get("seasonal_kind")
                or _kind_from_entry(entry, "seasonal_kind"),
            "fit_id": (entry.get("fit_ref", {}) or {}).get("fit_id"),
            "params": ((entry.get("fit_ref", {}) or {})
                       .get("agg_fit", {}) or {}).get("params"),
            "sse": ((entry.get("fit_ref", {}) or {})
                    .get("agg_fit", {}) or {}).get("sse"),
            "aic": ((entry.get("fit_ref", {}) or {})
                    .get("agg_fit", {}) or {}).get("aic"),
            "residual_std": ((entry.get("fit_ref", {}) or {})
                             .get("agg_fit", {}) or {}).get("residual_std"),
        }
    return out


def _kind_from_entry(entry, key):
    # 兼容旧结构里把 kind 放在顶层的情况（当前结构没有，防御性）。
    return entry.get(key, "")


def _recompute_partial(tree, snap, req, prior_bases, report):
    """局部重算：只重算换了拟合的门店 + 其祖先路径，其余节点复用旧基础值。

    调和仍对整棵树重跑（WLS 是稠密映射，见 kernels/hierarchy.py 文档）。
    """
    current_fit = {nid: ls["fit_id"] for nid, ls in snap["leaf"].items()}
    changed_leaves = {
        nid for nid, fid in current_fit.items()
        if int(prior_bases.get(nid, {}).get("fit_id", -1)) != int(fid)
    }
    affected: set[int] = set()
    for nid in changed_leaves:
        affected.update(tree.path_to_root_including(nid))

    spec = _model_spec(req)
    desc = tree.descendants_bottom()
    bases: Dict[int, Dict[str, Any]] = {}
    reused, recomputed = [], []
    for nid in tree.ordered_node_ids():
        if nid not in affected and nid in prior_bases:
            bases[nid] = dict(prior_bases[nid])
            reused.append(nid)
            continue
        recomputed.append(nid)
        n = tree.nodes[nid]
        if n.kind == "store":
            ls = snap["leaf"][nid]
            b = base_forecast_from_fit(
                ls["params"], ls["final_state"], ls["residuals"],
                horizon=int(req.horizon), confidence=float(req.confidence),
                interval_method=req.interval_method,
            )
            b["fit_id"] = ls["fit_id"]
            b["trend_kind"] = ls["trend_kind"]
            b["seasonal_kind"] = ls["seasonal_kind"]
        else:
            history = aggregate_history([
                snap["leaf"][b]["values"] for b in desc[nid]
            ])
            b = base_forecast_for_values(
                history, tree.period, int(req.horizon), spec,
                confidence=float(req.confidence),
                interval_method=req.interval_method, node_name=n.name,
            )
        bases[nid] = b
    report(0.5, f"局部重算 {len(recomputed)} 个节点，复用 {len(reused)} 个")
    return bases


# ---------------------------------------------------------------------------
# 过期判定（读时计算，不额外落库）
# ---------------------------------------------------------------------------

def _annotate_staleness(db, tree_id: int, obj) -> Dict[str, Any]:
    tree = load_tree(db, tree_id)
    series_ids = [n.series_id for n in tree.nodes.values()
                  if n.series_id is not None]
    latest = latest_fit_ids_for_series(db, series_ids)
    # JSON 对象的键在 PostgreSQL 上一定是字符串，SQLite 往返后可能被还原成
    # int；这里统一成 int(series_id) -> fit_id。
    refs = {int(k): int(v) for k, v in (obj.fit_refs or {}).items()}
    changed = [
        {"series_id": sid, "used_fit_id": refs.get(sid),
         "latest_fit_id": latest.get(sid)}
        for sid in series_ids
        if refs.get(sid) != latest.get(sid)
    ]
    return {
        "id": obj.id,
        "tree_id": obj.tree_id,
        "created_at": obj.created_at.isoformat() if obj.created_at else "",
        "label": obj.label,
        "horizon": obj.horizon,
        "confidence": obj.confidence,
        "interval_method": obj.interval_method,
        "auto": obj.auto,
        "trend_kind": obj.trend_kind,
        "seasonal_kind": obj.seasonal_kind,
        "locks": obj.locks,
        "reconciliation": obj.reconciliation,
        "future_dates": list(obj.future_dates or []),
        "nodes": list(obj.nodes or []),
        "fit_refs": {str(k): int(v) for k, v in refs.items()},
        "is_stale": bool(changed),
        "stale_detail": changed,
    }


def forecast_out(db, obj) -> Dict[str, Any]:
    return _annotate_staleness(db, obj.tree_id, obj)


def backtest_out(obj) -> Dict[str, Any]:
    return {
        "id": obj.id,
        "tree_id": obj.tree_id,
        "created_at": obj.created_at.isoformat() if obj.created_at else "",
        "label": obj.label,
        "origin_start": obj.origin_start,
        "horizon": obj.horizon,
        "stride": obj.stride,
        "confidence": obj.confidence,
        "interval_method": obj.interval_method,
        "auto": obj.auto,
        "trend_kind": obj.trend_kind,
        "seasonal_kind": obj.seasonal_kind,
        "locks": obj.locks,
        "reconciliation": obj.reconciliation,
        "result": obj.result,
    }


def list_forecasts(db, tree_id):
    return [_annotate_staleness(db, tree_id, o)
            for o in list_hierarchy_forecasts(db, tree_id)]


def get_forecast(db, forecast_id):
    obj = get_hierarchy_forecast(db, forecast_id)
    return None if obj is None else _annotate_staleness(db, obj.tree_id, obj)


def list_backtests(db, tree_id):
    return [backtest_out(o) for o in list_hierarchy_backtests(db, tree_id)]


def get_backtest_out(db, bt_id):
    obj = get_hierarchy_backtest(db, bt_id)
    return None if obj is None else backtest_out(obj)


# ---------------------------------------------------------------------------
# 层级回测
# ---------------------------------------------------------------------------

def run_hierarchy_backtest_job(req, report) -> Dict[str, Any]:
    report(0.01, "读取树与门店历史快照")
    with session_scope() as db:
        tree = load_tree(db, int(req.tree_id))
        if tree is None:
            raise ModelError(f"层级树 {req.tree_id} 不存在。")
        refs = get_store_refs(db, [
            tree.nodes[nid].series_id for nid in tree.bottom_ids
        ])
        leaf_values = {
            nid: np.asarray(refs[tree.nodes[nid].series_id].values, dtype=float)
            for nid in tree.bottom_ids
        }
        period = tree.period
        shape = HierarchyShape(
            node_ids=tree.ordered_node_ids(),
            levels={nid: n.level for nid, n in tree.nodes.items()},
            names={nid: n.name for nid, n in tree.nodes.items()},
            parent_children=tree.parent_children(),
            descendants_bottom=tree.descendants_bottom(),
            bottom_ids=tree.bottom_ids,
        )

    spec = _model_spec(req)
    report(0.05, "逐原点整树基础预测与调和")
    result = run_hierarchy_backtest(
        leaf_values, shape, period=period,
        origin_start=int(req.origin_start), horizon=int(req.horizon),
        stride=int(req.stride), spec=spec,
        confidence=float(req.confidence), interval_method=req.interval_method,
        progress_cb=lambda p: report(0.05 + 0.9 * p, f"回测原点 {p:.0%}"),
    )
    report(0.97, "保存层级回测结果")
    with session_scope() as db:
        obj = create_hierarchy_backtest(db, int(req.tree_id), {
            "label": req.label or "",
            "origin_start": int(req.origin_start),
            "horizon": int(req.horizon),
            "stride": int(req.stride),
            "confidence": float(req.confidence),
            "interval_method": req.interval_method,
            "auto": bool(req.auto),
            "trend_kind": req.trend_kind,
            "seasonal_kind": req.seasonal_kind,
            "locks": dict(req.locks or {}),
            "reconciliation": "wls",
            "result": result,
        })
        bt_id = obj.id
    return {"created_id": bt_id, "result": {"hierarchy_backtest_id": bt_id}}
