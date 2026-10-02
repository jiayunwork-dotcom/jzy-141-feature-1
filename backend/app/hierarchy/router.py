"""HTTP API for hierarchy management, forecasts and backtests."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..jobs import job_manager
from ..kernels.hw import ModelError, SEASONALS, TRENDS
from .. import storage as series_storage
from . import service, storage as hstorage
from .schemas import (
    HierarchyBacktestRequest,
    HierarchyForecastRequest,
    NodeCreate,
    TreeCreate,
)
from .tree import MAX_DEPTH, HierarchyError, build_snapshot

router = APIRouter(prefix="/api/hierarchy", tags=["hierarchy"])


# ---------------------------------------------------------------- trees/nodes

def _load_series_for(db, tree_id: int, extra_rows=None):
    """Map series_id -> Series for all leaf rows (plus staged rows)."""
    rows = list(hstorage.list_nodes(db, tree_id))
    if extra_rows is not None:
        rows = list(extra_rows)
    series_by_id = {}
    for r in rows:
        if r.series_id is not None and r.series_id not in series_by_id:
            s = series_storage.get_series(db, r.series_id)
            if s is not None:
                series_by_id[s.id] = s
    return series_by_id


def _node_out(n) -> dict:
    return {
        "id": n.id,
        "tree_id": n.tree_id,
        "parent_id": n.parent_id,
        "kind": n.kind,
        "name": n.name,
        "level": n.level,
        "series_id": n.series_id,
    }


def _tree_out(db, t) -> dict:
    nodes = hstorage.list_nodes(db, t.id)
    leaf_count = sum(1 for n in nodes if n.kind == "store")
    return {
        "id": t.id,
        "name": t.name,
        "structure_revision": t.structure_revision,
        "created_at": t.created_at.isoformat() if t.created_at else "",
        "node_count": len(nodes),
        "leaf_count": leaf_count,
        "nodes": [_node_out(n) for n in nodes],
    }


@router.get("/trees")
def list_trees(db: Session = Depends(get_db)):
    return [_tree_out(db, t) for t in hstorage.list_trees(db)]


@router.post("/trees")
def create_tree(payload: TreeCreate, db: Session = Depends(get_db)):
    t = hstorage.create_tree(db, payload.name)
    db.commit()
    db.refresh(t)
    return _tree_out(db, t)


@router.get("/trees/{tree_id}")
def tree_detail(tree_id: int, db: Session = Depends(get_db)):
    t = hstorage.get_tree(db, tree_id)
    if t is None:
        raise HTTPException(404, "层级树不存在")
    out = _tree_out(db, t)
    # The tree may be structurally complete enough to browse but not yet
    # runnable (e.g. a freshly created empty region).  Both flags are shown.
    series_by_id = _load_series_for(db, t.id)
    rows = hstorage.list_nodes(db, t.id)
    try:
        snap = build_snapshot(
            t.id, t.name, t.structure_revision, rows, series_by_id,
            require_complete=False,
        )
        out["valid"] = True
        out["error"] = None
        out["period"] = snap.period
        out["histories"] = {
            str(nid): {
                "dates": node.history_dates,
                "values": node.history_values,
            }
            for nid, node in snap.nodes.items()
        }
    except HierarchyError as exc:
        out["valid"] = False
        out["error"] = str(exc)
        out["period"] = None
        out["histories"] = {}
    try:
        build_snapshot(
            t.id, t.name, t.structure_revision, rows, series_by_id,
            require_complete=True,
        )
        out["ready_for_forecast"] = out["valid"]
        out["forecast_error"] = None
    except (HierarchyError, ModelError) as exc:
        out["ready_for_forecast"] = False
        out["forecast_error"] = str(exc)
    forecasts = hstorage.list_forecasts(db, tree_id)
    out["forecasts"] = [
        {"id": f.id,
         "created_at": f.created_at.isoformat() if f.created_at else "",
         "label": f.label,
         "stale": service.staleness_info(db, tree_id)
                 .get(f.id, {}).get("stale", False)}
        for f in forecasts
    ]
    return out


@router.post("/trees/{tree_id}/nodes")
def add_node(tree_id: int, payload: NodeCreate,
             db: Session = Depends(get_db)):
    t = hstorage.get_tree(db, tree_id)
    if t is None:
        raise HTTPException(404, "层级树不存在")
    parent = hstorage.get_node(db, payload.parent_id)
    if parent is None or parent.tree_id != tree_id:
        raise HTTPException(400, "上级节点不存在或不属于该树。")
    if payload.kind not in ("region", "store"):
        raise HTTPException(400, "kind 只能是 region/store。")
    if parent.level + 1 >= MAX_DEPTH:
        raise HTTPException(
            400, f"最多 {MAX_DEPTH} 层（全网/区域/门店），"
                 f"『{parent.name}』下不能再挂{payload.kind}。"
        )
    if payload.kind == "store":
        if not payload.series_id:
            raise HTTPException(400, "挂接门店必须指定 series_id。")
        s = series_storage.get_series(db, payload.series_id)
        if s is None:
            raise HTTPException(400, f"序列 {payload.series_id} 不存在。")
        if hstorage.find_leaf_by_series(db, tree_id, payload.series_id):
            raise HTTPException(
                400, f"序列『{s.name}』已经挂在这棵树中，不能重复挂接。"
            )
        node_name = payload.name or s.name
    else:
        node_name = payload.name

    # Stage the new node in-memory and run the SAME validator the jobs use,
    # so rejection messages name the mismatching child and the difference.
    existing = hstorage.list_nodes(db, tree_id)

    class _Row:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    fake_id = min(n.id for n in existing) - 1 if existing else -1
    staged = list(existing) + [_Row(
        id=fake_id, tree_id=tree_id, parent_id=payload.parent_id,
        kind=payload.kind, name=node_name,
        level=parent.level + 1, series_id=payload.series_id,
    )]
    series_by_id = _load_series_for(db, tree_id, extra_rows=staged)
    try:
        # Lenient: an existing empty region is fine; but period / start /
        # end mismatch of the new leaf is rejected and names the child.
        build_snapshot(t.id, t.name, t.structure_revision,
                       staged, series_by_id, require_complete=False)
    except HierarchyError as exc:
        raise HTTPException(400, str(exc))

    node = hstorage.add_node(
        db, t, payload.parent_id, payload.kind, node_name,
        series_id=payload.series_id if payload.kind == "store" else None,
    )
    db.commit()
    return _node_out(node)


@router.delete("/trees/{tree_id}/nodes/{node_id}")
def remove_node(tree_id: int, node_id: int, db: Session = Depends(get_db)):
    t = hstorage.get_tree(db, tree_id)
    if t is None:
        raise HTTPException(404, "层级树不存在")
    try:
        hstorage.delete_node(db, t, node_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    db.commit()
    return {"deleted": node_id}


# --------------------------------------------------------------- forecasts

@router.post("/forecasts")
def start_forecast(req: HierarchyForecastRequest):
    if req.trend_kind not in TRENDS:
        raise HTTPException(400, f"trend_kind 必须是 {TRENDS} 之一")
    if req.seasonal_kind not in SEASONALS:
        raise HTTPException(400, f"seasonal_kind 必须是 {SEASONALS} 之一")
    job = job_manager.submit(
        "hierarchy_forecast",
        lambda report: service.run_hierarchy_forecast_job(req, report),
    )
    return job.public()


@router.get("/trees/{tree_id}/forecasts")
def list_forecasts(tree_id: int, db: Session = Depends(get_db)):
    return [
        service.forecast_out(db, f)
        for f in hstorage.list_forecasts(db, tree_id)
    ]


@router.get("/forecasts/{forecast_id}")
def forecast_detail(forecast_id: int, db: Session = Depends(get_db)):
    f = hstorage.get_forecast(db, forecast_id)
    if f is None:
        raise HTTPException(404, "层级预测不存在")
    return service.forecast_out(db, f)


# --------------------------------------------------------------- backtests

@router.post("/backtests")
def start_backtest(req: HierarchyBacktestRequest):
    if req.trend_kind not in TRENDS:
        raise HTTPException(400, f"trend_kind 必须是 {TRENDS} 之一")
    if req.seasonal_kind not in SEASONALS:
        raise HTTPException(400, f"seasonal_kind 必须是 {SEASONALS} 之一")
    job = job_manager.submit(
        "hierarchy_backtest",
        lambda report: service.run_hierarchy_backtest_job(req, report),
    )
    return job.public()


@router.get("/trees/{tree_id}/backtests")
def list_backtests(tree_id: int, db: Session = Depends(get_db)):
    return [
        service.backtest_out(db, b)
        for b in hstorage.list_backtests(db, tree_id)
    ]


@router.get("/backtests/{backtest_id}")
def backtest_detail(backtest_id: int, db: Session = Depends(get_db)):
    b = hstorage.get_backtest(db, backtest_id)
    if b is None:
        raise HTTPException(404, "层级回测不存在")
    return service.backtest_out(db, b)
