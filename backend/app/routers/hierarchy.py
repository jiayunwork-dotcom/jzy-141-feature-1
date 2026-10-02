"""层级（门店—区域—全网）路由：树管理、挂接预检、层级预测/回测任务。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..jobs import job_manager
from ..kernels.hw import SEASONALS, TRENDS
from ..hierarchy_schemas import (
    HierarchyBacktestRequest,
    HierarchyForecastRequest,
    HierarchyTreeCreate,
)
from ..hierarchy_service import (
    HierarchyError,
    build_tree,
    get_backtest_out,
    get_forecast,
    list_backtests,
    list_forecasts,
    list_trees,
    run_hierarchy_backtest_job,
    run_hierarchy_forecast_job,
    tree_detail,
    validate_assignment,
)
from ..schemas import JobOut

router = APIRouter(prefix="/api/hierarchy", tags=["hierarchy"])


def _hierarchy_error(exc: HierarchyError) -> HTTPException:
    # detail 可以是结构化 dict（挂接拒绝原因列表），FastAPI 会原样 JSON 化。
    detail = exc.args[0] if exc.args else str(exc)
    return HTTPException(status_code=400, detail=detail)


def _validate_model(req) -> None:
    if not req.auto:
        if req.trend_kind not in TRENDS:
            raise HTTPException(400, f"trend_kind 必须是 {TRENDS} 之一")
        if req.seasonal_kind not in SEASONALS:
            raise HTTPException(400, f"seasonal_kind 必须是 {SEASONALS} 之一")
    if req.interval_method not in ("analytic", "simulate"):
        raise HTTPException(400, "interval_method 只能是 analytic/simulate")


@router.get("/trees")
def trees(db: Session = Depends(get_db)):
    out = []
    for t in list_trees(db):
        out.append({
            "id": t.id, "name": t.name, "period": t.period,
            "created_at": t.created_at.isoformat() if t.created_at else "",
            "n_nodes": len(t.nodes),
        })
    return out


@router.post("/trees/validate")
def validate_tree(req: HierarchyTreeCreate, db: Session = Depends(get_db)):
    return validate_assignment(req, db)


@router.post("/trees")
def create(req: HierarchyTreeCreate, db: Session = Depends(get_db)):
    try:
        out = build_tree(req, db)
        db.commit()
    except HierarchyError as exc:
        db.rollback()
        raise _hierarchy_error(exc)
    return out


@router.get("/trees/{tree_id}")
def detail(tree_id: int, db: Session = Depends(get_db)):
    try:
        return tree_detail(db, tree_id)
    except HierarchyError as exc:
        raise _hierarchy_error(exc)


@router.post("/forecasts", response_model=JobOut)
def forecast(req: HierarchyForecastRequest):
    _validate_model(req)
    if req.horizon < 1:
        raise HTTPException(400, "horizon 必须 >= 1")
    if not 0.5 <= req.confidence < 1:
        raise HTTPException(400, "confidence 必须在 [0.5, 1)")
    if req.mode not in ("full", "auto"):
        raise HTTPException(400, "mode 只能是 full/auto")
    job = job_manager.submit(
        "hierarchy_forecast",
        lambda report: run_hierarchy_forecast_job(req, report),
    )
    return job.public()


@router.get("/forecasts/by-tree/{tree_id}")
def forecasts_for_tree(tree_id: int, db: Session = Depends(get_db)):
    return list_forecasts(db, tree_id)


@router.get("/forecasts/{forecast_id}")
def forecast_detail(forecast_id: int, db: Session = Depends(get_db)):
    out = get_forecast(db, forecast_id)
    if out is None:
        raise HTTPException(404, "层级预测结果不存在")
    return out


@router.post("/backtests", response_model=JobOut)
def backtest(req: HierarchyBacktestRequest):
    _validate_model(req)
    if req.horizon < 1 or req.stride < 1:
        raise HTTPException(400, "horizon/stride 必须 >= 1")
    job = job_manager.submit(
        "hierarchy_backtest",
        lambda report: run_hierarchy_backtest_job(req, report),
    )
    return job.public()


@router.get("/backtests/by-tree/{tree_id}")
def backtests_for_tree(tree_id: int, db: Session = Depends(get_db)):
    return list_backtests(db, tree_id)


@router.get("/backtests/{backtest_id}")
def backtest_detail(backtest_id: int, db: Session = Depends(get_db)):
    out = get_backtest_out(db, backtest_id)
    if out is None:
        raise HTTPException(404, "层级回测结果不存在")
    return out
