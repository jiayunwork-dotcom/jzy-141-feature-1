"""真实线程并发：层级任务计算途中，另一计划员提交单序列重拟任务。

通过 JobManager 的线程池运行：层级任务先在单事务内钉死一组 fit 版本，
计算途中另一个线程给一家店插入新拟合；层级结果记录的 fit_refs 必须
整组一致（就是它快照到的版本），而不是新旧混合。
"""
from __future__ import annotations

import threading
import time
from datetime import date, timedelta

import numpy as np

from app.db import session_scope
from app import hierarchy_service as hs
from app import storage
from app.hierarchy_schemas import (
    HierarchyForecastRequest,
    HierarchyTreeCreate,
    RegionSpec,
)
from app.jobs import job_manager
from app.kernels.hw import forecast
from app.kernels.selection import fit_one

N, M = 164, 52
D0 = date(2022, 1, 3)


def _add_fit(db, sid, y, locks=None):
    fr = fit_one(y, "add", "add", M, locks=locks or {})
    fc = forecast(fr.final_state, 12, fr.residuals, level=0.95)
    return storage.create_fit(db, sid, {
        "label": "", "auto": False, "trend_kind": "add",
        "seasonal_kind": "add", "period": M,
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


def test_concurrent_refit_during_hierarchy_job():
    rng = np.random.default_rng(11)
    t = np.arange(N)
    ys = [
        100 + 0.2 * t + np.where(t % M == 48, 50, 0) + rng.normal(0, 1, N),
        150 + 0.3 * t + np.where(t % M == 20, 30, 0) + rng.normal(0, 1, N),
        80 + 0.1 * t + np.where(t % M == 48, 20, 0)
        + np.where(t % M == 20, 10, 0) + rng.normal(0, 1, N),
    ]
    dates = [(D0 + timedelta(weeks=k)).isoformat() for k in range(N)]
    sids = []
    with session_scope() as db:
        for name, y in zip(["并发店1", "并发店2", "并发店3"], ys):
            sid = storage.create_series(db, name, dates, y.tolist(), M).id
            sids.append(sid)
            _add_fit(db, sid, y)
        tree_id = hs.build_tree(HierarchyTreeCreate(
            name="并发树",
            regions=[RegionSpec(name="甲", series_ids=sids[:2]),
                     RegionSpec(name="乙", series_ids=sids[2:])],
        ), db)["tree_id"]
        db.commit()

    with session_scope() as db:
        old_fits = hs.latest_fit_ids_for_series(db, sids)

    barrier = threading.Event()

    def slow_report(p, stage):
        # 进入聚合节点拟合阶段后放行并发重拟线程
        if p >= 0.4:
            barrier.set()
        time.sleep(0.002)

    def hierarchy_task(report):
        req = HierarchyForecastRequest(
            tree_id=tree_id, horizon=12, auto=False,
            trend_kind="add", seasonal_kind="add", mode="full",
        )
        return hs.run_hierarchy_forecast_job(
            req, lambda p, s: (slow_report(p, s), report(p, s))[1]
        )

    def refit_task(report):
        barrier.wait(timeout=10)
        # 与层级任务并发地给第一家店锁参重拟（插入新 fit 行）
        with session_scope() as db:
            _add_fit(db, sids[0], ys[0], locks={"alpha": 0.5})
        return {"created_id": None, "result": {"refitted": True}}

    hjob = job_manager.submit("hierarchy_forecast", hierarchy_task)
    rjob = job_manager.submit("fit", refit_task, series_id=sids[0])

    # 等两个任务都结束
    deadline = time.time() + 60
    while time.time() < deadline:
        if hjob.status in ("done", "error") and rjob.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert hjob.status == "done", hjob.error
    assert rjob.status == "done", rjob.error

    fid = hjob.created_id
    with session_scope() as db:
        f = hs.get_forecast(db, fid)
        latest = hs.latest_fit_ids_for_series(db, sids)

    # 1) 记录的引用必须是一个自洽的旧组：店1 用的是旧 fit，其余也都非空
    assert f["fit_refs"][str(sids[0])] == old_fits[sids[0]]
    assert f["fit_refs"][str(sids[1])] == old_fits[sids[1]]
    assert f["fit_refs"][str(sids[2])] == old_fits[sids[2]]
    # 2) 店1 此刻已有更新拟合，结果应被标为过期
    assert latest[sids[0]] != old_fits[sids[0]]
    assert f["is_stale"] is True
    stale_series = {d["series_id"] for d in f["stale_detail"]}
    assert sids[0] in stale_series
