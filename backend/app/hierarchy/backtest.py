"""Rolling-origin backtest for a whole hierarchy.

At every origin k the *entire tree* is rebuilt using only data through week k:

1. leaf prefixes y_leaf[:k] are taken as given;
2. each aggregate's prefix history is re-summed from its descendant leaf
   prefixes (no future data);
3. every node is freshly fit (single-series kernels, same combination for a
   run) and produces h base forecasts;
4. bottom-up reconciliation runs exactly as for a live forecast.

Metrics are MAE / MASE per node (and pooled by level), both BEFORE and AFTER
reconciliation, so managers can see which level reconciliation helps and
which it hurts.  MASE scale per node/origin is the seasonal-naive scale of
that node's *training prefix*, matching the single-series convention.

Origin isolation is proven the same way as for single series: inject extreme
values after the origin and assert that origin's forecasts (base and
reconciled, every node) do not change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..kernels.backtest import _naive_mae_scale
from ..kernels.hw import ModelError, fit_hw, forecast
from ..kernels import optimizer
from ..kernels.reconcile import (
    NodeBaseForecast,
    ReconciledNode,
    reconcile,
)
from .tree import TreeSnapshot


@dataclass
class NodeOriginResult:
    origin: int
    base_forecast: List[float]
    reconciled_forecast: List[float]
    actual: List[float]
    base_mae: float
    reconciled_mae: float
    base_mase: float
    reconciled_mase: float
    scale: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class HierarchyBacktestOutcome:
    period: int
    horizon: int
    origins: List[int]
    origin_dates: List[str]
    # node_id -> per-origin rows
    per_node: Dict[int, List[NodeOriginResult]]
    # level -> pooled stats base/reconciled
    by_level: Dict[str, dict] = field(default_factory=dict)
    node_meta: Dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "horizon": self.horizon,
            "origins": list(self.origins),
            "origin_dates": list(self.origin_dates),
            "per_node": {
                str(nid): [r.to_dict() for r in rows]
                for nid, rows in self.per_node.items()
            },
            "by_level": self.by_level,
            "node_meta": {
                str(nid): meta for nid, meta in self.node_meta.items()
            },
        }


def _fit_forecast(
    values: np.ndarray,
    period: int,
    trend_kind: str,
    seasonal_kind: str,
    horizon: int,
    confidence: float,
    interval_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params, _, _ = optimizer.optimize(
        values, trend_kind, seasonal_kind, period
    )
    fr = fit_hw(values, trend_kind, seasonal_kind, period, params)
    fc = forecast(
        fr.final_state, horizon, fr.residuals,
        level=confidence, method=interval_method,
    )
    return (
        np.asarray(fc.point, dtype=float),
        np.asarray(fc.lower, dtype=float),
        np.asarray(fc.upper, dtype=float),
    )


def run_hierarchy_backtest(
    tree: TreeSnapshot,
    origin_start: int,
    horizon: int,
    stride: int = 1,
    trend_kind: str = "add",
    seasonal_kind: str = "add",
    confidence: float = 0.95,
    interval_method: str = "analytic",
    progress_cb=None,
) -> HierarchyBacktestOutcome:
    m = tree.period
    n = len(tree.root.history_values)
    if n < 2 * m + horizon:
        raise ModelError(
            f"层级回测至少需要 2 个季节 + {horizon} 个留出观测。"
        )
    if origin_start < 2 * m:
        raise ModelError(f"首个原点必须 >= 2 个季节（{2 * m}）。")
    if origin_start + horizon > n:
        raise ModelError("最后一个原点之后没有足够的实际值用于比较。")
    if stride < 1:
        raise ModelError("步长必须 >= 1。")

    origins = list(range(origin_start, n - horizon + 1, stride))
    per_node: Dict[int, List[NodeOriginResult]] = {
        nid: [] for nid in tree.nodes
    }
    # Full-length histories per node (leaves given, aggregates pre-summed).
    full_hist = {
        nid: np.asarray(node.history_values, dtype=float)
        for nid, node in tree.nodes.items()
    }

    for step, k in enumerate(origins):
        base: Dict[int, NodeBaseForecast] = {}
        for nid, node in tree.nodes.items():
            train = full_hist[nid][:k]
            point, lower, upper = _fit_forecast(
                train, m, trend_kind, seasonal_kind, horizon,
                confidence, interval_method,
            )
            base[nid] = NodeBaseForecast(
                node_id=nid,
                series_id=node.series_id,
                trend_kind=trend_kind,
                seasonal_kind=seasonal_kind,
                point=point, lower=lower, upper=upper,
                sse=0.0, aic=0.0, residual_std=0.0,
            )
        rec: Dict[int, ReconciledNode] = reconcile(tree, base)

        for nid, node in tree.nodes.items():
            actual = full_hist[nid][k:k + horizon]
            q = _naive_mae_scale(full_hist[nid][:k], m)
            q_eff = q if (q and np.isfinite(q) and q > 0) else np.nan

            def _mae(pred: np.ndarray) -> float:
                return float(np.mean(np.abs(pred - actual)))

            def _mase(pred: np.ndarray) -> float:
                return (
                    float(np.mean(np.abs(pred - actual)) / q_eff)
                    if np.isfinite(q_eff) else float("nan")
                )

            per_node[nid].append(NodeOriginResult(
                origin=int(k),
                base_forecast=[float(v) for v in base[nid].point],
                reconciled_forecast=[float(v) for v in rec[nid].point],
                actual=[float(v) for v in actual],
                base_mae=_mae(base[nid].point),
                reconciled_mae=_mae(rec[nid].point),
                base_mase=_mase(base[nid].point),
                reconciled_mase=_mase(rec[nid].point),
                scale=float(q) if np.isfinite(q) else float("nan"),
            ))
        if progress_cb:
            progress_cb((step + 1) / len(origins), f"回测原点 {step + 1}/{len(origins)}")

    by_level = _pool_by_level(tree, per_node)
    node_meta = {
        nid: {
            "name": node.name,
            "kind": node.kind,
            "level": node.level,
            "series_id": node.series_id,
        }
        for nid, node in tree.nodes.items()
    }
    dates = tree.root.history_dates
    return HierarchyBacktestOutcome(
        period=m,
        horizon=horizon,
        origins=origins,
        origin_dates=[dates[k] for k in origins],
        per_node=per_node,
        by_level=by_level,
        node_meta=node_meta,
    )


def _pool_by_level(
    tree: TreeSnapshot,
    per_node: Dict[int, List[NodeOriginResult]],
) -> Dict[str, dict]:
    levels: Dict[int, List[int]] = {}
    for nid, node in tree.nodes.items():
        levels.setdefault(node.level, []).append(nid)
    out = {}
    names = {0: "network", 1: "region", 2: "store"}
    for level, nids in levels.items():
        base_err, rec_err, scales = [], [], []
        for nid in nids:
            for row in per_node[nid]:
                h = len(row.actual)
                b = np.asarray(row.base_forecast) - np.asarray(row.actual)
                r = np.asarray(row.reconciled_forecast) - np.asarray(row.actual)
                base_err.append(b)
                rec_err.append(r)
                scales.extend([row.scale] * h)
        base_err = np.concatenate(base_err)
        rec_err = np.concatenate(rec_err)
        scales_arr = np.asarray(scales)
        finite = np.isfinite(scales_arr)

        def _mase_block(err: np.ndarray) -> float:
            return (
                float(np.mean(np.abs(err[finite]) / scales_arr[finite]))
                if np.any(finite) else float("nan")
            )

        out[names.get(level, f"level_{level}")] = {
            "level": level,
            "node_count": len(nids),
            "base_mae": float(np.mean(np.abs(base_err))),
            "reconciled_mae": float(np.mean(np.abs(rec_err))),
            "base_mase": _mase_block(base_err),
            "reconciled_mase": _mase_block(rec_err),
        }
    return out
