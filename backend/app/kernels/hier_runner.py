"""层级预测/回测的纯内核：对单个节点出基础预测、对整棵树跑滚动原点。

不触碰数据库；数据由服务层快照后传入，回测原点隔离因此天然成立
（每个原点只拿到 ``y[:k]`` 的切片）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .hierarchy import (
    NodeBase,
    reconcile,
    seasonal_naive_scale,
)
from .hw import HWState, ModelError, forecast
from .selection import auto_select, fit_one


@dataclass
class NodeModelSpec:
    """聚合节点（或回测中每个节点）的模型选型指令。"""

    auto: bool = True
    trend_kind: Optional[str] = None
    seasonal_kind: Optional[str] = None
    locks: Optional[Dict[str, float]] = None


def base_forecast_for_values(
    y: np.ndarray,
    period: int,
    horizon: int,
    spec: NodeModelSpec,
    confidence: float = 0.95,
    interval_method: str = "analytic",
    node_name: str = "",
) -> Dict[str, Any]:
    """对一段历史（门店叶子或加总后的聚合节点）出一份基础预测。

    任何失败都包成 ModelError 并带上节点名，服务层据此精确报告失败节点，
    且不会落库半成品。
    """
    y = np.asarray(y, dtype=float)
    try:
        if spec.auto or not spec.trend_kind or not spec.seasonal_kind:
            sel = auto_select(y, period, locks=spec.locks or {})
            fr = sel.best
        else:
            fr = fit_one(
                y, spec.trend_kind, spec.seasonal_kind, period,
                locks=spec.locks or {},
            )
        fc = forecast(
            fr.final_state, int(horizon), fr.residuals,
            level=float(confidence), method=interval_method,
        )
    except ModelError as exc:
        raise ModelError(f"节点「{node_name}」拟合失败：{exc}") from exc
    except Exception as exc:  # pragma: no cover - 数值安全网
        raise ModelError(f"节点「{node_name}」拟合失败：{exc}") from exc

    resid = np.asarray(fr.residuals, dtype=float)
    rvar = float(np.var(resid, ddof=1)) if resid.size > 1 else 0.0
    return {
        "point": np.asarray(fc.point, dtype=float),
        "lower": np.asarray(fc.lower, dtype=float),
        "upper": np.asarray(fc.upper, dtype=float),
        "residual_var": rvar,
        "trend_kind": fr.trend_kind,
        "seasonal_kind": fr.seasonal_kind,
        "params": {
            "alpha": float(fr.params.alpha),
            "beta": float(fr.params.beta),
            "gamma": float(fr.params.gamma),
            "phi": float(fr.params.phi),
        },
        "sse": float(fr.sse),
        "aic": float(fr.aic),
        "residual_std": float(fc.residual_std),
    }


def base_forecast_from_fit(
    params: Dict[str, float],
    final_state: Dict[str, Any],
    residuals: Sequence[float],
    horizon: int = 12,
    confidence: float = 0.95,
    interval_method: str = "analytic",
) -> Dict[str, Any]:
    """门店叶子：不改它的任何数字，直接用被引用拟合重算 1..horizon 区间。

    final_state/residuals 来自既有的单序列 Fit 行，forecast() 与单序列页
    调的是同一个函数，所以叶子基础预测与单序列结果完全一致。
    """
    season = np.asarray(final_state["season"], dtype=float)
    state = HWState(
        level=float(final_state["level"]),
        trend_kind=final_state["trend_kind"],
        seasonal_kind=final_state["seasonal_kind"],
        phi=float(final_state.get("phi", 1.0)),
        trend=(None if final_state.get("trend") is None
               else float(final_state["trend"])),
        season=season,
    )
    object.__setattr__(state, "_alpha", float(params.get("alpha", 0.3)))
    object.__setattr__(state, "_beta", float(params.get("beta", 0.0)))
    object.__setattr__(state, "_gamma", float(params.get("gamma", 0.3)))
    r = np.asarray(residuals, dtype=float)
    fc = forecast(state, int(horizon), r, level=float(confidence),
                  method=interval_method)
    rvar = float(np.var(r, ddof=1)) if r.size > 1 else 0.0
    return {
        "point": np.asarray(fc.point, dtype=float),
        "lower": np.asarray(fc.lower, dtype=float),
        "upper": np.asarray(fc.upper, dtype=float),
        "residual_var": rvar,
        "residual_std": float(fc.residual_std),
    }


# ---------------------------------------------------------------------------
# 层级滚动原点回测
# ---------------------------------------------------------------------------

@dataclass
class HierarchyShape:
    """回测内核所需的树描述（与数据库解耦）。"""

    node_ids: List[int]
    levels: Dict[int, int]
    names: Dict[int, str]
    parent_children: Dict[int, List[int]]
    descendants_bottom: Dict[int, List[int]]
    bottom_ids: List[int]
    # node_id -> 该节点每个原点的完整历史切片来源函数以外，直接给值函数
    # 由调用方通过 values_at 提供。


def _layer_stats(acc: Dict[int, Dict[str, list]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in sorted(acc):
        a = acc[level]
        def stat(key_err, key_ratio):
            errs = np.asarray(a[key_err], dtype=float)
            ratios = np.asarray(a[key_ratio], dtype=float)
            return {
                "mae": float(np.mean(np.abs(errs))) if errs.size else float("nan"),
                "mase": float(np.mean(ratios[np.isfinite(ratios)]))
                if np.isfinite(ratios).any() else float("nan"),
            }
        out[str(level)] = {
            "base": stat("base_err", "base_ratio"),
            "rec": stat("rec_err", "rec_ratio"),
            "naive": stat("naive_err", "naive_ratio"),
        }
    return out


def run_hierarchy_backtest(
    leaf_values: Dict[int, np.ndarray],
    shape: HierarchyShape,
    period: int,
    origin_start: int,
    horizon: int,
    stride: int,
    spec: NodeModelSpec,
    confidence: float = 0.95,
    interval_method: str = "analytic",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """滚动原点层级回测。

    每个原点 k 只使用各节点 ``history[:k]``：门店用自身序列切片，
    区域/全网用下级切片逐周相加（在循环内重算，未来数据物理上不可达）。
    返回按层汇总的调和前/后（及季节朴素）MAE、MASE，与逐原点明细。
    """
    leaves = shape.bottom_ids
    full = {b: np.asarray(leaf_values[b], dtype=float) for b in leaves}
    n = next(iter(full.values())).size
    for b in leaves:
        if full[b].size != n:
            raise ModelError(
                f"回测要求叶子历史等长：节点 {b} 为 {full[b].size}，基准为 {n}。")

    if origin_start < 2 * period:
        raise ModelError(f"首个原点必须 >= 2 个季节（{2 * period}）。")
    if origin_start + horizon > n:
        raise ModelError("最后一个原点之后没有足够的实际值用于比较。")

    origins = list(range(origin_start, n - horizon + 1, stride))
    origin_rows: List[Dict[str, Any]] = []
    layer_acc: Dict[int, Dict[str, list]] = {
        lv: {"base_err": [], "rec_err": [], "naive_err": [],
             "base_ratio": [], "rec_ratio": [], "naive_ratio": []}
        for lv in set(shape.levels.values())
    }

    def history_at(node_id: int, k: int) -> np.ndarray:
        leaves_under = shape.descendants_bottom[node_id]
        total = np.zeros(k, dtype=float)
        for b in leaves_under:
            total = total + full[b][:k]
        return total

    def actual_at(node_id: int, k: int) -> np.ndarray:
        leaves_under = shape.descendants_bottom[node_id]
        total = np.zeros(horizon, dtype=float)
        for b in leaves_under:
            total = total + full[b][k:k + horizon]
        return total

    for step, k in enumerate(origins):
        bases: List[NodeBase] = []
        actuals: Dict[int, np.ndarray] = {}
        scales: Dict[int, float] = {}
        naives: Dict[int, np.ndarray] = {}
        for nid in shape.node_ids:
            train = history_at(nid, k)
            actual = actual_at(nid, k)
            actuals[nid] = actual
            b = base_forecast_for_values(
                train, period, horizon, spec,
                confidence=confidence, interval_method=interval_method,
                node_name=shape.names[nid],
            )
            bases.append(NodeBase(
                node_id=nid, level=shape.levels[nid], name=shape.names[nid],
                point=b["point"], lower=b["lower"], upper=b["upper"],
                residual_var=b["residual_var"],
                trend_kind=b["trend_kind"], seasonal_kind=b["seasonal_kind"],
            ))
            q = seasonal_naive_scale(train, period)
            scales[nid] = q
            naives[nid] = np.array([
                train[k + j - period] if k + j - period >= 0 else np.nan
                for j in range(horizon)
            ])

        rec = reconcile(
            bases, leaves, shape.descendants_bottom, apply_intervals=False
        )

        node_rows: List[Dict[str, Any]] = []
        for nid in shape.node_ids:
            i = rec.row_index[nid]
            q = scales[nid]
            actual = actuals[nid]
            base_err = rec.base_point[i] - actual
            rec_err = rec.rec_point[i] - actual
            naive_err = naives[nid] - actual
            lv = shape.levels[nid]
            acc = layer_acc[lv]
            for e in base_err:
                acc["base_err"].append(float(e))
                acc["base_ratio"].append(
                    float(abs(e) / q) if q and np.isfinite(q) else float("nan"))
            for e in rec_err:
                acc["rec_err"].append(float(e))
                acc["rec_ratio"].append(
                    float(abs(e) / q) if q and np.isfinite(q) else float("nan"))
            for e in naive_err:
                acc["naive_err"].append(float(e))
                acc["naive_ratio"].append(
                    float(abs(e) / q) if q and np.isfinite(q) else float("nan"))
            node_rows.append({
                "node_id": nid,
                "level": lv,
                "name": shape.names[nid],
                "base_point": [float(v) for v in rec.base_point[i]],
                "rec_point": [float(v) for v in rec.rec_point[i]],
                "actual": [float(v) for v in actual],
                "naive_point": [float(v) for v in naives[nid]],
                "scale_q": float(q) if np.isfinite(q) else None,
                "base_mae": float(np.mean(np.abs(base_err))),
                "rec_mae": float(np.mean(np.abs(rec_err))),
                "naive_mae": float(np.mean(np.abs(naive_err))),
            })
        origin_rows.append({"origin": int(k), "nodes": node_rows})
        if progress_cb is not None:
            progress_cb((step + 1) / len(origins))

    level_names = {"0": "门店", "1": "区域", "2": "全网"}
    layers = _layer_stats(layer_acc)
    for key, label in level_names.items():
        if key in layers:
            layers[key]["label"] = label
    return {
        "period": period,
        "horizon": horizon,
        "origin_start": int(origin_start),
        "stride": int(stride),
        "origins": origin_rows,
        "layers": layers,
    }
