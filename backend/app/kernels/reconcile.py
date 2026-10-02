"""Hierarchical forecast reconciliation — chosen method and why.

Given base forecasts for every node of a store–region–network tree, reconcile
them so each parent's point forecast equals the sum of its direct children's
on every forecast date (relative tolerance 1e-6, all three levels).

Method chosen: **bottom-up (BU)**
    leaf reconciled forecasts = leaf base forecasts;
    every aggregate reconciled forecast = sum of its descendant leaves.

What this buys
~~~~~~~~~~~~~~
* Exact coherence, by construction, at every level and date.
* When the base forecasts are themselves coherent (e.g. additive HW fitted
  on noiseless summed series) reconciled and base values agree to numerical
  precision, so nothing is "fixed" that was not broken.
* A single-child aggregate reconciles to its (only) child exactly — required
  for region 乙 in the reference scenario.
* **Path-local updates**: if one leaf is refit, only that leaf's stored base
  column and the columns of its ancestors change.  A "local recompute"
  therefore produces bit-for-bit (tolerance 1e-9) identical reconciled values
  to a full-tree rerun, because unchanged leaves reuse their stored base
  values and BU never reallocates any mass.  This is the property the partial
  vs. full rerun test pins down.

What was deliberately given up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Top-down splits: they force leaf forecasts to follow the network view,
  discarding store-level information and destroying single-series numbers.
* Weighted least squares / MinT reconciliation (Hyndman et al.): it pools
  forecast-error covariance across all nodes, which is statistically more
  efficient when base forecasts disagree, but every reconciled node is a
  linear combination of *all* base forecasts — touching one leaf changes the
  whole tree, so the two recomputation paths could only agree by redoing the
  global solve anyway.  Choosing BU trades that potential accuracy gain for
  exact local/full recompute equivalence and auditable arithmetic.

Prediction intervals
~~~~~~~~~~~~~~~~~~~~
Reconciled point forecasts are simple sums, but store errors are correlated
and the hierarchy layer does not estimate cross-series covariances (that is
the MinT information we declined to use).  Reconciled intervals therefore use
the **comonotonic conservative sum**: parent half-width = sum of child
half-widths.  It is an exact upper bound for any correlation structure (the
union bound), and because each child's half-width is monotone non-decreasing
in horizon, a sum of them is too — so reconciled interval widths remain
monotone in h.  Leaf bands are the single-series bands unchanged; aggregate
base bands come from the aggregate's own HW fit.  The two band sets need not
agree; the stored reconciled band is the conservative sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..kernels.hw import (
    HWParams,
    HWState,
    ModelError,
    fit_hw,
    forecast,
)
from ..kernels import selection
from ..hierarchy.tree import NodeSnapshot, TreeSnapshot


@dataclass
class NodeBaseForecast:
    """Base (unreconciled) forecast for one node."""

    node_id: int
    series_id: Optional[int]
    trend_kind: str
    seasonal_kind: str
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sse: float
    aic: float
    residual_std: float
    # For store leaves this is the Fit row id used (None for freshly fitted
    # transient runs, e.g. inside backtests).
    fit_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "series_id": self.series_id,
            "trend_kind": self.trend_kind,
            "seasonal_kind": self.seasonal_kind,
            "point": [float(v) for v in self.point],
            "lower": [float(v) for v in self.lower],
            "upper": [float(v) for v in self.upper],
            "sse": float(self.sse),
            "aic": float(self.aic),
            "residual_std": float(self.residual_std),
            "fit_id": self.fit_id,
        }


def state_from_fit(fit) -> HWState:
    """Rebuild a live HWState (with shocks/params attached) from a Fit row."""
    fs = fit.final_state
    state = HWState(
        level=float(fs["level"]),
        trend=None if fs.get("trend") is None else float(fs["trend"]),
        season=np.asarray(fs["season"], dtype=float),
        trend_kind=fs["trend_kind"],
        seasonal_kind=fs["seasonal_kind"],
        phi=float(fs.get("phi", 1.0)),
    )
    residuals = np.asarray(fit.residuals, dtype=float)
    fitted = np.asarray(fit.fitted, dtype=float)
    if fs["seasonal_kind"] == "mul":
        shocks = residuals / np.maximum(np.abs(fitted), 1e-12)
    else:
        shocks = residuals
    object.__setattr__(state, "_shocks", shocks)
    p = fit.params or {}
    object.__setattr__(state, "_alpha", float(p.get("alpha", 0.3)))
    object.__setattr__(state, "_beta", float(p.get("beta", 0.0)))
    object.__setattr__(state, "_gamma", float(p.get("gamma", 0.3)))
    return state


def base_forecast_from_fit(
    node: NodeSnapshot, fit, horizon: int, confidence: float,
    interval_method: str,
) -> NodeBaseForecast:
    """Extend a store leaf's base forecast from a pinned persisted Fit.

    Point forecasts come straight from the fitted final state — the exact
    numbers the single-series page produced for that fit (and re-running the
    recursion deterministically gives the same values).  Bands are always
    regenerated for the requested confidence/method, so switching the level
    on a hierarchy run cannot silently reuse bands made at another level.
    """
    fr = forecast(
        state_from_fit(fit), horizon,
        np.asarray(fit.residuals, dtype=float),
        level=confidence, method=interval_method,
    )
    residual_std = float((fit.forecast or {}).get("residual_std", fr.residual_std))
    return NodeBaseForecast(
        node_id=node.id,
        series_id=node.series_id,
        trend_kind=fit.trend_kind,
        seasonal_kind=fit.seasonal_kind,
        point=np.asarray(fr.point, dtype=float),
        lower=np.asarray(fr.lower, dtype=float),
        upper=np.asarray(fr.upper, dtype=float),
        sse=float(fit.sse), aic=float(fit.aic),
        residual_std=residual_std,
        fit_id=fit.id,
    )


def fit_base_forecast(
    node: NodeSnapshot,
    values: np.ndarray,
    period: int,
    trend_kind: str,
    seasonal_kind: str,
    horizon: int,
    confidence: float,
    interval_method: str,
    locks: Optional[dict] = None,
) -> NodeBaseForecast:
    """Fit an aggregate (or any node without a pinned fit) from scratch."""
    y = np.asarray(values, dtype=float)
    fr = selection.fit_one(
        y, trend_kind, seasonal_kind, period, locks=locks or {}
    )
    fc = forecast(
        fr.final_state, horizon, fr.residuals,
        level=confidence, method=interval_method,
    )
    return NodeBaseForecast(
        node_id=node.id,
        series_id=node.series_id,
        trend_kind=trend_kind,
        seasonal_kind=seasonal_kind,
        point=np.asarray(fc.point, dtype=float),
        lower=np.asarray(fc.lower, dtype=float),
        upper=np.asarray(fc.upper, dtype=float),
        sse=float(fr.sse),
        aic=float(fr.aic),
        residual_std=float(fc.residual_std),
    )


@dataclass
class ReconciledNode:
    node_id: int
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "point": [float(v) for v in self.point],
            "lower": [float(v) for v in self.lower],
            "upper": [float(v) for v in self.upper],
        }


def reconcile(
    tree: TreeSnapshot,
    base: Dict[int, NodeBaseForecast],
) -> Dict[int, ReconciledNode]:
    """Bottom-up reconciliation of points and conservative interval sums."""
    out: Dict[int, ReconciledNode] = {}
    h = _check_same_horizon(base)

    def visit(nid: int) -> None:
        node = tree.node(nid)
        if node.kind == "store":
            b = base[nid]
            out[nid] = ReconciledNode(
                node_id=nid, point=b.point.copy(),
                lower=b.lower.copy(), upper=b.upper.copy(),
            )
            return
        kids = node.children_ids
        for c in kids:
            if c not in out:
                visit(c)
        point = np.zeros(h, dtype=float)
        lower = np.zeros(h, dtype=float)
        upper = np.zeros(h, dtype=float)
        for c in kids:
            r = out[c]
            point = point + r.point
            # Conservative (comonotonic) band sums.
            lower = lower + r.lower
            upper = upper + r.upper
        out[nid] = ReconciledNode(
            node_id=nid, point=point, lower=lower, upper=upper,
        )

    visit(tree.root_id)
    return out


def _check_same_horizon(base: Dict[int, NodeBaseForecast]) -> int:
    horizons = {b.point.size for b in base.values()}
    if len(horizons) != 1:
        raise ModelError("各节点基础预测步长不一致，无法调和。")
    return horizons.pop()


def check_coherence(
    tree: TreeSnapshot,
    rec: Dict[int, ReconciledNode],
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> List[str]:
    """Return human-readable violations of parent == sum(children); [] if OK."""
    problems: List[str] = []
    for node in tree.nodes.values():
        if not node.children_ids:
            continue
        parent = rec[node.id].point
        total = np.zeros_like(parent)
        for c in node.children_ids:
            total = total + rec[c].point
        if not np.allclose(parent, total, rtol=rtol, atol=atol):
            bad = int(np.argmax(np.abs(parent - total)))
            problems.append(
                f"节点『{node.name}』在第 {bad + 1} 步不满足上下一致："
                f"{parent[bad]:.6g} != 子节点之和 {total[bad]:.6g}"
            )
    return problems


def run_tree_base(
    tree: TreeSnapshot,
    horizon: int,
    confidence: float,
    interval_method: str,
    trend_kind: str,
    seasonal_kind: str,
    leaf_fits: Optional[Dict[int, object]] = None,
    progress_cb=None,
    node_failure: str = "raise",
):
    """Compute base forecasts for every node.

    Store leaves use their pinned persisted Fit rows (``leaf_fits`` maps
    series_id -> Fit); aggregates are fit on the summed history with the
    requested (trend, seasonal) combination.

    ``node_failure="collect"`` gathers failures instead of raising, returning
    ``(base, failures)`` — used so a job can report every failing node and
    never persist a half-reconciled result.
    """
    base: Dict[int, NodeBaseForecast] = {}
    failures: List[str] = []
    leaves = tree.leaves()
    aggregates = [n for n in tree.nodes.values() if n.kind != "store"]
    total = len(tree.nodes)
    done = 0

    for node in leaves:
        try:
            fit = (leaf_fits or {}).get(node.series_id)
            if fit is None:
                raise ModelError(
                    f"门店节点『{node.name}』引用的拟合结果不存在。"
                )
            base[node.id] = base_forecast_from_fit(
                node, fit, horizon, confidence, interval_method
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"门店节点『{node.name}』拟合失败：{exc}")
        done += 1
        if progress_cb:
            progress_cb(done / total, f"门店基础预测 {done}/{len(leaves)}")

    for node in aggregates:
        try:
            base[node.id] = fit_base_forecast(
                node,
                node.history_values,
                tree.period,
                trend_kind,
                seasonal_kind,
                horizon,
                confidence,
                interval_method,
            )
        except Exception as exc:  # noqa: BLE001
            kind = "全网" if node.kind == "network" else "区域"
            failures.append(f"{kind}节点『{node.name}』拟合失败：{exc}")
        done += 1
        if progress_cb:
            progress_cb(done / total, f"聚合节点基础预测 {done}")

    if failures and node_failure == "raise":
        raise ModelError("；".join(failures))
    if node_failure == "collect":
        return base, failures
    return base


def reconcile_payload(
    tree: TreeSnapshot,
    base: Dict[int, NodeBaseForecast],
    future_dates: List[str],
) -> dict:
    """Run reconciliation and assemble the JSON stored per forecast."""
    rec = reconcile(tree, base)
    problems = check_coherence(tree, rec)
    if problems:
        raise ModelError("调和结果未通过一致性校验：" + "；".join(problems))
    nodes_out = {}
    for nid, node in tree.nodes.items():
        b = base[nid]
        r = rec[nid]
        nodes_out[str(nid)] = {
            "node_id": nid,
            "name": node.name,
            "kind": node.kind,
            "level": node.level,
            "parent_id": node.parent_id,
            "series_id": node.series_id,
            "history_dates": list(node.history_dates),
            "history_values": [float(v) for v in node.history_values],
            "base": b.to_dict(),
            "reconciled": r.to_dict(),
            "diff_point": [
                float(r.point[j] - b.point[j])
                for j in range(b.point.size)
            ],
        }
    return {
        "method": "bottom_up",
        "future_dates": list(future_dates),
        "nodes": nodes_out,
    }
