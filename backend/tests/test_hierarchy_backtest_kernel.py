"""Hierarchical backtest: metrics, coherence per origin, origin isolation."""
from __future__ import annotations

import numpy as np

from _hierarchy_helpers import make_reference_tree
from app.hierarchy.backtest import run_hierarchy_backtest


def test_backtest_runs_and_reports_by_level():
    tree = make_reference_tree()
    bt = run_hierarchy_backtest(
        tree, origin_start=2 * 52 + 2, horizon=6, stride=20,
        trend_kind="add", seasonal_kind="add",
    )
    levels = bt.by_level
    assert set(levels) == {"network", "region", "store"}
    for lvl in levels.values():
        assert lvl["base_mae"] >= 0
        assert np.isfinite(lvl["base_mase"])
        assert np.isfinite(lvl["reconciled_mase"])
    # Noiseless series: reconciliation cannot hurt the leaves (BU keeps them)
    # and the pooled MAE at every level is tiny.
    for lvl in levels.values():
        assert lvl["reconciled_mae"] < 1.0
    # every node has per-origin rows
    assert len(bt.per_node) == len(tree.nodes)
    for nid, rows in bt.per_node.items():
        assert len(rows) == len(bt.origins)
        assert len(rows[0].base_forecast) == 6
        assert len(rows[0].reconciled_forecast) == 6


def test_backtest_is_coherent_at_every_origin():
    tree = make_reference_tree()
    bt = run_hierarchy_backtest(
        tree, origin_start=2 * 52 + 2, horizon=5, stride=30,
        trend_kind="add", seasonal_kind="add",
    )
    by_id = {nid: str(nid) for nid in tree.nodes}
    for row_idx, origin in enumerate(bt.origins):
        rec = {}
        for nid, rows in bt.per_node.items():
            rec[nid] = np.asarray(rows[row_idx].reconciled_forecast)
        for node in tree.nodes.values():
            if not node.children_ids:
                continue
            total = sum(rec[c] for c in node.children_ids)
            np.testing.assert_allclose(
                rec[node.id], total, rtol=1e-6, atol=1e-9,
                err_msg=f"origin {origin} node {node.name}",
            )


def test_backtest_origin_isolation_injected_extremes():
    tree = make_reference_tree()
    origin = 2 * 52 + 5
    h = 5
    bt1 = run_hierarchy_backtest(
        tree, origin_start=origin, horizon=h, stride=10_000,
        trend_kind="add", seasonal_kind="add",
    )
    # Corrupt the raw leaf histories after the origin; rebuild tree.
    from _hierarchy_helpers import reference_store_values
    vals = [v.copy() for v in reference_store_values()]
    for v in vals:
        v[origin + 1:] += 9e5
    contaminated = make_reference_tree(values=vals)
    bt2 = run_hierarchy_backtest(
        contaminated, origin_start=origin, horizon=h, stride=10_000,
        trend_kind="add", seasonal_kind="add",
    )
    for nid in tree.nodes:
        a = bt1.per_node[nid][0]
        b = bt2.per_node[nid][0]
        np.testing.assert_allclose(
            b.base_forecast, a.base_forecast, atol=1e-8,
            err_msg=f"node {nid} base leak",
        )
        np.testing.assert_allclose(
            b.reconciled_forecast, a.reconciled_forecast, atol=1e-8,
            err_msg=f"node {nid} reconciled leak",
        )
