"""Reconciliation kernel tests: coherence, reference numbers, intervals."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from app.kernels.reconcile import (
    NodeBaseForecast,
    check_coherence,
    reconcile,
    reconcile_payload,
)
from _hierarchy_helpers import (
    base_fit_all,
    make_reference_tree,
    reference_store_values,
    weekly_dates,
)


def _ids(tree):
    ids = {}
    for n in tree.nodes.values():
        ids[n.name] = n.id
    return ids


def test_reference_tree_aggregation_is_weekly_sum():
    tree = make_reference_tree()
    vals = [np.asarray(v, dtype=float) for v in reference_store_values()]
    nodes = {n.name: n for n in tree.nodes.values()}
    np.testing.assert_allclose(
        nodes["区域甲"].history_values, vals[0] + vals[1], atol=1e-12
    )
    np.testing.assert_allclose(
        nodes["区域乙"].history_values, vals[2], atol=1e-12
    )
    np.testing.assert_allclose(
        nodes["全网"].history_values,
        vals[0] + vals[1] + vals[2], atol=1e-12,
    )


def test_reference_numbers_and_coherence():
    tree = make_reference_tree()
    base = base_fit_all(tree, horizon=12)
    rec = reconcile(tree, base)
    ids = _ids(tree)

    # Coherent at every level/date with the required relative tolerance.
    assert check_coherence(tree, rec, rtol=1e-6, atol=0.0) == []

    # Network first-step point forecast is 428.4.
    assert rec[ids["全网"]].point[0] == pytest.approx(428.4, abs=1e-9)

    # Base forecasts are themselves coherent on this noiseless example:
    # |reconciled - base| <= 0.01 at every node/date.
    for nid, node in tree.nodes.items():
        diff = np.abs(rec[nid].point - base[nid].point)
        assert np.max(diff) <= 0.01, (node.name, np.max(diff))

    # Region 乙 has one child: reconciled == leaf 3 date by date.
    np.testing.assert_allclose(
        rec[ids["区域乙"]].point, rec[ids["门店三"]].point, atol=1e-12
    )
    # Region 甲 reconciled == store1 + store2.
    np.testing.assert_allclose(
        rec[ids["区域甲"]].point,
        rec[ids["门店一"]].point + rec[ids["门店二"]].point,
        atol=1e-12,
    )


def test_bottom_up_leaves_unchanged_and_coherent_with_noisy_base():
    """Even when base forecasts disagree, BU keeps leaves and fixes parents."""
    rng = np.random.default_rng(7)
    tree = make_reference_tree()
    h = 12
    base = {}
    for node in tree.nodes.values():
        if node.kind == "store":
            point = 100 + np.cumsum(rng.normal(0, 1, h))
        else:
            point = 200 + rng.normal(0, 5, h)  # deliberately incompatible
        base[node.id] = NodeBaseForecast(
            node_id=node.id, series_id=node.series_id,
            trend_kind="add", seasonal_kind="add",
            point=point, lower=point - 5, upper=point + 5,
            sse=1.0, aic=1.0, residual_std=1.0,
        )
    rec = reconcile(tree, base)
    ids = _ids(tree)
    # leaves are untouched
    np.testing.assert_array_equal(
        rec[ids["门店一"]].point, base[ids["门店一"]].point
    )
    assert check_coherence(tree, rec, rtol=1e-12, atol=1e-12) == []
    # network base was wrong (~200) but reconciled equals leaf sums (~300+)
    assert rec[ids["全网"]].point[0] != base[ids["全网"]].point[0]


def test_reconciled_interval_widths_monotone_and_conservative():
    tree = make_reference_tree()
    base = base_fit_all(tree, horizon=12)
    rec = reconcile(tree, base)
    for nid, node in tree.nodes.items():
        width = rec[nid].upper - rec[nid].lower
        assert np.all(np.diff(width) >= -1e-9), node.name
    ids = _ids(tree)
    # comonotonic sum: root half-width == sum of region half-widths
    half_root = (rec[ids["全网"]].upper - rec[ids["全网"]].lower) / 2
    half_a = (rec[ids["区域甲"]].upper - rec[ids["区域甲"]].lower) / 2
    half_b = (rec[ids["区域乙"]].upper - rec[ids["区域乙"]].lower) / 2
    np.testing.assert_allclose(half_root, half_a + half_b, atol=1e-9)


def test_payload_contains_diff_and_history():
    tree = make_reference_tree()
    base = base_fit_all(tree, horizon=12)
    payload = reconcile_payload(
        tree, base, weekly_dates(12, date(2022, 1, 3))
    )
    assert payload["method"] == "bottom_up"
    assert len(payload["future_dates"]) == 12
    node = payload["nodes"][str(tree.root_id)]
    assert len(node["diff_point"]) == 12
    assert len(node["history_values"]) == 164
    assert node["diff_point"][0] == pytest.approx(0.0, abs=0.01)
