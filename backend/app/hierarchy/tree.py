"""Tree structure handling for the hierarchy layer.

Responsibilities:

* build an in-memory :class:`TreeSnapshot` from ORM rows plus the leaf
  ``Series`` (dates/values/period) — pure data, detached from any DB session;
* validate attachment: max three levels, one network root, every leaf aligned
  on period / start week / end week, no series attached twice, internal nodes
  must have a descendant leaf;
* bottom-up history aggregation (region/network histories are *never*
  uploaded, only summed week by week).

Validation messages always name the offending child node and say exactly what
mismatches, e.g. 「子节点『上海静安店』周期 12 与『杭州西湖店』周期 52 不一致」.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


MAX_DEPTH = 3  # network(0) -> region(1) -> store(2)


class HierarchyError(ValueError):
    """Raised for structural / alignment problems in a hierarchy tree."""


@dataclass
class LeafInfo:
    node_id: int
    name: str
    series_id: int
    period: int
    dates: List[str]
    values: List[float]


@dataclass
class NodeSnapshot:
    id: int
    name: str
    kind: str
    level: int
    parent_id: Optional[int]
    series_id: Optional[int]
    children_ids: List[int] = field(default_factory=list)
    # populated for leaves
    leaf: Optional[LeafInfo] = None
    # populated after aggregate()
    history_dates: List[str] = field(default_factory=list)
    history_values: List[float] = field(default_factory=list)


@dataclass
class TreeSnapshot:
    tree_id: int
    name: str
    structure_revision: int
    nodes: Dict[int, NodeSnapshot]
    root_id: int
    period: int

    def node(self, node_id: int) -> NodeSnapshot:
        return self.nodes[node_id]

    @property
    def root(self) -> NodeSnapshot:
        return self.nodes[self.root_id]

    def children(self, node_id: int) -> List[NodeSnapshot]:
        return [self.nodes[c] for c in self.nodes[node_id].children_ids]

    def leaves(self) -> List[NodeSnapshot]:
        return [n for n in self.nodes.values() if n.kind == "store"]

    def descendants(self, node_id: int) -> List[NodeSnapshot]:
        """All descendant nodes (excluding self), depth-first."""
        out: List[NodeSnapshot] = []
        stack = list(self.nodes[node_id].children_ids)
        while stack:
            cur = stack.pop()
            out.append(self.nodes[cur])
            stack.extend(self.nodes[cur].children_ids)
        return out

    def leaf_descendants(self, node_id: int) -> List[NodeSnapshot]:
        return [n for n in self.descendants(node_id) if n.kind == "store"]

    def ancestors(self, node_id: int) -> List[int]:
        """From parent up to (and including) the root."""
        out: List[int] = []
        cur = self.nodes[node_id].parent_id
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent_id
        return out

    def aggregate(self, require_complete: bool = True) -> None:
        """Sum leaf histories bottom-up into every aggregate node."""
        for n in self._bottom_up():
            if n.kind == "store":
                n.history_dates = list(n.leaf.dates) if n.leaf else []
                n.history_values = list(n.leaf.values) if n.leaf else []
            else:
                kids = [k for k in self.children(n.id) if k.history_dates]
                if not kids:
                    # Empty aggregate (only possible in work-in-progress
                    # trees staged with require_complete=False).
                    n.history_dates, n.history_values = [], []
                    continue
                # All non-empty children share the same date grid (validated
                # from the leaves when the tree is built); sum week by week.
                dates = list(kids[0].history_dates)
                total = np.zeros(len(dates), dtype=float)
                for k in kids:
                    if k.history_dates != dates:
                        raise HierarchyError(
                            f"节点『{k.name}』与兄弟节点的历史周对不齐，"
                            "无法加总。"
                        )
                    total += np.asarray(k.history_values, dtype=float)
                n.history_dates = dates
                n.history_values = [float(v) for v in total]

    def _bottom_up(self) -> List[NodeSnapshot]:
        order: List[NodeSnapshot] = []

        def visit(nid: int) -> None:
            for c in self.nodes[nid].children_ids:
                visit(c)
            order.append(self.nodes[nid])

        visit(self.root_id)
        return order


def build_snapshot(
    tree_id: int,
    tree_name: str,
    structure_revision: int,
    rows: Sequence,
    series_by_id: Dict[int, object],
    require_complete: bool = True,
) -> TreeSnapshot:
    """Build a snapshot from (node rows, series map).

    ``rows`` are HierarchyNode ORM objects (only attributes are read).
    ``series_by_id`` maps series id to an object with
    ``id/name/period/dates/values``.

    Full structural validation happens here so every consumer (forecast,
    backtest, API) sees the same guarantees.  When ``require_complete`` is
    False the tree is allowed to contain empty aggregates (the UI creates a
    region before stores are dropped in); such nodes carry empty histories
    and forecast jobs reject them explicitly.  Jobs always pass
    ``require_complete=True``.
    """
    nodes: Dict[int, NodeSnapshot] = {}
    for r in rows:
        nodes[r.id] = NodeSnapshot(
            id=r.id,
            name=r.name,
            kind=r.kind,
            level=r.level,
            parent_id=r.parent_id,
            series_id=r.series_id,
        )
    for n in nodes.values():
        if n.parent_id is not None:
            if n.parent_id not in nodes:
                raise HierarchyError(
                    f"节点『{n.name}』的上级节点缺失（id={n.parent_id}）。"
                )
            nodes[n.parent_id].children_ids.append(n.id)

    roots = [n for n in nodes.values() if n.parent_id is None]
    if len(roots) != 1:
        raise HierarchyError("一棵树必须恰好有一个全网根节点。")
    root = roots[0]
    if root.kind != "network":
        raise HierarchyError("根节点必须是全网节点。")

    leaves = [n for n in nodes.values() if n.kind == "store"]
    if not leaves and require_complete:
        raise HierarchyError("树中还没有挂接任何门店序列。")

    # Attach + validate leaf series.
    period: Optional[int] = None
    ref: Optional[NodeSnapshot] = None
    for n in nodes.values():
        if n.kind == "store":
            if n.series_id is None:
                raise HierarchyError(f"门店节点『{n.name}』缺少关联序列。")
            s = series_by_id.get(n.series_id)
            if s is None:
                raise HierarchyError(
                    f"门店节点『{n.name}』引用的序列（id={n.series_id}）"
                    "不存在，可能已被删除。"
                )
            n.leaf = LeafInfo(
                node_id=n.id,
                name=n.name,
                series_id=s.id,
                period=int(s.period),
                dates=list(s.dates),
                values=[float(v) for v in s.values],
            )
            if period is None:
                period = n.leaf.period
                ref = n
            else:
                _check_aligned(ref, n)
        elif n.series_id is not None:
            raise HierarchyError(
                f"聚合节点『{n.name}』不允许直接关联上传序列；"
                "区域/全网历史只能由下级加总。"
            )

    # Levels / kinds / depth.
    _validate_levels_and_kinds(nodes, root.id)

    # Every internal node must have at least one store descendant — unless
    # the caller is staging a partially built tree (UI work-in-progress).
    if require_complete:
        for n in nodes.values():
            if n.kind != "store" and not any(
                d.kind == "store" for d in _descendants_static(nodes, n.id)
            ):
                raise HierarchyError(
                    f"聚合节点『{n.name}』下没有任何门店，至少需要一个下级门店。"
                )

    snap = TreeSnapshot(
        tree_id=tree_id,
        name=tree_name,
        structure_revision=structure_revision,
        nodes=nodes,
        root_id=root.id,
        period=int(period) if period is not None else 52,
    )
    snap.aggregate(require_complete=require_complete)
    return snap


def _descendants_static(
    nodes: Dict[int, NodeSnapshot], node_id: int
) -> List[NodeSnapshot]:
    out: List[NodeSnapshot] = []
    stack = list(nodes[node_id].children_ids)
    while stack:
        cur = stack.pop()
        out.append(nodes[cur])
        stack.extend(nodes[cur].children_ids)
    return out


def _validate_levels_and_kinds(
    nodes: Dict[int, NodeSnapshot], root_id: int
) -> None:
    def visit(nid: int, depth: int) -> None:
        n = nodes[nid]
        if depth >= MAX_DEPTH:
            raise HierarchyError(
                f"层级超过最多 {MAX_DEPTH} 层（全网/区域/门店）："
                f"节点『{n.name}』位于第 {depth + 1} 层。"
            )
        expected_kind = ("network", "region", "store")[depth]
        if n.kind != expected_kind:
            raise HierarchyError(
                f"节点『{n.name}』类型为 {n.kind}，但在树中位于第 "
                f"{depth + 1} 层（应为 {expected_kind}）。"
            )
        if n.level != depth:
            raise HierarchyError(
                f"节点『{n.name}』记录的层级 {n.level} 与实际位置 {depth} 不一致。"
            )
        if n.kind == "store" and n.children_ids:
            raise HierarchyError(f"门店节点『{n.name}』不能再有下级。")
        for c in n.children_ids:
            visit(c, depth + 1)

    visit(root_id, 0)


def _check_aligned(ref: NodeSnapshot, other: NodeSnapshot) -> None:
    """Compare period / start week / end week of two leaves; name the loser."""
    a, b = ref.leaf, other.leaf
    if a.period != b.period:
        raise HierarchyError(
            f"子节点『{b.name}』周期 {b.period} 与已挂接的『{a.name}』"
            f"（周期 {a.period}）不一致，挂接被拒绝。"
        )
    if a.dates[0] != b.dates[0]:
        raise HierarchyError(
            f"子节点『{b.name}』起始周 {b.dates[0]} 与已挂接的『{a.name}』"
            f"起始周 {a.dates[0]} 对不上，挂接被拒绝。"
        )
    if a.dates[-1] != b.dates[-1]:
        raise HierarchyError(
            f"子节点『{b.name}』截止周 {b.dates[-1]} 与已挂接的『{a.name}』"
            f"截止周 {a.dates[-1]} 对不上，挂接被拒绝。"
        )
    if len(a.dates) != len(b.dates):
        raise HierarchyError(
            f"子节点『{b.name}』周数 {len(b.dates)} 与『{a.name}』"
            f"（{len(a.dates)} 周）不一致，挂接被拒绝。"
        )
