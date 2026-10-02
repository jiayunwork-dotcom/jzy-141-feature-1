"""层级结构与层级结果的仓储层（repository）。

职责：在 SQLAlchemy 会话上创建/读取树、节点、层级预测与层级回测；
树结构在内存中以 dataclass 表示，便于服务层与测试使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .hierarchy_models import (
    HierarchyBacktest,
    HierarchyForecast,
    HierarchyNode,
    HierarchyTree,
)
from .models import Series


@dataclass
class StoreRef:
    series_id: int
    name: str
    period: int
    dates: List[str]
    values: List[float]


@dataclass
class NodeInfo:
    node_id: int
    kind: str          # store | region | network
    level: int         # 0 | 1 | 2
    name: str
    parent_id: Optional[int]
    series_id: Optional[int]
    children: List[int] = field(default_factory=list)


@dataclass
class TreeInfo:
    tree_id: int
    name: str
    period: int
    dates: List[str]
    root_id: Optional[int]
    nodes: Dict[int, NodeInfo] = field(default_factory=dict)

    # ---- 遍历辅助 ----
    @property
    def bottom_ids(self) -> List[int]:
        return [nid for nid, n in self.nodes.items() if n.kind == "store"]

    def ancestors_of(self, node_id: int) -> List[int]:
        """自底向上返回祖先：父、祖父……根（不含自己）。"""
        out: List[int] = []
        cur = self.nodes[node_id].parent_id
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent_id
        return out

    def path_to_root_including(self, node_id: int) -> List[int]:
        return [node_id] + self.ancestors_of(node_id)

    def descendants_bottom(self) -> Dict[int, List[int]]:
        """每个节点 -> 它覆盖到的门店节点 id 列表（求和矩阵的依据）。"""
        out: Dict[int, List[int]] = {
            nid: [nid] for nid, n in self.nodes.items() if n.kind == "store"
        }
        # 自底向上合并；树最多三层，简单不动点即可。
        while True:
            progressed = False
            for nid, n in self.nodes.items():
                if nid in out:
                    continue
                if n.children and all(c in out for c in n.children):
                    leaves: List[int] = []
                    for c in n.children:
                        leaves.extend(out[c])
                    out[nid] = leaves
                    progressed = True
            if not progressed:
                break
        return out

    def parent_children(self) -> Dict[int, List[int]]:
        return {nid: list(n.children)
                for nid, n in self.nodes.items() if n.children}

    def ordered_node_ids(self) -> List[int]:
        """根在最前、门店在最后的稳定顺序（用于结果列表/矩阵）。"""
        return sorted(self.nodes, key=lambda nid: (
            self.nodes[nid].level, nid))


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def get_store_refs(db: Session, series_ids: List[int]) -> Dict[int, StoreRef]:
    refs: Dict[int, StoreRef] = {}
    if not series_ids:
        return refs
    rows = db.scalars(select(Series).where(Series.id.in_(series_ids)))
    for s in rows:
        refs[s.id] = StoreRef(
            series_id=s.id, name=s.name, period=int(s.period),
            dates=list(s.dates), values=[float(v) for v in s.values],
        )
    return refs


def load_tree(db: Session, tree_id: int) -> Optional[TreeInfo]:
    tree = db.get(HierarchyTree, tree_id)
    if tree is None:
        return None
    rows = list(db.scalars(
        select(HierarchyNode)
        .where(HierarchyNode.tree_id == tree_id)
        .order_by(HierarchyNode.id)
    ))
    info = TreeInfo(
        tree_id=tree.id, name=tree.name, period=int(tree.period),
        dates=list(tree.dates), root_id=tree.root_id,
    )
    for r in rows:
        info.nodes[r.id] = NodeInfo(
            node_id=r.id, kind=r.kind, level=int(r.level), name=r.name,
            parent_id=r.parent_id, series_id=r.series_id,
        )
    for r in rows:
        if r.parent_id is not None:
            info.nodes[r.parent_id].children.append(r.id)
    return info


def list_trees(db: Session) -> List[HierarchyTree]:
    return list(db.scalars(
        select(HierarchyTree).order_by(HierarchyTree.created_at.desc())
    ))


# ---------------------------------------------------------------------------
# 创建树
# ---------------------------------------------------------------------------

def create_tree(
    db: Session,
    name: str,
    period: int,
    dates: List[str],
    regions: List[Dict[str, Any]],
    network_name: str,
) -> HierarchyTree:
    """按已通过挂接校验的结构落库。

    ``regions``: [{"name": str, "stores": [StoreRef, ...]}]，顺序即区域
    在全网下的顺序。
    """
    tree = HierarchyTree(name=name, period=period, dates=list(dates))
    db.add(tree)
    db.flush()

    root = HierarchyNode(
        tree_id=tree.id, name=network_name, kind="network", level=2,
        parent_id=None, series_id=None,
    )
    db.add(root)
    db.flush()

    for region in regions:
        rnode = HierarchyNode(
            tree_id=tree.id, name=region["name"], kind="region", level=1,
            parent_id=root.id, series_id=None,
        )
        db.add(rnode)
        db.flush()
        for ref in region["stores"]:
            db.add(HierarchyNode(
                tree_id=tree.id, name=ref.name, kind="store", level=0,
                parent_id=rnode.id, series_id=ref.series_id,
            ))
    db.flush()
    tree.root_id = root.id
    db.flush()
    return tree


# ---------------------------------------------------------------------------
# 层级预测结果
# ---------------------------------------------------------------------------

def create_hierarchy_forecast(
    db: Session, tree_id: int, payload: Dict[str, Any]
) -> HierarchyForecast:
    obj = HierarchyForecast(tree_id=tree_id, **payload)
    db.add(obj)
    db.flush()
    return obj


def list_hierarchy_forecasts(
    db: Session, tree_id: int
) -> List[HierarchyForecast]:
    return list(db.scalars(
        select(HierarchyForecast)
        .where(HierarchyForecast.tree_id == tree_id)
        .order_by(HierarchyForecast.created_at.desc())
    ))


def get_hierarchy_forecast(
    db: Session, forecast_id: int
) -> Optional[HierarchyForecast]:
    return db.get(HierarchyForecast, forecast_id)


def latest_fit_ids_for_series(
    db: Session, series_ids: List[int]
) -> Dict[int, int]:
    """每个门店序列当前最新一次拟合的 id（过期判定用）。

    fit id 随插入单调递增，最大 id 即最新一次拟合。
    """
    from .models import Fit
    out: Dict[int, int] = {}
    for sid in series_ids:
        fid = db.scalars(
            select(Fit.id).where(Fit.series_id == sid)
            .order_by(Fit.id.desc()).limit(1)
        ).first()
        if fid is not None:
            out[sid] = int(fid)
    return out


# ---------------------------------------------------------------------------
# 层级回测结果
# ---------------------------------------------------------------------------

def create_hierarchy_backtest(
    db: Session, tree_id: int, payload: Dict[str, Any]
) -> HierarchyBacktest:
    obj = HierarchyBacktest(tree_id=tree_id, **payload)
    db.add(obj)
    db.flush()
    return obj


def list_hierarchy_backtests(
    db: Session, tree_id: int
) -> List[HierarchyBacktest]:
    return list(db.scalars(
        select(HierarchyBacktest)
        .where(HierarchyBacktest.tree_id == tree_id)
        .order_by(HierarchyBacktest.created_at.desc())
    ))


def get_hierarchy_backtest(
    db: Session, bt_id: int
) -> Optional[HierarchyBacktest]:
    return db.get(HierarchyBacktest, bt_id)
