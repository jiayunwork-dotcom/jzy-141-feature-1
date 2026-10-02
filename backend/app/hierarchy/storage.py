"""Repository layer for hierarchy trees / forecasts / backtests."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    HierarchyBacktest,
    HierarchyForecast,
    HierarchyNode,
    HierarchyTree,
)


def create_tree(db: Session, name: str) -> HierarchyTree:
    tree = HierarchyTree(name=name, structure_revision=1)
    db.add(tree)
    db.flush()
    root = HierarchyNode(
        tree_id=tree.id,
        parent_id=None,
        kind="network",
        name=f"{name}（全网）",
        level=0,
        series_id=None,
    )
    db.add(root)
    db.flush()
    return tree


def get_tree(db: Session, tree_id: int) -> Optional[HierarchyTree]:
    return db.get(HierarchyTree, tree_id)


def list_trees(db: Session) -> List[HierarchyTree]:
    return list(
        db.scalars(
            select(HierarchyTree).order_by(HierarchyTree.created_at.desc())
        )
    )


def get_root(db: Session, tree_id: int) -> HierarchyNode:
    node = db.scalar(
        select(HierarchyNode)
        .where(HierarchyNode.tree_id == tree_id)
        .where(HierarchyNode.parent_id.is_(None))
    )
    if node is None:
        raise ValueError(f"树 {tree_id} 缺少根节点")
    return node


def list_nodes(db: Session, tree_id: int) -> List[HierarchyNode]:
    return list(
        db.scalars(
            select(HierarchyNode)
            .where(HierarchyNode.tree_id == tree_id)
            .order_by(HierarchyNode.level, HierarchyNode.id)
        )
    )


def get_node(db: Session, node_id: int) -> Optional[HierarchyNode]:
    return db.get(HierarchyNode, node_id)


def find_leaf_by_series(
    db: Session, tree_id: int, series_id: int
) -> Optional[HierarchyNode]:
    return db.scalar(
        select(HierarchyNode)
        .where(HierarchyNode.tree_id == tree_id)
        .where(HierarchyNode.series_id == series_id)
    )


def add_node(
    db: Session,
    tree: HierarchyTree,
    parent_id: int,
    kind: str,
    name: str,
    series_id: Optional[int] = None,
) -> HierarchyNode:
    parent = db.get(HierarchyNode, parent_id)
    if parent is None or parent.tree_id != tree.id:
        raise ValueError("上级节点不存在或不属于该树。")
    node = HierarchyNode(
        tree_id=tree.id,
        parent_id=parent_id,
        kind=kind,
        name=name,
        level=parent.level + 1,
        series_id=series_id,
    )
    db.add(node)
    tree.structure_revision += 1
    db.flush()
    return node


def delete_node(db: Session, tree: HierarchyTree, node_id: int) -> None:
    """Delete a node and its whole subtree (cascades on the FK)."""
    node = db.get(HierarchyNode, node_id)
    if node is None or node.tree_id != tree.id:
        raise ValueError("节点不存在或不属于该树。")
    if node.parent_id is None:
        raise ValueError("全网根节点不能删除。")
    db.delete(node)
    tree.structure_revision += 1
    db.flush()


def create_forecast(
    db: Session, tree_id: int, payload: Dict[str, Any]
) -> HierarchyForecast:
    obj = HierarchyForecast(tree_id=tree_id, **payload)
    db.add(obj)
    db.flush()
    return obj


def list_forecasts(db: Session, tree_id: int) -> List[HierarchyForecast]:
    return list(
        db.scalars(
            select(HierarchyForecast)
            .where(HierarchyForecast.tree_id == tree_id)
            .order_by(HierarchyForecast.created_at.desc())
        )
    )


def get_forecast(db: Session, forecast_id: int) -> Optional[HierarchyForecast]:
    return db.get(HierarchyForecast, forecast_id)


def create_backtest(
    db: Session, tree_id: int, payload: Dict[str, Any]
) -> HierarchyBacktest:
    obj = HierarchyBacktest(tree_id=tree_id, **payload)
    db.add(obj)
    db.flush()
    return obj


def list_backtests(db: Session, tree_id: int) -> List[HierarchyBacktest]:
    return list(
        db.scalars(
            select(HierarchyBacktest)
            .where(HierarchyBacktest.tree_id == tree_id)
            .order_by(HierarchyBacktest.created_at.desc())
        )
    )


def get_backtest(db: Session, backtest_id: int) -> Optional[HierarchyBacktest]:
    return db.get(HierarchyBacktest, backtest_id)
