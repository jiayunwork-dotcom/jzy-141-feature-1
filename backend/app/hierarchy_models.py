"""层级结构（门店—区域—全网）的 ORM 模型。

新层级存储独立成模块，不混入既有 models.py；在 db.init_db 中一并注册建表。

* HierarchyNode  —— 树节点。叶子 ``kind="store"`` 引用既有 Series；
  聚合节点 ``kind in ("region","network")`` 不允许单独上传历史，其历史
  永远由下级逐周加总得到。
* HierarchyForecast —— 一次层级预测：每个节点一份基础预测 + 调和预测，
  并逐节点记录引用的拟合（门店引用单序列 Fit；聚合节点记录其确定性
  加总拟合的参数/SSE 快照，不写入 fits 表）。
* HierarchyBacktest —— 一次层级滚动原点回测。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base
from .models import Series


class HierarchyNode(Base):
    __tablename__ = "hierarchy_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tree_id: Mapped[int] = mapped_column(
        ForeignKey("hierarchy_trees.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # store | region | network
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # 0=门店 1=区域 2=全网（最多三层）
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("hierarchy_nodes.id", ondelete="CASCADE"), nullable=True
    )
    # 仅门店节点引用既有单序列
    series_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("series.id", ondelete="RESTRICT"), nullable=True, index=True
    )

    tree: Mapped["HierarchyTree"] = relationship(back_populates="nodes")
    series: Mapped[Optional[Series]] = relationship()


class HierarchyTree(Base):
    __tablename__ = "hierarchy_trees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # 普通整数而不是双向外键，避免 tree/nodes 两张表的循环 DDL；
    # root 节点同时也以 parent_id IS NULL 的形式存在于 hierarchy_nodes。
    root_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    period: Mapped[int] = mapped_column(Integer, nullable=False)
    dates: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    nodes: Mapped[list[HierarchyNode]] = relationship(
        back_populates="tree", cascade="all, delete-orphan",
        foreign_keys=[HierarchyNode.tree_id],
    )
    forecasts: Mapped[list["HierarchyForecast"]] = relationship(
        back_populates="tree", cascade="all, delete-orphan"
    )
    backtests: Mapped[list["HierarchyBacktest"]] = relationship(
        back_populates="tree", cascade="all, delete-orphan"
    )


class HierarchyForecast(Base):
    __tablename__ = "hierarchy_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tree_id: Mapped[int] = mapped_column(
        ForeignKey("hierarchy_trees.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    label: Mapped[str] = mapped_column(String(255), default="")
    horizon: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=0.95)
    interval_method: Mapped[str] = mapped_column(String(20), default="analytic")
    auto: Mapped[bool] = mapped_column(Boolean, default=True)  # 聚合节点选型
    trend_kind: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    seasonal_kind: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    locks: Mapped[dict] = mapped_column(JSON, default=dict)
    reconciliation: Mapped[str] = mapped_column(String(20), default="wls")
    future_dates: Mapped[list] = mapped_column(JSON, default=list)
    # nodes: [{node_id, level, name, fit_id(leaf) or agg_fit snapshot,
    #          base:{point,lower,upper}, rec:{point,lower,upper},
    #          residual_var, trend_kind, seasonal_kind}]
    nodes: Mapped[list] = mapped_column(JSON, default=list)
    # 门店 series_id -> 引用的 Fit id（任务实际使用的那一组）
    fit_refs: Mapped[dict] = mapped_column(JSON, default=dict)

    tree: Mapped[HierarchyTree] = relationship(back_populates="forecasts")


class HierarchyBacktest(Base):
    __tablename__ = "hierarchy_backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tree_id: Mapped[int] = mapped_column(
        ForeignKey("hierarchy_trees.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    label: Mapped[str] = mapped_column(String(255), default="")
    origin_start: Mapped[int] = mapped_column(Integer)
    horizon: Mapped[int] = mapped_column(Integer)
    stride: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.95)
    interval_method: Mapped[str] = mapped_column(String(20), default="analytic")
    auto: Mapped[bool] = mapped_column(Boolean, default=True)
    trend_kind: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    seasonal_kind: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    locks: Mapped[dict] = mapped_column(JSON, default=dict)
    reconciliation: Mapped[str] = mapped_column(String(20), default="wls")
    # 结构详见 hierarchy_service.run_hierarchy_backtest_job
    result: Mapped[dict] = mapped_column(JSON, default=dict)

    tree: Mapped[HierarchyTree] = relationship(back_populates="backtests")
