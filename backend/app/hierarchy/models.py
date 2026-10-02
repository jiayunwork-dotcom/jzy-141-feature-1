"""ORM models for the store–region–network hierarchy.

A tree has at most three levels:

    level 0  network (root, one per tree; region/network history is never
             uploaded — it is summed from below)
    level 1  region (aggregate, >= 1 children)
    level 2  store  (leaf; ``series_id`` points at an existing Series)

``HierarchyTree.structure_revision`` is bumped on every structural change
(add/remove node).  Persisted results store the revision they were computed
against; a mismatch marks them stale even without looking at fit versions.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from ..db import Base


class HierarchyTree(Base):
    __tablename__ = "hierarchy_trees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    # Bumped on every structural change; used for staleness checks.
    structure_revision: Mapped[int] = mapped_column(Integer, default=1)

    nodes: Mapped[list["HierarchyNode"]] = relationship(
        back_populates="tree",
        cascade="all, delete-orphan",
        order_by="HierarchyNode.id",
    )
    forecasts: Mapped[list["HierarchyForecast"]] = relationship(
        back_populates="tree", cascade="all, delete-orphan"
    )
    backtests: Mapped[list["HierarchyBacktest"]] = relationship(
        back_populates="tree", cascade="all, delete-orphan"
    )


class HierarchyNode(Base):
    __tablename__ = "hierarchy_nodes"
    __table_args__ = (
        UniqueConstraint("tree_id", "series_id",
                         name="uq_hierarchy_leaf_once"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tree_id: Mapped[int] = mapped_column(
        ForeignKey("hierarchy_trees.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("hierarchy_nodes.id", ondelete="CASCADE"),
        nullable=True, index=True,
    )
    # "network" | "region" | "store"
    kind: Mapped[str] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(255))
    level: Mapped[int] = mapped_column(Integer)
    # Only store leaves reference a series.
    series_id: Mapped[int | None] = mapped_column(
        ForeignKey("series.id", ondelete="RESTRICT"),
        nullable=True, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    tree: Mapped[HierarchyTree] = relationship(back_populates="nodes")
    parent: Mapped["HierarchyNode | None"] = relationship(
        remote_side="HierarchyNode.id",
        backref=backref(
            "children",
            order_by="HierarchyNode.id",
            cascade="all, delete-orphan",
        ),
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
    trend_kind: Mapped[str] = mapped_column(String(20))
    seasonal_kind: Mapped[str] = mapped_column(String(20))
    # Structural revision the result was computed against.
    structure_revision: Mapped[int] = mapped_column(Integer)
    # "full" | "local" — documented reconciliation recomputation paths.
    recompute_mode: Mapped[str] = mapped_column(String(16), default="full")
    # {"node_id": fit_id} for store leaves; aggregates carry their base fit
    # inline in ``result`` (aggregates have no uploaded history / Fit rows).
    fit_refs: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON)

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
    trend_kind: Mapped[str] = mapped_column(String(20))
    seasonal_kind: Mapped[str] = mapped_column(String(20))
    structure_revision: Mapped[int] = mapped_column(Integer)
    result: Mapped[dict] = mapped_column(JSON)

    tree: Mapped[HierarchyTree] = relationship(back_populates="backtests")
