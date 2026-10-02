"""层级预测专用的纯工具：挂接兼容性校验、历史逐周加总。

这些函数不触碰数据库，便于在后端单测与（镜像同一套规则的）前端单测中
分别固定下来。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LeafSpec:
    """挂接校验所需的门店序列最小信息。"""

    series_id: int
    name: str
    period: int
    dates: Sequence[str]
    values: Sequence[float]


@dataclass
class AttachmentIssue:
    """一条挂接拒绝原因，精确定位到区域与子节点。"""

    region_name: str
    series_id: int
    series_name: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "region_name": self.region_name,
            "series_id": self.series_id,
            "series_name": self.series_name,
            "reason": self.reason,
        }


@dataclass
class TreeBuildCheck:
    ok: bool
    issues: List[AttachmentIssue]
    """挂接拒绝原因列表（可能多条，每个对不上的子节点一条）。"""
    duplicate_series: List[int] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
            "duplicate_series": list(self.duplicate_series or []),
        }


def _aligned_dates(a: LeafSpec, b: LeafSpec) -> Optional[str]:
    """返回两条叶子序列日期对不上的原因；对得上返回 None。"""
    if a.period != b.period:
        return f"季节周期不一致：{a.name} 为 {a.period}，{b.name} 为 {b.period}"
    if len(a.dates) != len(b.dates):
        return (
            f"周数不一致：{a.name} 有 {len(a.dates)} 周，"
            f"{b.name} 有 {len(b.dates)} 周"
        )
    if a.dates[0] != b.dates[0]:
        return f"起始周不一致：{a.name} 从 {a.dates[0]} 开始，{b.name} 从 {b.dates[0]} 开始"
    if a.dates[-1] != b.dates[-1]:
        return f"结束周不一致：{a.name} 止于 {a.dates[-1]}，{b.name} 止于 {b.dates[-1]}"
    # 起止相同且周数相同（周数据 7 天一步仍在单序列上传时保证），
    # 这里再逐周比对，缺周差异也会被抓到。
    for k, (da, db) in enumerate(zip(a.dates, b.dates)):
        if da != db:
            return f"第 {k + 1} 周对不上：{a.name} 是 {da}，{b.name} 是 {db}"
    return None


def check_tree_assignment(regions: Sequence[tuple[str, Sequence[LeafSpec]]]
                          ) -> TreeBuildCheck:
    """校验一棵「全网—区域—门店」树的挂接方案。

    ``regions`` 为 (区域名, 该区域下门店叶子) 列表。规则：

    * 至少一个区域，每个区域至少一个门店；
    * 同一门店在整棵树里只能出现一次；
    * 所有叶子的周期、起止周、逐周日期网格必须完全一致，否则按
      「哪个区域下的哪个子节点、差在哪里」逐条给出原因。
    """
    issues: List[AttachmentIssue] = []
    if not regions:
        return TreeBuildCheck(False, [AttachmentIssue(
            "(全网)", -1, "(无)", "至少需要一个区域。")])

    seen: dict[int, str] = {}
    duplicates: List[int] = []
    all_leaves: List[tuple[str, LeafSpec]] = []
    for region_name, leaves in regions:
        if not leaves:
            issues.append(AttachmentIssue(
                region_name, -1, "(无)", f"区域「{region_name}」下至少需要一个门店。"))
        for leaf in leaves:
            if leaf.series_id in seen:
                duplicates.append(leaf.series_id)
                issues.append(AttachmentIssue(
                    region_name, leaf.series_id, leaf.name,
                    f"门店「{leaf.name}」已挂在区域「{seen[leaf.series_id]}」下，"
                    "同一门店不能重复挂接。"))
            else:
                seen[leaf.series_id] = region_name
                all_leaves.append((region_name, leaf))

    # 以第一个叶子为基准，其余逐个比对（保证每个出错子节点都被点名）。
    if all_leaves:
        base_region, base = all_leaves[0]
        for region_name, leaf in all_leaves[1:]:
            reason = _aligned_dates(base, leaf)
            if reason:
                issues.append(AttachmentIssue(
                    region_name, leaf.series_id, leaf.name, reason))

    return TreeBuildCheck(not issues, issues, duplicates)


def aggregate_history(
    children_values: Sequence[Sequence[float]],
) -> np.ndarray:
    """把直接下级的历史逐周相加，得到父节点（区域/全网）历史。

    调用方必须已用 :func:`check_tree_assignment` 保证各子序列等长对齐。
    """
    if not children_values:
        raise ValueError("父节点至少需要一个子节点。")
    arr = [np.asarray(v, dtype=float) for v in children_values]
    n = arr[0].size
    for i, a in enumerate(arr[1:], start=1):
        if a.size != n:
            raise ValueError(
                f"子节点历史长度不一致：第 0 个为 {n}，第 {i} 个为 {a.size}。"
                "挂接校验应当已经拦截这种情况。")
    total = np.array(arr[0], dtype=float, copy=True)
    for a in arr[1:]:
        total = total + a
    return total
