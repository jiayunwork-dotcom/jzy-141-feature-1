"""层级调和（forecast reconciliation）内核：纯数值、无数据库。

调和方法的选择
==============
我们采用 **WLS（加权最小二乘）调和**（Hyndman et al. 2011 的线性调和框架，
权重只用各层自身的一步残差方差、不估跨层协方差，即对角 WLS）：

    tilde Y = S (S' W^{-1} S)^{-1} S' W^{-1} Y_hat = P_mu Y_hat

其中

* ``Y_hat`` —— 全部节点（门店/区域/全网）各自独立得到的 **基础点预测**，
  门店用它自己被锁定版本的拟合，区域/全网用下级历史加总后各自拟合；
* ``S`` —— 求和矩阵（n × m），描述每个节点等于哪些门店之和；
* ``W`` —— 对角矩阵，取各节点样本内一步残差方差；无噪声时数值为零，
  用一个相对地板 ``1e-8`` 倍最大方差、且不小于绝对地板托底，保证矩阵非
  奇异（地板只影响数值稳定性，不改变「本来就一致」情形的结果）。

选了什么、放弃了什么
--------------------
* **bottom-up（自下而上）**：调和后父节点恒等于门店之和，但完全丢弃区域、
  全网自身的拟合信息——这恰恰是区域经理最关心的两层。已放弃作为默认。
* **top-down（自上而下）**：偏信全网、按历史比例往下摊，门店层会被明显
  拉动，且比例选择本身主观。已放弃。
* **WLS（本实现）**：综合各层拟合误差，残差小（更可信）的层权重高；
  任意父节点每一期都等于直接下级之和（线性约束 ``Y = S Y_bottom``），
  三层逐期成立。单孩子区域因求和约束结构上与唯一门店逐期相等，无需特判。

对「局部重算」的影响
~~~~~~~~~~~~~~~~~~~~
WLS 的映射 ``P_mu`` 对整棵树是稠密的：任何一个节点基础预测/权重变化，
理论上所有节点的调和值都会被重新映射。因此本实现的「局部重算」语义是：

* **只重算受影响节点的基础预测与权重**（换了拟合的门店，以及它的全部
  祖先；其余节点直接复用上次落库的基础预测与权重），
* **调和步骤对整棵树重跑一次**。

因为调和是全局线性映射，「复用未变基础值 + 全树重映射」与「全部重算后
重映射」给出的调和结果逐期完全相同（测试固定到 1e-9）。不存在「只改一条
路径上的调和值、不碰兄弟节点」的廉价增量——那是 bottom-up/top-down 的
性质，不是 WLS 的性质，文档在此显式说明这一取舍。

区间
====
基础区间由各节点单序列的同一套（解析/模拟）机制给出。调和区间 **不重新**
做联合自助，而是把同一线性映射 ``P_mu`` 作用到基础下界/上界上：这保证
调和后区间与点预测使用相同的层间折让方向，是对真实联合预测区间的线性近
似，不再保持名义覆盖率（与单序列区间的关系在此显式说明）。映射后半宽
再做逐节点「累积最大值」，因此每个节点区间宽度随步长单调不减；单孩子区
域与其唯一门店逐期相同。单序列区间本身（叶子基础值）与既有序列页完全一
致，调和只发生在层级层。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class NodeBase:
    """一个节点在一次层级预测中的基础（未调和）结果。"""

    node_id: int
    level: int                 # 0=门店 1=区域 2=全网
    name: str
    point: np.ndarray          # shape (h,)
    lower: np.ndarray          # shape (h,)
    upper: np.ndarray          # shape (h,)
    residual_var: float        # 样本内一步残差方差（WLS 权重）
    fit_id: Optional[int] = None       # 门店：引用的单序列拟合 id
    trend_kind: str = ""
    seasonal_kind: str = ""


@dataclass
class ReconciledTree:
    h: int
    node_ids: List[int]
    node_levels: List[int]
    node_names: List[str]
    base_point: np.ndarray     # (n, h)
    base_lower: np.ndarray
    base_upper: np.ndarray
    rec_point: np.ndarray
    rec_lower: np.ndarray
    rec_upper: np.ndarray
    residual_var: np.ndarray   # (n,)
    bottom_ids: List[int]      # 门店层节点 id（m 个，列序与 S 一致）
    # node_id -> 在各矩阵中的行下标
    row_index: Dict[int, int] = field(default_factory=dict)
    mapping: Optional[np.ndarray] = None  # P_mu (n, n)，便于测试/局部重算

    def node_slice(self, node_id: int) -> dict:
        i = self.row_index[node_id]
        return {
            "node_id": node_id,
            "base_point": self.base_point[i],
            "base_lower": self.base_lower[i],
            "base_upper": self.base_upper[i],
            "rec_point": self.rec_point[i],
            "rec_lower": self.rec_lower[i],
            "rec_upper": self.rec_upper[i],
            "residual_var": float(self.residual_var[i]),
        }


def summing_matrix(
    node_ids: Sequence[int],
    bottom_ids: Sequence[int],
    descendants_bottom: Dict[int, List[int]],
) -> np.ndarray:
    """构造求和矩阵 S（n × m）。

    ``descendants_bottom[n]`` 给出节点 n 直接/间接覆盖到的门店 id 列表。
    """
    bindex = {b: j for j, b in enumerate(bottom_ids)}
    n, m = len(node_ids), len(bottom_ids)
    S = np.zeros((n, m))
    for i, nid in enumerate(node_ids):
        for b in descendants_bottom[nid]:
            S[i, bindex[b]] = 1.0
    return S


def _stabilized_weights(variance: np.ndarray) -> np.ndarray:
    """W = diag(variance)，对零方差加地板（无噪声合成数据时必需）。"""
    w = np.asarray(variance, dtype=float).copy()
    finite = w[np.isfinite(w)]
    ref = float(np.max(finite)) if finite.size else 1.0
    floor = max(ref * 1e-8, 1e-12)
    w = np.where((w > 0) & np.isfinite(w), w, floor)
    return np.maximum(w, floor)


def reconcile(bases: Sequence[NodeBase],
              bottom_ids: Sequence[int],
              descendants_bottom: Dict[int, List[int]],
              apply_intervals: bool = True) -> ReconciledTree:
    """对一次层级预测的全部节点基础值做 WLS 调和。

    要求各节点 ``point/lower/upper`` 长度相同（horizon 一致）。
    """
    node_ids = [b.node_id for b in bases]
    levels = [b.level for b in bases]
    names = [b.name for b in bases]
    n, h = len(bases), len(bases[0].point)
    for b in bases[1:]:
        if len(b.point) != h:
            raise ValueError("调和要求所有节点的预测步长一致。")

    Y = np.vstack([np.asarray(b.point, dtype=float) for b in bases])
    Lo = np.vstack([np.asarray(b.lower, dtype=float) for b in bases])
    Up = np.vstack([np.asarray(b.upper, dtype=float) for b in bases])
    var = np.array([max(float(b.residual_var), 0.0) for b in bases])

    S = summing_matrix(node_ids, bottom_ids, descendants_bottom)
    m = S.shape[1]
    w_diag = _stabilized_weights(var)
    Winv = np.diag(1.0 / w_diag)
    # F = (S' W^-1 S) 是 m×m 的对称正定矩阵（S 列满秩、W 正定）。
    F = S.T @ Winv @ S
    Finv = np.linalg.inv(F)
    # 调和映射：tilde Y = S Finv S' W^-1 Y
    P = S @ Finv @ S.T @ Winv

    Yr = P @ Y
    if apply_intervals:
        Lor = P @ Lo
        Upr = P @ Up
        # 以调和后点预测为中心，半宽取映射后半宽；再按步长累积最大值，
        # 保证宽度随 h 单调不减（线性映射不自动保证这一点）。
        half = np.maximum((Upr - Lor) / 2.0, 0.0)
        half = np.maximum.accumulate(half, axis=1)
        Lor = Yr - half
        Upr = Yr + half
    else:
        Lor, Upr = Yr.copy(), Yr.copy()

    row_index = {nid: i for i, nid in enumerate(node_ids)}
    return ReconciledTree(
        h=h, node_ids=list(node_ids), node_levels=list(levels),
        node_names=list(names),
        base_point=Y, base_lower=Lo, base_upper=Up,
        rec_point=Yr, rec_lower=Lor, rec_upper=Upr,
        residual_var=var, bottom_ids=list(bottom_ids),
        row_index=row_index, mapping=P,
    )


def coherence_residual(rec: ReconciledTree,
                       parent_children: Dict[int, List[int]]) -> np.ndarray:
    """父节点点预测 − 直接下级点预测之和，形状 (n_parents, h)。"""
    out: List[np.ndarray] = []
    for pid, children in parent_children.items():
        i = rec.row_index[pid]
        diff = rec.rec_point[i].copy()
        for cid in children:
            diff = diff - rec.rec_point[rec.row_index[cid]]
        out.append(diff)
    return np.vstack(out) if out else np.zeros((0, rec.h))


# ---------------------------------------------------------------------------
# 层级回测指标
# ---------------------------------------------------------------------------

def mae_mase(
    pred: np.ndarray, actual: np.ndarray, scale_q: float
) -> Dict[str, float]:
    """MAE 与 MASE（MASE 的尺度 Q 为训练段季节朴素一步 MAE）。"""
    err = np.asarray(pred, dtype=float) - np.asarray(actual, dtype=float)
    mae = float(np.mean(np.abs(err)))
    mase = float(mae / scale_q) if scale_q and np.isfinite(scale_q) and scale_q > 0 \
        else float("nan")
    return {"mae": mae, "mase": mase}


def seasonal_naive_scale(train: np.ndarray, m: int) -> float:
    if train.size <= m:
        return float("nan")
    return float(np.mean(np.abs(train[m:] - train[:-m])))
