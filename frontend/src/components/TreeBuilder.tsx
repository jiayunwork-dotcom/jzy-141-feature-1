import { useMemo, useState } from "react";
import { apiClient } from "../api/client";
import type { Series, TreeInfo } from "../api/types";
import {
  HierarchyAttachError,
  validateAttachment,
} from "../utils/hierarchy";

interface Props {
  tree: TreeInfo;
  series: Series[];
  changed: () => void;
}

/** Region creation + store attachment with immediate, specific rejection. */
export default function TreeBuilder({ tree, series, changed }: Props) {
  const [selectedParent, setSelectedParent] = useState<number | null>(null);
  const [newRegionName, setNewRegionName] = useState("");
  const [pickedSeries, setPickedSeries] = useState<number | null>(null);
  const [storeName, setStoreName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const regions = tree.nodes.filter((n) => n.kind === "region");
  const usedSeries = new Set(
    tree.nodes.filter((n) => n.series_id != null).map((n) => n.series_id!)
  );
  const available = useMemo(
    () => series.filter((s) => !usedSeries.has(s.id)),
    [series, usedSeries]
  );
  const rootId = tree.nodes.find((n) => n.kind === "network")?.id ?? null;

  async function guard(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      changed();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function addRegion() {
    if (rootId == null) return;
    const name = newRegionName.trim();
    if (!name) {
      setError("请填写区域名称。");
      return;
    }
    return guard(() =>
      apiClient.addTreeNode(tree.id, {
        parent_id: rootId,
        kind: "region",
        name,
      })
    ).then(() => setNewRegionName(""));
  }

  function attachStore() {
    setError(null);
    const parent =
      selectedParent ?? regions[0]?.id ?? null;
    if (parent == null) {
      setError("请先选择或创建一个区域。");
      return;
    }
    const s = available.find((x) => x.id === pickedSeries)
      ?? series.find((x) => x.id === pickedSeries);
    if (!s) {
      setError("请选择要挂接的门店序列。");
      return;
    }
    // Frontend mirror of backend alignment rules (also unit-tested).
    const attached = tree.nodes
      .filter((n) => n.kind === "store" && n.series_id != null)
      .map((n) => {
        const se = series.find((x) => x.id === n.series_id)!;
        return { name: n.name, period: se.period, dates: se.dates };
      });
    try {
      validateAttachment(
        { name: storeName.trim() || s.name, period: s.period, dates: s.dates },
        attached
      );
    } catch (e) {
      setError(
        e instanceof HierarchyAttachError
          ? e.message
          : `挂接校验失败：${(e as Error).message}`
      );
      return;
    }
    return guard(() =>
      apiClient.addTreeNode(tree.id, {
        parent_id: parent,
        kind: "store",
        name: storeName.trim() || s.name,
        series_id: s.id,
      })
    ).then(() => {
      setPickedSeries(null);
      setStoreName("");
    });
  }

  function remove(nodeId: number) {
    return guard(() => apiClient.removeTreeNode(tree.id, nodeId));
  }

  return (
    <div className="panel">
      <h2>管理层级（最多三层：全网 → 区域 → 门店）</h2>

      <div className="tree-list">
        {tree.nodes.map((n) => {
          const indent = n.level * 18;
          const linked =
            n.series_id != null
              ? series.find((s) => s.id === n.series_id)
              : null;
          return (
            <div key={n.id} className="tree-row" style={{ marginLeft: indent }}>
              <span className="tree-kind">{kindLabel(n.kind)}</span>
              <strong>{n.name}</strong>
              {linked && (
                <span className="muted">
                  {" "}
                  · 序列 #{linked.id} {linked.period} 周 ·{" "}
                  {linked.dates[0]} ~ {linked.dates[linked.dates.length - 1]}
                </span>
              )}
              {n.kind !== "network" && (
                <button
                  className="link-btn danger"
                  disabled={busy}
                  onClick={() => remove(n.id)}
                  title="删除该节点及其下级"
                >
                  删除
                </button>
              )}
            </div>
          );
        })}
      </div>

      {tree.ready_for_forecast === false && (
        <div className="alert warn">
          树尚未就绪：{tree.forecast_error}
        </div>
      )}
      {tree.error && <div className="alert error">{tree.error}</div>}
      {error && <div className="alert error">{error}</div>}

      <div className="row" style={{ marginTop: 10 }}>
        <div className="field">
          <label>新建区域（挂到全网下）</label>
          <input
            value={newRegionName}
            placeholder="例如：区域甲"
            onChange={(e) => setNewRegionName(e.target.value)}
          />
        </div>
        <button onClick={addRegion} disabled={busy}>
          新建区域
        </button>
      </div>

      <div className="row">
        <div className="field">
          <label>选择区域</label>
          <select
            value={selectedParent ?? regions[0]?.id ?? ""}
            onChange={(e) => setSelectedParent(Number(e.target.value))}
          >
            {regions.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>选择现有序列挂为门店</label>
          <select
            value={pickedSeries ?? ""}
            onChange={(e) => setPickedSeries(Number(e.target.value))}
          >
            <option value="">— 请选择 —</option>
            {available.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}（周期 {s.period}，{s.dates.length} 周，
                {s.dates[0]} 起）
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>门店显示名（可选）</label>
          <input
            value={storeName}
            placeholder="默认用序列名"
            onChange={(e) => setStoreName(e.target.value)}
          />
        </div>
        <button onClick={attachStore} disabled={busy}>
          挂接门店
        </button>
      </div>
    </div>
  );
}

function kindLabel(kind: string): string {
  return kind === "network" ? "全网" : kind === "region" ? "区域" : "门店";
}
