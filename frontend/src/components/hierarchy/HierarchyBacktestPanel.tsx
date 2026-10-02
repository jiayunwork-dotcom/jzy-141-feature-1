import { useState } from "react";
import type { HierarchyBacktestResult } from "../../api/types";
import { fmtNumber } from "../../utils/format";

interface Settings {
  originStart: number;
  horizon: number;
  stride: number;
}

interface Props {
  treeLength: number;
  period: number;
  settings: Settings;
  onChange: (s: Settings) => void;
  onRun: () => void;
  busy: boolean;
  results: HierarchyBacktestResult[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

const LAYER_ORDER = ["0", "1", "2"];

/** 层级回测：参数 + 历史结果选择 + 按层的调和前/后 MAE、MASE 对照。 */
export default function HierarchyBacktestPanel({
  treeLength,
  period,
  settings,
  onChange,
  onRun,
  busy,
  results,
  selectedId,
  onSelect,
}: Props) {
  const [showOrigins, setShowOrigins] = useState(false);
  const minOrigin = 2 * period;
  const selected = results.find((r) => r.id === selectedId) ?? null;
  const set = (patch: Partial<Settings>) => onChange({ ...settings, ...patch });

  return (
    <div className="panel">
      <h2>层级滚动原点回测</h2>
      <div className="row">
        <div className="field">
          <label>首个原点（≥ {minOrigin}）</label>
          <input
            type="number" min={minOrigin} max={treeLength - 1}
            value={settings.originStart}
            onChange={(e) => set({ originStart: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label>步长 h（周）</label>
          <input
            type="number" min={1}
            value={settings.horizon}
            onChange={(e) => set({ horizon: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label>原点间隔</label>
          <input
            type="number" min={1}
            value={settings.stride}
            onChange={(e) => set({ stride: Number(e.target.value) })}
          />
        </div>
        <button onClick={onRun} disabled={busy}>
          {busy ? "回测中…" : "运行层级回测"}
        </button>
      </div>

      {results.length > 0 && (
        <div className="field" style={{ marginTop: 12 }}>
          <label>历史回测（可互相比较）</label>
          <select
            value={selectedId ?? ""}
            onChange={(e) => onSelect(Number(e.target.value))}
          >
            {results.map((r) => (
              <option key={r.id} value={r.id}>
                #{r.id} {r.created_at.replace("T", " ").slice(0, 19)} · 原点{" "}
                {r.origin_start} · h={r.horizon} · 间隔 {r.stride}
              </option>
            ))}
          </select>
        </div>
      )}

      {selected && (
        <>
          <div className="table-wrap" style={{ marginTop: 14 }}>
            <table>
              <thead>
                <tr>
                  <th>层</th>
                  <th>调和前 MAE</th>
                  <th>调和后 MAE</th>
                  <th>MAE 变化</th>
                  <th>调和前 MASE</th>
                  <th>调和后 MASE</th>
                  <th>季节朴素 MAE</th>
                  <th>季节朴素 MASE</th>
                </tr>
              </thead>
              <tbody>
                {LAYER_ORDER.filter((k) => selected.result.layers[k]).map((k) => {
                  const lay = selected.result.layers[k];
                  const dMae = lay.rec.mae - lay.base.mae;
                  return (
                    <tr key={k}>
                      <td>{lay.label}</td>
                      <td>{fmtNumber(lay.base.mae, 4)}</td>
                      <td>{fmtNumber(lay.rec.mae, 4)}</td>
                      <td
                        style={{
                          color:
                            Math.abs(dMae) < 1e-12
                              ? undefined
                              : dMae < 0
                              ? "var(--ok)"
                              : "var(--danger)",
                        }}
                      >
                        {dMae > 0 ? "+" : ""}
                        {fmtNumber(dMae, 4)}
                      </td>
                      <td>{fmtNumber(lay.base.mase, 4)}</td>
                      <td>{fmtNumber(lay.rec.mase, 4)}</td>
                      <td>{fmtNumber(lay.naive.mae, 4)}</td>
                      <td>{fmtNumber(lay.naive.mase, 4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
            绿色 = 调和后该层误差下降（受益）；红色 = 上升（受损）。每个原点只用
            截至该原点的数据完成整树基础预测与调和（后端有原点隔离测试）。
          </div>

          <button
            className="ghost"
            style={{ marginTop: 10 }}
            onClick={() => setShowOrigins((v) => !v)}
          >
            {showOrigins ? "收起逐原点明细" : "展开逐原点/节点明细"}
          </button>
          {showOrigins && (
            <div className="table-wrap" style={{ marginTop: 10, maxHeight: 360 }}>
              <table>
                <thead>
                  <tr>
                    <th>原点</th>
                    <th>层</th>
                    <th>节点</th>
                    <th>基础 MAE</th>
                    <th>调和 MAE</th>
                    <th>朴素 MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.result.origins.flatMap((o) =>
                    [...o.nodes]
                      .sort((a, b) => a.level - b.level)
                      .map((n) => (
                        <tr key={`${o.origin}-${n.node_id}`}>
                          <td>{o.origin}</td>
                          <td>{n.level}</td>
                          <td>{n.name}</td>
                          <td>{fmtNumber(n.base_mae, 4)}</td>
                          <td>{fmtNumber(n.rec_mae, 4)}</td>
                          <td>{fmtNumber(n.naive_mae, 4)}</td>
                        </tr>
                      ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
