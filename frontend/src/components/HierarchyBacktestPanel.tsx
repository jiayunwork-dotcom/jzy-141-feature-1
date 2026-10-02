import { useMemo } from "react";
import type { HierarchyBacktestResult } from "../api/types";
import { fmtNumber } from "../utils/format";

interface Props {
  backtest: HierarchyBacktestResult;
}

const LEVEL_LABEL: Record<string, string> = {
  network: "全网",
  region: "区域",
  store: "门店",
};

/** By-level base vs reconciled MAE/MASE so managers see who wins/loses. */
export default function HierarchyBacktestPanel({ backtest }: Props) {
  const { result } = backtest;

  const levelRows = useMemo(
    () => Object.entries(result.by_level).sort((a) => a[1].level),
    [result.by_level]
  );

  const nodeRows = useMemo(() => {
    return Object.entries(result.node_meta)
      .sort((a, b) => a[1].level - b[1].level || a[1].name.localeCompare(b[1].name))
      .map(([nid, meta]) => {
        const rows = result.per_node[nid] ?? [];
        const maeBase = avg(rows.map((r) => r.base_mae));
        const maeRec = avg(rows.map((r) => r.reconciled_mae));
        const maseBase = avg(rows.map((r) => r.base_mase));
        const maseRec = avg(rows.map((r) => r.reconciled_mase));
        return { nid, meta, maeBase, maeRec, maseBase, maseRec };
      });
  }, [result.node_meta, result.per_node]);

  return (
    <div className="panel">
      <h2>层级回测 · 调和前 vs 调和后（按层）</h2>
      <div className="muted" style={{ marginBottom: 8 }}>
        {result.origins.length} 个原点（{result.origin_dates[0]} ~{" "}
        {result.origin_dates[result.origin_dates.length - 1]}），步长{" "}
        {result.horizon}；负数方向的差表示调和让该层变好。
      </div>

      <table className="compare-table">
        <thead>
          <tr>
            <th>层级</th>
            <th>节点数</th>
            <th>基础 MAE</th>
            <th>调和 MAE</th>
            <th>MAE 变化</th>
            <th>基础 MASE</th>
            <th>调和 MASE</th>
            <th>MASE 变化</th>
          </tr>
        </thead>
        <tbody>
          {levelRows.map(([key, s]) => {
            const dMae = s.reconciled_mae - s.base_mae;
            const dMase = s.reconciled_mase - s.base_mase;
            return (
              <tr key={key}>
                <td>{LEVEL_LABEL[key] ?? key}</td>
                <td>{s.node_count}</td>
                <td>{fmtNumber(s.base_mae, 4)}</td>
                <td>{fmtNumber(s.reconciled_mae, 4)}</td>
                <td className={dMae < -1e-9 ? "winner" : dMae > 1e-9 ? "loser" : ""}>
                  {dMae > 0 ? "+" : ""}
                  {fmtNumber(dMae, 4)}
                </td>
                <td>{fmtNumber(s.base_mase, 4)}</td>
                <td>{fmtNumber(s.reconciled_mase, 4)}</td>
                <td className={dMase < -1e-9 ? "winner" : dMase > 1e-9 ? "loser" : ""}>
                  {dMase > 0 ? "+" : ""}
                  {fmtNumber(dMase, 4)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <h3 style={{ marginTop: 18 }}>逐节点</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>层</th>
              <th>节点</th>
              <th>基础 MAE</th>
              <th>调和 MAE</th>
              <th>基础 MASE</th>
              <th>调和 MASE</th>
            </tr>
          </thead>
          <tbody>
            {nodeRows.map((r) => (
              <tr key={r.nid}>
                <td>{LEVEL_LABEL[r.meta.kind] ?? r.meta.kind}</td>
                <td>{r.meta.name}</td>
                <td>{fmtNumber(r.maeBase, 4)}</td>
                <td>{fmtNumber(r.maeRec, 4)}</td>
                <td>{fmtNumber(r.maseBase, 4)}</td>
                <td>{fmtNumber(r.maseRec, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function avg(xs: number[]): number {
  const finite = xs.filter((x) => Number.isFinite(x));
  if (finite.length === 0) return Number.NaN;
  return finite.reduce((a, b) => a + b, 0) / finite.length;
}
