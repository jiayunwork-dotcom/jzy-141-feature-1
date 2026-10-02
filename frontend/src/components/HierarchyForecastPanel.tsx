import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  HierarchyForecastResult,
  HierarchyNodeResult,
} from "../api/types";
import { fmtNumber } from "../utils/format";

interface Props {
  forecast: HierarchyForecastResult;
}

interface ChartDatum {
  date: string;
  history?: number;
  base?: number;
  reconciled?: number;
}

/** Per-node history + base vs reconciled forecast + their difference. */
export default function HierarchyForecastPanel({ forecast }: Props) {
  const nodes = useMemo(
    () => Object.values(forecast.result.nodes),
    [forecast]
  );
  const [picked, setPicked] = useState<number | null>(null);
  const node =
    nodes.find((n) => n.node_id === picked)
    ?? nodes.find((n) => n.kind === "network")
    ?? nodes[0];

  const data = useMemo<ChartDatum[]>(() => {
    const rows: ChartDatum[] = node.history_dates.map((date, i) => ({
      date,
      history: node.history_values[i],
    }));
    // Bridge row so the two forecast lines start at the last history point.
    const lastDate = rows[rows.length - 1]?.date ?? "";
    const lastValue = node.history_values[node.history_values.length - 1];
    rows.push({ date: lastDate, base: lastValue, reconciled: lastValue });
    forecast.result.future_dates.forEach((date, i) => {
      rows.push({
        date,
        base: node.base.point[i],
        reconciled: node.reconciled.point[i],
      });
    });
    return rows;
  }, [node, forecast.result.future_dates]);

  const maxDiff = Math.max(0, ...node.diff_point.map((d) => Math.abs(d)));

  return (
    <div className="panel">
      <h2>节点历史 / 基础预测 / 调和后预测</h2>

      <div className="row">
        <div className="field">
          <label>选择节点</label>
          <select
            value={node.node_id}
            onChange={(e) => setPicked(Number(e.target.value))}
          >
            {nodes.map((n) => (
              <option key={n.node_id} value={n.node_id}>
                {kindLabel(n.kind)} · {n.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>基础→调和最大绝对差</label>
          <div className="k-value">{fmtNumber(maxDiff, 4)}</div>
        </div>
        <div className="field">
          <label>引用拟合</label>
          <div className="muted" style={{ fontSize: 13 }}>
            {node.base.fit_id != null
              ? `fit_id=${node.base.fit_id}`
              : "聚合节点（按加总历史现场拟合）"}
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={data} margin={{ top: 8, right: 18, bottom: 8, left: 4 }}>
          <CartesianGrid stroke="#eef1f5" />
          <XAxis dataKey="date" minTickGap={48} fontSize={11} />
          <YAxis fontSize={11} domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="history"
            name="历史（叶子为原始，聚合为加总）"
            stroke="#2b3a55"
            dot={false}
            strokeWidth={1.6}
          />
          <Line
            type="monotone"
            dataKey="base"
            name="基础预测"
            stroke="#d98c2b"
            strokeDasharray="6 4"
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="reconciled"
            name="调和后预测"
            stroke="#1e7f4f"
            dot={false}
            strokeWidth={2}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>预测周</th>
              <th>基础预测</th>
              <th>调和后</th>
              <th>差（调和−基础）</th>
              <th>调和下限</th>
              <th>调和上限</th>
              <th>区间宽</th>
            </tr>
          </thead>
          <tbody>
            {forecast.result.future_dates.map((d, i) => {
              const width =
                node.reconciled.upper[i] - node.reconciled.lower[i];
              return (
                <tr key={d}>
                  <td>{d}</td>
                  <td>{fmtNumber(node.base.point[i], 3)}</td>
                  <td>{fmtNumber(node.reconciled.point[i], 3)}</td>
                  <td>{fmtNumber(node.diff_point[i], 4)}</td>
                  <td>{fmtNumber(node.reconciled.lower[i], 3)}</td>
                  <td>{fmtNumber(node.reconciled.upper[i], 3)}</td>
                  <td>{fmtNumber(width, 3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function kindLabel(kind: string): string {
  return kind === "network" ? "全网" : kind === "region" ? "区域" : "门店";
}
