import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HierarchyForecastNode, HierarchyTreeDetail } from "../../api/types";

interface Props {
  tree: HierarchyTreeDetail;
  nodeId: number;
  forecast: HierarchyForecastNode | null;
}

interface Datum {
  date: string;
  history?: number;
  base?: number;
  rec?: number;
}

/**
 * 单节点视图：该节点历史（门店=原始，聚合=下级逐周加总）+ 基础点预测
 * （虚线）+ 调和后点预测（实线）。区间在下方表格按节点展示。
 */
export default function HierarchyNodeChart({ tree, nodeId, forecast }: Props) {
  const history = tree.histories[String(nodeId)] ?? [];
  const data: Datum[] = tree.dates.map((date, i) => ({
    date,
    history: history[i],
  }));

  if (forecast) {
    // 调和后预测由后端 future_dates 对齐（树的最后一周之后连续 h 周）。
    // 这里从 tree.dates 末尾推未来日期，避免依赖外层传参。
    const futureDates = futureWeeks(tree.dates, forecast.rec.point.length);
    forecast.rec.point.forEach((v, i) => {
      data.push({
        date: futureDates[i],
        base: forecast.base.point[i],
        rec: v,
      });
    });
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={data} margin={{ top: 8, right: 18, bottom: 8, left: 4 }}>
          <CartesianGrid stroke="#eef1f5" />
          <XAxis dataKey="date" minTickGap={48} fontSize={11} />
          <YAxis fontSize={11} domain={["auto", "auto"]} />
          <Tooltip />
          <Line
            type="monotone" dataKey="history" name="历史（聚合节点为加总）"
            stroke="#33415c" dot={false} strokeWidth={1.6} isAnimationActive={false}
          />
          <Line
            type="monotone" dataKey="base" name="基础预测（调和前）"
            stroke="#b7791f" strokeDasharray="5 4" dot={false}
            strokeWidth={1.5} isAnimationActive={false}
          />
          <Line
            type="monotone" dataKey="rec" name="调和后预测"
            stroke="#1e7f4f" dot={false} strokeWidth={2} isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="legend">
        <span style={{ color: "#33415c" }}><span style={{ background: "#33415c" }} />历史</span>
        <span style={{ color: "#b7791f" }}><span style={{ background: "#b7791f" }} />基础（调和前）</span>
        <span style={{ color: "#1e7f4f" }}><span style={{ background: "#1e7f4f" }} />调和后</span>
      </div>
    </div>
  );
}

function futureWeeks(dates: string[], h: number): string[] {
  if (!dates.length || h <= 0) return [];
  const last = Date.parse(dates[dates.length - 1] + "T00:00:00Z");
  const out: string[] = [];
  for (let k = 1; k <= h; k++) {
    out.push(new Date(last + k * 7 * 864e5).toISOString().slice(0, 10));
  }
  return out;
}
