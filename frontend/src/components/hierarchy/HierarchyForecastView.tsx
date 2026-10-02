import type { HierarchyForecast, HierarchyTreeDetail } from "../../api/types";
import { fmtNumber } from "../../utils/format";

interface Props {
  tree: HierarchyTreeDetail;
  forecast: HierarchyForecast;
  selectedNodeId: number;
  onSelectNode: (id: number) => void;
}

const LEVEL_LABEL: Record<number, string> = { 0: "门店", 1: "区域", 2: "全网" };

/** 节点列表 + 每节点「基础 vs 调和」差异概览 + 逐步明细表。 */
export default function HierarchyForecastView({
  tree,
  forecast,
  selectedNodeId,
  onSelectNode,
}: Props) {
  const byId = new Map(forecast.nodes.map((n) => [n.node_id, n]));
  const selected =
    byId.get(selectedNodeId) ?? forecast.nodes[0] ?? null;

  return (
    <>
      <div className="panel">
        <h2>
          节点结果
          {forecast.is_stale && (
            <span className="badge red" style={{ marginLeft: 10 }}>
              已过期
            </span>
          )}
        </h2>
        {forecast.is_stale && (
          <div className="alert warn">
            该结果引用的拟合已不是最新：
            {forecast.stale_detail
              .map(
                (d) =>
                  `序列 ${d.series_id}（用拟合 #${d.used_fit_id ?? "?"}，最新 #${d.latest_fit_id ?? "?"}）`
              )
              .join("；")}
            。请点击上方「整棵预测」或「局部重算」获取最新数字。
          </div>
        )}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>层</th>
                <th>节点</th>
                <th>引用</th>
                <th>第一步 基础</th>
                <th>第一步 调和后</th>
                <th>第一步 差</th>
                <th>12 步内最大|差|</th>
                <th>调和区间末步宽</th>
              </tr>
            </thead>
            <tbody>
              {[...forecast.nodes]
                .sort((a, b) => a.level - b.level || a.node_id - b.node_id)
                .map((n) => {
                  const diffs = n.rec.point.map((v, i) => v - n.base.point[i]);
                  const maxAbs = Math.max(...diffs.map((d) => Math.abs(d)));
                  const lastWidth =
                    n.rec.upper[n.rec.upper.length - 1] -
                    n.rec.lower[n.rec.lower.length - 1];
                  return (
                    <tr
                      key={n.node_id}
                      style={{
                        cursor: "pointer",
                        background:
                          n.node_id === selected?.node_id ? "#eef4ff" : undefined,
                      }}
                      onClick={() => onSelectNode(n.node_id)}
                    >
                      <td>{LEVEL_LABEL[n.level]}</td>
                      <td>{n.name}</td>
                      <td className="muted">
                        {n.kind === "store"
                          ? `单序列拟合 #${n.fit_ref.fit_id}`
                          : "加总后聚合拟合"}
                      </td>
                      <td>{fmtNumber(n.base.point[0], 2)}</td>
                      <td>{fmtNumber(n.rec.point[0], 2)}</td>
                      <td>{fmtNumber(diffs[0], 3)}</td>
                      <td>{fmtNumber(maxAbs, 3)}</td>
                      <td>{fmtNumber(lastWidth, 2)}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <div className="panel">
          <h2>
            {LEVEL_LABEL[selected.level]} · {selected.name}：基础 vs 调和后
          </h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>周起始</th>
                  <th>基础点预测</th>
                  <th>调和后点预测</th>
                  <th>差（调和−基础）</th>
                  <th>调和下限</th>
                  <th>调和上限</th>
                  <th>区间宽</th>
                </tr>
              </thead>
              <tbody>
                {forecast.future_dates.map((date, i) => {
                  const width = selected.rec.upper[i] - selected.rec.lower[i];
                  return (
                    <tr key={date}>
                      <td>{date}</td>
                      <td>{fmtNumber(selected.base.point[i], 3)}</td>
                      <td>{fmtNumber(selected.rec.point[i], 3)}</td>
                      <td
                        style={{
                          color:
                            Math.abs(
                              selected.rec.point[i] - selected.base.point[i]
                            ) > 1e-9
                              ? "#b7791f"
                              : undefined,
                        }}
                      >
                        {fmtNumber(
                          selected.rec.point[i] - selected.base.point[i],
                          4
                        )}
                      </td>
                      <td>{fmtNumber(selected.rec.lower[i], 2)}</td>
                      <td>{fmtNumber(selected.rec.upper[i], 2)}</td>
                      <td>{fmtNumber(width, 2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
