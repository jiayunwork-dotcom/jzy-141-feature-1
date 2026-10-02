import { useCallback, useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type {
  HierarchyBacktestResult,
  HierarchyForecastResult,
  Job,
  Series,
  TreeInfo,
} from "../api/types";
import JobProgress from "../components/JobProgress";
import TreeBuilder from "../components/TreeBuilder";
import HierarchyBacktestPanel from "../components/HierarchyBacktestPanel";
import HierarchyForecastPanel from "../components/HierarchyForecastPanel";
import { useJob } from "../hooks/useJob";
import { fmtNumber } from "../utils/format";

export default function HierarchyPage() {
  const [trees, setTrees] = useState<TreeInfo[]>([]);
  const [treeId, setTreeId] = useState<number | null>(null);
  const [tree, setTree] = useState<TreeInfo | null>(null);
  const [series, setSeries] = useState<Series[]>([]);
  const [forecasts, setForecasts] = useState<HierarchyForecastResult[]>([]);
  const [backtests, setBacktests] = useState<HierarchyBacktestResult[]>([]);
  const [selectedForecast, setSelectedForecast] = useState<number | null>(
    null
  );
  const [selectedBacktest, setSelectedBacktest] = useState<number | null>(
    null
  );
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newTreeName, setNewTreeName] = useState("");

  const [horizon, setHorizon] = useState(12);
  const [confidence, setConfidence] = useState(0.95);
  const [trendKind, setTrendKind] = useState("add");
  const [seasonalKind, setSeasonalKind] = useState("add");
  const [btOrigin, setBtOrigin] = useState(2 * 52);
  const [btHorizon, setBtHorizon] = useState(8);
  const [btStride, setBtStride] = useState(8);

  const reloadAll = useCallback(async () => {
    const [ts, ss] = await Promise.all([
      apiClient.listTrees(),
      apiClient.listSeries(),
    ]);
    setTrees(ts);
    setSeries(ss);
    if (treeId != null) {
      const [t, fcs, bts] = await Promise.all([
        apiClient.getTree(treeId),
        apiClient.listHierarchyForecasts(treeId),
        apiClient.listHierarchyBacktests(treeId),
      ]);
      setTree(t);
      setForecasts(fcs);
      setBacktests(bts);
      setSelectedForecast((cur) => cur ?? fcs[0]?.id ?? null);
      setSelectedBacktest((cur) => cur ?? bts[0]?.id ?? null);
      if (t.period) setBtOrigin(2 * t.period);
    }
  }, [treeId]);

  useEffect(() => {
    reloadAll().catch((e) =>
      setError(e instanceof Error ? e.message : String(e))
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [treeId]);

  const onDone = useCallback(
    async (job: Job) => {
      if (job.status === "done") {
        await reloadAll();
        if (job.kind === "hierarchy_forecast" && job.created_id) {
          setSelectedForecast(job.created_id);
        }
        if (job.kind === "hierarchy_backtest" && job.created_id) {
          setSelectedBacktest(job.created_id);
        }
      }
    },
    [reloadAll]
  );
  const { job } = useJob(jobId, { onDone });

  async function createTree() {
    setError(null);
    const name = newTreeName.trim();
    if (!name) {
      setError("请填写树名称（如：全国全网）。");
      return;
    }
    try {
      const t = await apiClient.createTree(name);
      setNewTreeName("");
      setTreeId(t.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function runForecast(mode: "full" | "local", basedOn?: number) {
    if (treeId == null) return;
    setError(null);
    try {
      const j = await apiClient.startHierarchyForecast({
        tree_id: treeId,
        horizon,
        confidence,
        interval_method: "analytic",
        trend_kind: trendKind,
        seasonal_kind: seasonalKind,
        recompute_mode: mode,
        based_on_id: basedOn ?? null,
      });
      setJobId(j.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function runBacktest() {
    if (treeId == null) return;
    setError(null);
    try {
      const j = await apiClient.startHierarchyBacktest({
        tree_id: treeId,
        origin_start: btOrigin,
        horizon: btHorizon,
        stride: btStride,
        confidence,
        interval_method: "analytic",
        trend_kind: trendKind,
        seasonal_kind: seasonalKind,
      });
      setJobId(j.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const fc =
    forecasts.find((f) => f.id === selectedForecast) ?? forecasts[0] ?? null;
  const bt =
    backtests.find((b) => b.id === selectedBacktest) ?? backtests[0] ?? null;
  const busy = job?.status === "running" || job?.status === "pending";

  return (
    <div>
      <h1 className="page-title">门店—区域—全网 层级预测</h1>
      {error && <div className="alert error">{error}</div>}

      <div className="panel">
        <h2>选择 / 新建层级树</h2>
        <div className="row">
          <div className="field">
            <label>层级树</label>
            <select
              value={treeId ?? ""}
              onChange={(e) => setTreeId(e.target.value ? Number(e.target.value) : null)}
            >
              <option value="">— 请选择 —</option>
              {trees.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}（{t.leaf_count} 家门店）
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>新建树（自动带一个全网根节点）</label>
            <input
              value={newTreeName}
              placeholder="例如：全国全网"
              onChange={(e) => setNewTreeName(e.target.value)}
            />
          </div>
          <button onClick={createTree}>新建</button>
        </div>
      </div>

      {tree && (
        <>
          <TreeBuilder tree={tree} series={series} changed={reloadAll} />

          <div className="panel">
            <h2>发起层级预测</h2>
            <div className="row">
              <div className="field">
                <label>预测步长</label>
                <input
                  type="number" min={1} value={horizon}
                  onChange={(e) => setHorizon(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>置信水平</label>
                <select
                  value={confidence}
                  onChange={(e) => setConfidence(Number(e.target.value))}
                >
                  <option value={0.8}>80%</option>
                  <option value={0.9}>90%</option>
                  <option value={0.95}>95%</option>
                  <option value={0.99}>99%</option>
                </select>
              </div>
              <div className="field">
                <label>区域/全网趋势</label>
                <select
                  value={trendKind}
                  onChange={(e) => setTrendKind(e.target.value)}
                >
                  <option value="none">无</option>
                  <option value="add">加法</option>
                  <option value="add_damped">加法阻尼</option>
                </select>
              </div>
              <div className="field">
                <label>区域/全网季节</label>
                <select
                  value={seasonalKind}
                  onChange={(e) => setSeasonalKind(e.target.value)}
                >
                  <option value="add">加法</option>
                  <option value="mul">乘法</option>
                </select>
              </div>
              <button
                onClick={() => runForecast("full")}
                disabled={busy || tree.ready_for_forecast === false}
                title={
                  tree.ready_for_forecast === false
                    ? tree.forecast_error ?? ""
                    : ""
                }
              >
                整树预测
              </button>
              <button
                onClick={() => fc && runForecast("local", fc.id)}
                disabled={busy || !fc || tree.ready_for_forecast === false}
                title="只重算受影响路径（结果与整树一致）"
              >
                局部重算
              </button>
            </div>
            <div className="muted">
              门店基础预测取自各门店最新一次单序列拟合；区域/全网历史由下级逐周加总，
              用上面的模型现场拟合；调和采用自底向上加总，调和后父子逐周严格相等。
            </div>
          </div>

          {jobId && <JobProgress job={job} />}

          {fc && <HierarchyForecastPanel forecast={fc} />}

          <div className="panel">
            <h2>历史层级预测（可多次结果互相比照）</h2>
            {forecasts.length === 0 ? (
              <div className="muted">尚无层级预测。</div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th></th>
                      <th>时间</th>
                      <th>备注</th>
                      <th>模式</th>
                      <th>状态</th>
                      <th>过期原因 / 操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecasts.map((f) => (
                      <tr key={f.id} className={f.stale ? "stale-row" : ""}>
                        <td>
                          <input
                            type="radio"
                            checked={fc?.id === f.id}
                            onChange={() => setSelectedForecast(f.id)}
                          />
                        </td>
                        <td>{f.created_at.replace("T", " ").slice(0, 19)}</td>
                        <td>{f.label || "—"}</td>
                        <td>{f.recompute_mode === "local" ? "局部" : "整树"}</td>
                        <td>
                          {f.stale ? (
                            <span className="stale-badge">已过期</span>
                          ) : (
                            <span className="fresh-badge">最新</span>
                          )}
                        </td>
                        <td>
                          {f.stale ? (
                            <>
                              <span className="stale-reasons">
                                {f.stale_reasons.join("；")}
                              </span>{" "}
                              <button
                                disabled={busy}
                                onClick={() => runForecast("full")}
                              >
                                一键重算
                              </button>
                            </>
                          ) : (
                            <span className="muted">引用的均为最新拟合</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="panel">
            <h2>层级滚动原点回测</h2>
            <div className="row">
              <div className="field">
                <label>首个原点（≥ {2 * (tree.period ?? 52)}）</label>
                <input
                  type="number"
                  value={btOrigin}
                  onChange={(e) => setBtOrigin(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>步长 h</label>
                <input
                  type="number" min={1} value={btHorizon}
                  onChange={(e) => setBtHorizon(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>原点步长</label>
                <input
                  type="number" min={1} value={btStride}
                  onChange={(e) => setBtStride(Number(e.target.value))}
                />
              </div>
              <button
                onClick={runBacktest}
                disabled={busy || tree.ready_for_forecast === false}
              >
                开始层级回测
              </button>
            </div>
          </div>

          {bt && <HierarchyBacktestPanel backtest={bt} />}

          <div className="panel">
            <h2>历史层级回测</h2>
            {backtests.length === 0 ? (
              <div className="muted">尚无层级回测。</div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th></th>
                      <th>时间</th>
                      <th>原点</th>
                      <th>h</th>
                      <th>步长</th>
                      <th>状态</th>
                      <th>全网 基础/调和 MAE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtests.map((b) => (
                      <tr key={b.id} className={b.stale ? "stale-row" : ""}>
                        <td>
                          <input
                            type="radio"
                            checked={bt?.id === b.id}
                            onChange={() => setSelectedBacktest(b.id)}
                          />
                        </td>
                        <td>{b.created_at.replace("T", " ").slice(0, 19)}</td>
                        <td>{b.origin_start}</td>
                        <td>{b.horizon}</td>
                        <td>{b.stride}</td>
                        <td>
                          {b.stale ? (
                            <span className="stale-badge">已过期</span>
                          ) : (
                            <span className="fresh-badge">最新</span>
                          )}
                        </td>
                        <td>
                          {fmtNumber(
                            b.result.by_level.network?.base_mae ?? NaN, 3
                          )}{" "}
                          /{" "}
                          {fmtNumber(
                            b.result.by_level.network?.reconciled_mae ?? NaN, 3
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
