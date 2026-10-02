import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { apiClient } from "../api/client";
import type {
  HierarchyBacktestResult,
  HierarchyForecast,
  HierarchyTreeDetail,
  Job,
} from "../api/types";
import JobProgress from "../components/JobProgress";
import HierarchyBacktestPanel from "../components/hierarchy/HierarchyBacktestPanel";
import HierForecastPanel, {
  type HierForecastSettings,
} from "../components/hierarchy/HierForecastPanel";
import HierarchyForecastView from "../components/hierarchy/HierarchyForecastView";
import HierarchyNodeChart from "../components/hierarchy/HierarchyNodeChart";
import { useJob } from "../hooks/useJob";

const LEVEL_LABEL: Record<number, string> = { 0: "门店", 1: "区域", 2: "全网" };

export default function HierarchyDetailPage() {
  const { treeId } = useParams();
  const id = Number(treeId);

  const [tree, setTree] = useState<HierarchyTreeDetail | null>(null);
  const [forecasts, setForecasts] = useState<HierarchyForecast[]>([]);
  const [backtests, setBacktests] = useState<HierarchyBacktestResult[]>([]);
  const [selectedForecastId, setSelectedForecastId] = useState<number | null>(
    null
  );
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [selectedBtId, setSelectedBtId] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const [fsettings, setFSettings] = useState<HierForecastSettings>({
    horizon: 12,
    confidence: 0.95,
    intervalMethod: "analytic",
    auto: true,
    trendKind: "add",
    seasonalKind: "add",
  });
  const [bsettings, setBSettings] = useState({
    originStart: 120,
    horizon: 8,
    stride: 17,
  });

  const loadAll = useCallback(async () => {
    const [t, fs, bts] = await Promise.all([
      apiClient.getTree(id),
      apiClient.listHierarchyForecasts(id),
      apiClient.listHierarchyBacktests(id),
    ]);
    setTree(t);
    setForecasts(fs);
    setBacktests(bts);
    setSelectedForecastId((cur) => cur ?? fs[0]?.id ?? null);
    setSelectedBtId((cur) => cur ?? bts[0]?.id ?? null);
    setSelectedNodeId((cur) => cur ?? t.nodes.find((n) => n.level === 2)?.node_id
      ?? t.nodes[0]?.node_id ?? null);
    setBSettings((s) => ({
      ...s,
      originStart: Math.max(2 * t.period, s.originStart),
    }));
  }, [id]);

  useEffect(() => {
    loadAll().catch((e) =>
      setFormError(e instanceof Error ? e.message : String(e))
    );
  }, [loadAll]);

  const onJobDone = useCallback(
    async (job: Job) => {
      if (job.status === "done") {
        await loadAll();
        if (job.kind === "hierarchy_forecast" && job.created_id) {
          setSelectedForecastId(job.created_id);
        }
        if (job.kind === "hierarchy_backtest" && job.created_id) {
          setSelectedBtId(job.created_id);
        }
      }
    },
    [loadAll]
  );
  const { job } = useJob(jobId, { onDone: onJobDone });

  const selectedForecast = useMemo(
    () => forecasts.find((f) => f.id === selectedForecastId) ?? forecasts[0] ?? null,
    [forecasts, selectedForecastId]
  );

  async function runForecast(mode: "full" | "auto") {
    setFormError(null);
    try {
      const j = await apiClient.startHierarchyForecast({
        tree_id: id,
        horizon: fsettings.horizon,
        confidence: fsettings.confidence,
        interval_method: fsettings.intervalMethod,
        auto: fsettings.auto,
        trend_kind: fsettings.auto ? null : fsettings.trendKind,
        seasonal_kind: fsettings.auto ? null : fsettings.seasonalKind,
        locks: {},
        mode,
      });
      setJobId(j.id);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  }

  async function runBacktest() {
    setFormError(null);
    try {
      const j = await apiClient.startHierarchyBacktest({
        tree_id: id,
        origin_start: bsettings.originStart,
        horizon: bsettings.horizon,
        stride: bsettings.stride,
        confidence: 0.95,
        interval_method: "analytic",
        auto: fsettings.auto,
        trend_kind: fsettings.auto ? null : fsettings.trendKind,
        seasonal_kind: fsettings.auto ? null : fsettings.seasonalKind,
        locks: {},
      });
      setJobId(j.id);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!tree) {
    return <div className="muted">加载中… {formError}</div>;
  }

  const orderedNodes = [...tree.nodes].sort(
    (a, b) => a.level - b.level || a.node_id - b.node_id
  );
  const selectedNode =
    tree.nodes.find((n) => n.node_id === selectedNodeId) ??
    orderedNodes[0] ??
    null;
  const selectedForecastNode =
    selectedForecast?.nodes.find((n) => n.node_id === selectedNode?.node_id) ??
    null;

  return (
    <div>
      <h1 className="page-title">
        层级 · {tree.name}{" "}
        <Link to="/hierarchy" className="muted" style={{ fontSize: 13 }}>
          ← 返回层级列表
        </Link>
      </h1>
      {formError && <div className="alert error">{formError}</div>}

      <div className="panel">
        <h2>树结构与节点历史（点击节点查看）</h2>
        <div className="row" style={{ gap: 8 }}>
          {orderedNodes.map((n) => (
            <button
              key={n.node_id}
              className={
                n.node_id === selectedNode?.node_id ? "" : "ghost"
              }
              onClick={() => setSelectedNodeId(n.node_id)}
              style={{ padding: "6px 10px" }}
            >
              {LEVEL_LABEL[n.level]}·{n.name}
            </button>
          ))}
        </div>
        {selectedNode && (
          <div style={{ marginTop: 14 }}>
            <HierarchyNodeChart
              tree={tree}
              nodeId={selectedNode.node_id}
              forecast={selectedForecastNode}
            />
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              {selectedNode.kind === "store"
                ? "门店历史为上传的单序列原始值。"
                : `${selectedNode.name} 的历史由 ${
                    tree.histories[String(selectedNode.node_id)]?.length ?? 0
                  } 周下级销量逐周加总得到，不接受单独上传。`}
            </div>
          </div>
        )}
      </div>

      <HierForecastPanel
        settings={fsettings}
        onChange={setFSettings}
        onRun={runForecast}
        busy={job?.status === "running" || job?.status === "pending"}
        hasPrior={forecasts.length > 0}
      />
      {jobId && <JobProgress job={job} />}

      <div className="panel">
        <h2>历史层级预测（可互相比较）</h2>
        {forecasts.length === 0 ? (
          <div className="muted">尚无层级预测结果。</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th>时间</th>
                  <th>步长</th>
                  <th>调和</th>
                  <th>状态</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {forecasts.map((f) => (
                  <tr key={f.id}>
                    <td>
                      <input
                        type="radio"
                        name="hfpick"
                        checked={f.id === selectedForecast?.id}
                        onChange={() => setSelectedForecastId(f.id)}
                      />
                    </td>
                    <td>{f.created_at.replace("T", " ").slice(0, 19)}</td>
                    <td>{f.horizon}</td>
                    <td>{f.reconciliation}</td>
                    <td>
                      {f.is_stale ? (
                        <span className="badge red">已过期</span>
                      ) : (
                        <span className="badge green">最新拟合</span>
                      )}
                    </td>
                    <td>
                      <button
                        className="ghost"
                        disabled={job?.status === "running"}
                        onClick={async () => {
                          // 一键重算：沿用该次设置，整棵重跑
                          setFSettings((s) => ({
                            ...s,
                            horizon: f.horizon,
                            confidence: f.confidence,
                            intervalMethod: f.interval_method as
                              | "analytic"
                              | "simulate",
                            auto: f.auto,
                            trendKind: f.trend_kind ?? "add",
                            seasonalKind: f.seasonal_kind ?? "add",
                          }));
                          try {
                            const j = await apiClient.startHierarchyForecast({
                              tree_id: id,
                              horizon: f.horizon,
                              confidence: f.confidence,
                              interval_method: f.interval_method as
                                | "analytic"
                                | "simulate",
                              auto: f.auto,
                              trend_kind: f.trend_kind,
                              seasonal_kind: f.seasonal_kind,
                              locks: f.locks,
                              mode: "auto",
                            });
                            setJobId(j.id);
                          } catch (e) {
                            setFormError(
                              e instanceof Error ? e.message : String(e)
                            );
                          }
                        }}
                      >
                        一键重算
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selectedForecast && (
        <HierarchyForecastView
          tree={tree}
          forecast={selectedForecast}
          selectedNodeId={selectedNode?.node_id ?? -1}
          onSelectNode={setSelectedNodeId}
        />
      )}

      <HierarchyBacktestPanel
        treeLength={tree.dates.length}
        period={tree.period}
        settings={bsettings}
        onChange={setBSettings}
        onRun={runBacktest}
        busy={job?.status === "running" || job?.status === "pending"}
        results={backtests}
        selectedId={selectedBtId}
        onSelect={setSelectedBtId}
      />
    </div>
  );
}
