export interface Series {
  id: number;
  name: string;
  period: number;
  dates: string[];
  values: number[];
  missing_dates: string[];
  created_at: string;
}

export interface HWParams {
  alpha: number;
  beta: number;
  gamma: number;
  phi: number;
}

export interface CandidateScore {
  trend_kind: string;
  seasonal_kind: string;
  feasible: boolean;
  aic: number | null;
  sse: number | null;
  params: HWParams | null;
  reason: string | null;
}

export interface ForecastInfo {
  point: number[];
  lower: number[];
  upper: number[];
  level: number;
  method: string;
  residual_std: number;
  horizon: number;
  future_dates: string[];
}

export interface FitResult {
  id: number;
  series_id: number;
  created_at: string;
  label: string;
  auto: boolean;
  trend_kind: string;
  seasonal_kind: string;
  period: number;
  params: HWParams;
  locks: Record<string, number>;
  sse: number;
  aic: number;
  residuals: number[];
  fitted: number[];
  forecast: ForecastInfo;
  initial_state: {
    level: number;
    trend: number | null;
    season: number[];
  };
  final_state: {
    level: number;
    trend: number | null;
    season: number[];
    trend_kind: string;
    seasonal_kind: string;
    phi: number;
  };
  scores: CandidateScore[];
}

export interface OriginRow {
  origin: number;
  train_size: number;
  forecast: number[];
  actual: number[];
  naive_forecast: number[];
  errors: number[];
  naive_errors: number[];
  mae: number;
  mape: number;
  mase: number;
  naive_mae: number;
  naive_mape: number;
  naive_mase: number;
  params: HWParams;
  aic: number;
  scale: number;
}

export interface BacktestResult {
  id: number;
  series_id: number;
  created_at: string;
  label: string;
  origin_start: number;
  horizon: number;
  stride: number;
  confidence: number;
  interval_method: string;
  auto: boolean;
  trend_kind: string | null;
  seasonal_kind: string | null;
  locks: Record<string, number>;
  result: {
    period: number;
    horizon: number;
    origins: OriginRow[];
    model: { mae: number; mape: number; mase: number };
    naive: { mae: number; mape: number; mase: number };
    model_kind: { trend_kind: string; seasonal_kind: string };
  };
}

export interface Job {
  id: string;
  kind:
    | "fit"
    | "backtest"
    | "hierarchy_forecast"
    | "hierarchy_backtest";
  status: "pending" | "running" | "done" | "error";
  progress: number;
  stage: string;
  result:
    | {
        fit_id?: number;
        backtest_id?: number;
        hierarchy_forecast_id?: number;
        hierarchy_backtest_id?: number;
      }
    | null;
  error: string | null;
  series_id: number | null;
  created_id: number | null;
}

export interface FitRequest {
  series_id: number;
  horizon: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  auto: boolean;
  trend_kind?: string | null;
  seasonal_kind?: string | null;
  locks: Record<string, number>;
  label?: string;
}

export interface BacktestRequest {
  series_id: number;
  origin_start: number;
  horizon: number;
  stride: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  auto: boolean;
  trend_kind?: string | null;
  seasonal_kind?: string | null;
  locks: Record<string, number>;
  label?: string;
}

// ── 层级（门店—区域—全网）────────────────────────────────────────────

export interface AttachmentIssue {
  region_name: string;
  series_id: number;
  series_name: string;
  reason: string;
}

export interface TreeValidation {
  ok: boolean;
  missing_series: number[];
  issues: AttachmentIssue[];
}

export interface HierarchyNodeInfo {
  node_id: number;
  name: string;
  kind: "store" | "region" | "network";
  level: 0 | 1 | 2;
  parent_id: number | null;
  series_id: number | null;
  children: number[];
}

export interface HierarchyTreeSummary {
  id: number;
  name: string;
  period: number;
  created_at: string;
  n_nodes: number;
}

export interface HierarchyTreeDetail {
  id: number;
  name: string;
  period: number;
  dates: string[];
  root_id: number | null;
  nodes: HierarchyNodeInfo[];
  /** node_id（字符串键）-> 该节点逐周历史：门店为原始值，聚合节点为加总值 */
  histories: Record<string, number[]>;
}

export interface PointBand {
  point: number[];
  lower: number[];
  upper: number[];
}

export interface HierarchyForecastNode {
  node_id: number;
  kind: "store" | "region" | "network";
  level: 0 | 1 | 2;
  name: string;
  series_id: number | null;
  residual_var: number;
  base: PointBand;
  rec: PointBand;
  fit_ref: {
    fit_id?: number;
    agg_fit?: {
      params: HWParams;
      sse: number;
      aic: number;
      residual_std: number;
      trend_kind: string;
      seasonal_kind: string;
    };
  };
}

export interface StaleDetail {
  series_id: number;
  used_fit_id: number | null;
  latest_fit_id: number | null;
}

export interface HierarchyForecast {
  id: number;
  tree_id: number;
  created_at: string;
  label: string;
  horizon: number;
  confidence: number;
  interval_method: string;
  auto: boolean;
  trend_kind: string | null;
  seasonal_kind: string | null;
  locks: Record<string, number>;
  reconciliation: string;
  future_dates: string[];
  nodes: HierarchyForecastNode[];
  fit_refs: Record<string, number>;
  is_stale: boolean;
  stale_detail: StaleDetail[];
}

export interface HierarchyBacktestNodeRow {
  node_id: number;
  level: number;
  name: string;
  base_point: number[];
  rec_point: number[];
  actual: number[];
  naive_point: number[];
  scale_q: number | null;
  base_mae: number;
  rec_mae: number;
  naive_mae: number;
}

export interface HierarchyBacktestResult {
  id: number;
  tree_id: number;
  created_at: string;
  label: string;
  origin_start: number;
  horizon: number;
  stride: number;
  confidence: number;
  interval_method: string;
  auto: boolean;
  trend_kind: string | null;
  seasonal_kind: string | null;
  locks: Record<string, number>;
  reconciliation: string;
  result: {
    period: number;
    horizon: number;
    origin_start: number;
    stride: number;
    origins: Array<{ origin: number; nodes: HierarchyBacktestNodeRow[] }>;
    layers: Record<
      string,
      {
        label: string;
        base: { mae: number; mase: number };
        rec: { mae: number; mase: number };
        naive: { mae: number; mase: number };
      }
    >;
  };
}

export interface HierarchyForecastRequest {
  tree_id: number;
  horizon: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  auto: boolean;
  trend_kind?: string | null;
  seasonal_kind?: string | null;
  locks: Record<string, number>;
  label?: string;
  mode?: "full" | "auto";
}

export interface HierarchyBacktestRequest {
  tree_id: number;
  origin_start: number;
  horizon: number;
  stride: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  auto: boolean;
  trend_kind?: string | null;
  seasonal_kind?: string | null;
  locks: Record<string, number>;
  label?: string;
}
