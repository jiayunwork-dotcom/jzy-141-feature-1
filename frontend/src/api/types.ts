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
  kind: "fit" | "backtest" | "hierarchy_forecast" | "hierarchy_backtest";
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

// ---------------------------------------------------------------- hierarchy

export interface HierarchyNodeInfo {
  id: number;
  tree_id: number;
  parent_id: number | null;
  kind: "network" | "region" | "store";
  name: string;
  level: number;
  series_id: number | null;
}

export interface TreeInfo {
  id: number;
  name: string;
  structure_revision: number;
  created_at: string;
  node_count: number;
  leaf_count: number;
  nodes: HierarchyNodeInfo[];
  valid?: boolean;
  ready_for_forecast?: boolean;
  error?: string | null;
  forecast_error?: string | null;
  period?: number | null;
  histories?: Record<string, { dates: string[]; values: number[] }>;
  forecasts?: { id: number; created_at: string; label: string; stale: boolean }[];
}

export interface ReconciledNodeResult {
  node_id: number;
  point: number[];
  lower: number[];
  upper: number[];
}

export interface HierarchyNodeResult {
  node_id: number;
  name: string;
  kind: "network" | "region" | "store";
  level: number;
  parent_id: number | null;
  series_id: number | null;
  history_dates: string[];
  history_values: number[];
  base: {
    node_id: number;
    series_id: number | null;
    trend_kind: string;
    seasonal_kind: string;
    point: number[];
    lower: number[];
    upper: number[];
    sse: number;
    aic: number;
    residual_std: number;
    fit_id: number | null;
  };
  reconciled: ReconciledNodeResult;
  diff_point: number[];
}

export interface HierarchyForecastResult {
  id: number;
  tree_id: number;
  created_at: string;
  label: string;
  horizon: number;
  confidence: number;
  interval_method: string;
  trend_kind: string;
  seasonal_kind: string;
  structure_revision: number;
  recompute_mode: "full" | "local";
  fit_refs: Record<string, number>;
  stale: boolean;
  stale_reasons: string[];
  result: {
    method: string;
    future_dates: string[];
    nodes: Record<string, HierarchyNodeResult>;
    recompute_mode?: string;
    reused_aggregate_node_ids?: number[];
  };
}

export interface HierarchyLevelStats {
  level: number;
  node_count: number;
  base_mae: number;
  reconciled_mae: number;
  base_mase: number;
  reconciled_mase: number;
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
  trend_kind: string;
  seasonal_kind: string;
  structure_revision: number;
  stale: boolean;
  result: {
    period: number;
    horizon: number;
    origins: number[];
    origin_dates: string[];
    per_node: Record<
      string,
      {
        origin: number;
        base_forecast: number[];
        reconciled_forecast: number[];
        actual: number[];
        base_mae: number;
        reconciled_mae: number;
        base_mase: number;
        reconciled_mase: number;
        scale: number;
      }[]
    >;
    by_level: Record<string, HierarchyLevelStats>;
    node_meta: Record<
      string,
      { name: string; kind: string; level: number; series_id: number | null }
    >;
  };
}

export interface HierarchyForecastRequest {
  tree_id: number;
  horizon: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  trend_kind: string;
  seasonal_kind: string;
  recompute_mode: "full" | "local";
  based_on_id?: number | null;
  label?: string;
}

export interface HierarchyBacktestRequest {
  tree_id: number;
  origin_start: number;
  horizon: number;
  stride: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  trend_kind: string;
  seasonal_kind: string;
  label?: string;
}
