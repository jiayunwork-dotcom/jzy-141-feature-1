import type {
  BacktestRequest,
  BacktestResult,
  FitRequest,
  FitResult,
  HierarchyBacktestRequest,
  HierarchyBacktestResult,
  HierarchyForecastRequest,
  HierarchyForecast,
  HierarchyTreeDetail,
  HierarchyTreeSummary,
  Job,
  Series,
  TreeValidation,
} from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail !== undefined) detail = body.detail;
    } catch {
      /* ignore non-JSON error body */
    }
    // 挂接拒绝等场景 detail 是结构化对象（含 issues 列表），原样挂到 error
    // 上供页面逐条展示；message 仍尽量给出可读文本。
    const message =
      typeof detail === "string"
        ? detail
        : (detail as { message?: string })?.message ?? res.statusText;
    const err = new Error(message) as Error & { detail?: unknown };
    err.detail = detail;
    throw err;
  }
  return (await res.json()) as T;
}

export const apiClient = {
  listSeries: () => request<Series[]>("/series"),

  getSeries: (id: number) => request<Series>(`/series/${id}`),

  createSeries: (payload: {
    name: string;
    dates: string[];
    values: number[];
    period: number;
  }) =>
    request<Series>("/series", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  uploadSeries: async (
    file: File,
    name: string,
    period: number
  ): Promise<Series> => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    form.append("period", String(period));
    const res = await fetch(BASE + "/series/upload", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body?.detail ?? res.statusText);
    }
    return res.json();
  },

  startFit: (payload: FitRequest) =>
    request<Job>("/fits", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listFits: (seriesId: number) =>
    request<FitResult[]>(`/fits/by-series/${seriesId}`),

  getFit: (id: number) => request<FitResult>(`/fits/${id}`),

  startBacktest: (payload: BacktestRequest) =>
    request<Job>("/backtests", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listBacktests: (seriesId: number) =>
    request<BacktestResult[]>(`/backtests/by-series/${seriesId}`),

  getBacktest: (id: number) =>
    request<BacktestResult>(`/backtests/${id}`),

  getJob: (id: string) => request<Job>(`/jobs/${id}`),

  // ── 层级 ──────────────────────────────────────────────────────────
  listTrees: () => request<HierarchyTreeSummary[]>("/hierarchy/trees"),

  validateTree: (payload: {
    name: string;
    network_name: string;
    regions: Array<{ name: string; series_ids: number[] }>;
  }) =>
    request<TreeValidation>("/hierarchy/trees/validate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  createTree: (payload: {
    name: string;
    network_name: string;
    regions: Array<{ name: string; series_ids: number[] }>;
  }) =>
    request<{ tree_id: number }>("/hierarchy/trees", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getTree: (id: number) =>
    request<HierarchyTreeDetail>(`/hierarchy/trees/${id}`),

  startHierarchyForecast: (payload: HierarchyForecastRequest) =>
    request<Job>("/hierarchy/forecasts", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listHierarchyForecasts: (treeId: number) =>
    request<HierarchyForecast[]>(`/hierarchy/forecasts/by-tree/${treeId}`),

  getHierarchyForecast: (id: number) =>
    request<HierarchyForecast>(`/hierarchy/forecasts/${id}`),

  startHierarchyBacktest: (payload: HierarchyBacktestRequest) =>
    request<Job>("/hierarchy/backtests", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listHierarchyBacktests: (treeId: number) =>
    request<HierarchyBacktestResult[]>(`/hierarchy/backtests/by-tree/${treeId}`),

  getHierarchyBacktest: (id: number) =>
    request<HierarchyBacktestResult>(`/hierarchy/backtests/${id}`),
};
