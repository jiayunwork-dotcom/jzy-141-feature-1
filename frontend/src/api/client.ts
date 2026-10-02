import type {
  BacktestRequest,
  BacktestResult,
  FitRequest,
  FitResult,
  HierarchyBacktestRequest,
  HierarchyBacktestResult,
  HierarchyForecastRequest,
  HierarchyForecastResult,
  Job,
  Series,
  TreeInfo,
} from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore non-JSON error body */
    }
    throw new Error(detail);
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

  // -------------------------------------------------------------- hierarchy

  listTrees: () => request<TreeInfo[]>("/hierarchy/trees"),

  createTree: (name: string) =>
    request<TreeInfo>("/hierarchy/trees", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  getTree: (id: number) => request<TreeInfo>(`/hierarchy/trees/${id}`),

  addTreeNode: (
    treeId: number,
    payload: {
      parent_id: number;
      kind: "region" | "store";
      name: string;
      series_id?: number | null;
    }
  ) =>
    request(`/hierarchy/trees/${treeId}/nodes`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  removeTreeNode: (treeId: number, nodeId: number) =>
    request<{ deleted: number }>(
      `/hierarchy/trees/${treeId}/nodes/${nodeId}`,
      { method: "DELETE" }
    ),

  startHierarchyForecast: (payload: HierarchyForecastRequest) =>
    request<Job>("/hierarchy/forecasts", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listHierarchyForecasts: (treeId: number) =>
    request<HierarchyForecastResult[]>(
      `/hierarchy/trees/${treeId}/forecasts`
    ),

  getHierarchyForecast: (id: number) =>
    request<HierarchyForecastResult>(`/hierarchy/forecasts/${id}`),

  startHierarchyBacktest: (payload: HierarchyBacktestRequest) =>
    request<Job>("/hierarchy/backtests", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listHierarchyBacktests: (treeId: number) =>
    request<HierarchyBacktestResult[]>(
      `/hierarchy/trees/${treeId}/backtests`
    ),

  getHierarchyBacktest: (id: number) =>
    request<HierarchyBacktestResult>(`/hierarchy/backtests/${id}`),
};
