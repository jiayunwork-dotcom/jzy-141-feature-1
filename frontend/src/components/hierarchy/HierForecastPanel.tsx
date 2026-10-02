import { useState } from "react";

export interface HierForecastSettings {
  horizon: number;
  confidence: number;
  intervalMethod: "analytic" | "simulate";
  auto: boolean;
  trendKind: string;
  seasonalKind: string;
}

interface Props {
  settings: HierForecastSettings;
  onChange: (next: HierForecastSettings) => void;
  onRun: (mode: "full" | "auto") => void;
  busy: boolean;
  hasPrior: boolean;
}

/**
 * 层级预测设置：这里的选型只作用于 **区域/全网聚合节点**；
 * 门店叶子一律使用它们各自被引用的既有单序列拟合（数字不因挂入树而改变）。
 */
export default function HierForecastPanel({
  settings,
  onChange,
  onRun,
  busy,
  hasPrior,
}: Props) {
  const [showAdv, setShowAdv] = useState(false);
  const set = (patch: Partial<HierForecastSettings>) =>
    onChange({ ...settings, ...patch });

  return (
    <div className="panel">
      <h2>层级预测（WLS 调和）</h2>
      <div className="row">
        <div className="field">
          <label>聚合节点选型</label>
          <select
            value={settings.auto ? "auto" : "manual"}
            onChange={(e) => set({ auto: e.target.value === "auto" })}
          >
            <option value="auto">自动选型（按 AIC）</option>
            <option value="manual">手动指定组合</option>
          </select>
        </div>
        {!settings.auto && (
          <>
            <div className="field">
              <label>趋势项</label>
              <select
                value={settings.trendKind}
                onChange={(e) => set({ trendKind: e.target.value })}
              >
                <option value="none">无趋势</option>
                <option value="add">加法趋势</option>
                <option value="add_damped">加法趋势（阻尼）</option>
              </select>
            </div>
            <div className="field">
              <label>季节项</label>
              <select
                value={settings.seasonalKind}
                onChange={(e) => set({ seasonalKind: e.target.value })}
              >
                <option value="add">加法季节</option>
                <option value="mul">乘法季节</option>
              </select>
            </div>
          </>
        )}
        <div className="field">
          <label>预测步长（周）</label>
          <input
            type="number" min={1} max={104}
            value={settings.horizon}
            onChange={(e) => set({ horizon: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label>置信水平</label>
          <select
            value={settings.confidence}
            onChange={(e) => set({ confidence: Number(e.target.value) })}
          >
            <option value={0.8}>80%</option>
            <option value={0.9}>90%</option>
            <option value={0.95}>95%</option>
            <option value={0.99}>99%</option>
          </select>
        </div>
        <div className="field">
          <label>区间方法</label>
          <select
            value={settings.intervalMethod}
            onChange={(e) =>
              set({ intervalMethod: e.target.value as "analytic" | "simulate" })
            }
          >
            <option value="analytic">解析近似</option>
            <option value="simulate">模拟</option>
          </select>
        </div>
      </div>

      <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
        调和采用对角 WLS（按各节点一步残差方差加权）；区间为同一线性映射作用后的
        近似带，宽度随步长单调不减。门店叶子的基础预测与其单序列页完全一致。
        <button
          className="ghost"
          style={{ marginLeft: 8, padding: "2px 8px" }}
          onClick={() => setShowAdv((v) => !v)}
        >
          {showAdv ? "收起说明" : "局部重算说明"}
        </button>
      </div>
      {showAdv && (
        <div className="alert info" style={{ marginTop: 8, fontSize: 12 }}>
          「局部重算」只重算换了拟合的门店及其祖先路径的基础预测，其余节点复用
          上次结果；调和因 WLS 是全树稠密映射仍会整树重跑，与「整棵重算」逐日期
          一致（后端测试固定到 1e-9）。
        </div>
      )}

      <div className="row" style={{ marginTop: 14 }}>
        <button onClick={() => onRun("full")} disabled={busy}>
          {busy ? "计算中…" : "整棵预测（全量重算）"}
        </button>
        <button
          className="ghost"
          onClick={() => onRun("auto")}
          disabled={busy || !hasPrior}
          title={hasPrior ? "只重算受影响路径，调和结果与全量一致" : "需要先有一次结果"}
        >
          局部重算（仅受影响路径）
        </button>
      </div>
    </div>
  );
}
