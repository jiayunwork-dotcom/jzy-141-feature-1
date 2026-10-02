import { useMemo, useState } from "react";
import { apiClient } from "../../api/client";
import type { Series } from "../../api/types";
import {
  checkAssignment,
  summarizeIssues,
  type AssignableSeries,
  type RegionDraft,
} from "../../utils/hierarchy";

interface Props {
  series: Series[];
  onCreated: (treeId: number) => void;
}

let regionSeq = 0;

function makeRegion(): { key: number; name: string; series_ids: number[] } {
  regionSeq += 1;
  return { key: regionSeq, name: `区域 ${regionSeq}`, series_ids: [] };
}

export default function TreeBuilder({ series, onCreated }: Props) {
  const [treeName, setTreeName] = useState("");
  const [networkName, setNetworkName] = useState("全网");
  const [regions, setRegions] = useState<
    Array<{ key: number; name: string; series_ids: number[] }>
  >(() => [makeRegion()]);
  const [issues, setIssues] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const seriesById = useMemo(() => {
    const m = new Map<number, AssignableSeries>();
    series.forEach((s) =>
      m.set(s.id, { id: s.id, name: s.name, period: s.period, dates: s.dates })
    );
    return m;
  }, [series]);

  const drafts: RegionDraft[] = regions.map((r) => ({
    name: r.name,
    series_ids: r.series_ids,
  }));

  // 选入即做本地镜像校验，挂接被拒时逐条说明原因。
  const localCheck = useMemo(
    () => checkAssignment(drafts, seriesById),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [regions, seriesById]
  );

  const assignedIds = useMemo(
    () => new Set(regions.flatMap((r) => r.series_ids)),
    [regions]
  );

  function updateRegion(key: number, patch: Partial<(typeof regions)[number]>) {
    setRegions((rs) =>
      rs.map((r) => (r.key === key ? { ...r, ...patch } : r))
    );
  }

  function toggleStore(key: number, sid: number) {
    setRegions((rs) =>
      rs.map((r) => {
        if (r.key !== key) return r;
        const has = r.series_ids.includes(sid);
        return {
          ...r,
          series_ids: has
            ? r.series_ids.filter((x) => x !== sid)
            : [...r.series_ids, sid],
        };
      })
    );
  }

  async function submit() {
    setIssues([]);
    const check = checkAssignment(drafts, seriesById);
    if (!check.ok) {
      setIssues([summarizeIssues(check.issues)]);
      return;
    }
    setBusy(true);
    try {
      // 后端会再用同一套规则校验，并在持久化事务中拒绝不一致挂接。
      const created = await apiClient.createTree({
        name: treeName || "未命名层级",
        network_name: networkName || "全网",
        regions: drafts.map((r) => ({ name: r.name, series_ids: r.series_ids })),
      });
      onCreated(created.tree_id);
    } catch (e) {
      const err = e as Error & { detail?: { issues?: { reason: string }[] } };
      const serverIssues = err.detail?.issues?.map((i) => i.reason);
      setIssues(serverIssues && serverIssues.length ? serverIssues : [err.message]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2>新建层级树（门店 → 区域 → 全网）</h2>
      <div className="row">
        <div className="field">
          <label>层级名称</label>
          <input
            value={treeName}
            onChange={(e) => setTreeName(e.target.value)}
            placeholder="例如：2026 上半年补货层级"
          />
        </div>
        <div className="field">
          <label>全网节点名</label>
          <input
            value={networkName}
            onChange={(e) => setNetworkName(e.target.value)}
            style={{ width: 140 }}
          />
        </div>
      </div>

      {regions.map((region) => (
        <div
          key={region.key}
          className="panel"
          style={{ background: "#f8fafc", margin: "12px 0" }}
        >
          <div className="row">
            <div className="field">
              <label>区域名</label>
              <input
                value={region.name}
                onChange={(e) =>
                  updateRegion(region.key, { name: e.target.value })
                }
                style={{ width: 180 }}
              />
            </div>
            {regions.length > 1 && (
              <button
                className="ghost danger"
                onClick={() =>
                  setRegions((rs) => rs.filter((r) => r.key !== region.key))
                }
              >
                删除区域
              </button>
            )}
          </div>
          <div className="table-wrap" style={{ marginTop: 10 }}>
            <table>
              <thead>
                <tr>
                  <th>选入</th>
                  <th>门店序列</th>
                  <th>周期</th>
                  <th>周数</th>
                  <th>起始周</th>
                  <th>结束周</th>
                </tr>
              </thead>
              <tbody>
                {series.map((s) => {
                  const checked = region.series_ids.includes(s.id);
                  const usedElsewhere = assignedIds.has(s.id) && !checked;
                  return (
                    <tr key={s.id}>
                      <td>
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={usedElsewhere}
                          onChange={() => toggleStore(region.key, s.id)}
                        />
                      </td>
                      <td>{s.name}</td>
                      <td>{s.period}</td>
                      <td>{s.dates.length}</td>
                      <td>{s.dates[0]}</td>
                      <td>{s.dates[s.dates.length - 1]}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      <div className="row">
        <button className="ghost" onClick={() => setRegions((rs) => [...rs, makeRegion()])}>
          + 添加区域
        </button>
        <button onClick={submit} disabled={busy}>
          {busy ? "提交中…" : "校验并创建层级树"}
        </button>
      </div>

      {!localCheck.ok && localCheck.issues.length > 0 && (
        <div className="alert warn" style={{ marginTop: 12, whiteSpace: "pre-line" }}>
          {summarizeIssues(localCheck.issues)}
        </div>
      )}
      {issues.map((msg, i) => (
        <div key={i} className="alert error" style={{ marginTop: 10, whiteSpace: "pre-line" }}>
          {msg}
        </div>
      ))}
    </div>
  );
}
