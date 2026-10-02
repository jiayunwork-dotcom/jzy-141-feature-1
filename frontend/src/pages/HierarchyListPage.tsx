import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiClient } from "../api/client";
import type { HierarchyTreeSummary, Series } from "../api/types";
import TreeBuilder from "../components/hierarchy/TreeBuilder";

export default function HierarchyListPage() {
  const navigate = useNavigate();
  const [trees, setTrees] = useState<HierarchyTreeSummary[]>([]);
  const [series, setSeries] = useState<Series[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [t, s] = await Promise.all([
        apiClient.listTrees(),
        apiClient.listSeries(),
      ]);
      setTrees(t);
      setSeries(s);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <h1 className="page-title">层级管理（门店—区域—全网）</h1>
      {error && <div className="alert error">{error}</div>}
      <TreeBuilder series={series} onCreated={async (id) => {
        await load();
        navigate(`/hierarchy/${id}`);
      }} />

      <div className="panel">
        <h2>已建层级（{trees.length}）</h2>
        {loading ? (
          <div className="muted">加载中…</div>
        ) : trees.length === 0 ? (
          <div className="muted">还没有层级树，请在上方选择门店挂接到区域。</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>名称</th>
                  <th>周期</th>
                  <th>节点数</th>
                  <th>创建时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {trees.map((t) => (
                  <tr key={t.id}>
                    <td>{t.name}</td>
                    <td>{t.period}</td>
                    <td>{t.n_nodes}</td>
                    <td>{t.created_at.replace("T", " ").slice(0, 19)}</td>
                    <td>
                      <Link to={`/hierarchy/${t.id}`}>查看 / 层级预测 / 回测</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
