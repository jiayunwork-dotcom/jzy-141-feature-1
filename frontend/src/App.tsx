import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import SeriesListPage from "./pages/SeriesListPage";
import FitPage from "./pages/FitPage";
import BacktestPage from "./pages/BacktestPage";
import HierarchyListPage from "./pages/HierarchyListPage";
import HierarchyDetailPage from "./pages/HierarchyDetailPage";

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>补货预测平台</h1>
        <div className="sub">Holt–Winters · 周销量 · 层级调和</div>
        <nav>
          <NavLink to="/series" className={({ isActive }) => (isActive ? "active" : "")}>
            序列列表
          </NavLink>
          <NavLink to="/hierarchy" className={({ isActive }) => (isActive ? "active" : "")}>
            门店—区域—全网
          </NavLink>
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/series" replace />} />
          <Route path="/series" element={<SeriesListPage />} />
          <Route path="/series/:seriesId/fit" element={<FitPage />} />
          <Route path="/series/:seriesId/backtest" element={<BacktestPage />} />
          <Route path="/hierarchy" element={<HierarchyListPage />} />
          <Route path="/hierarchy/:treeId" element={<HierarchyDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}
