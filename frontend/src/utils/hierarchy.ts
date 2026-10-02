/**
 * 层级挂接校验（前端镜像版）。
 *
 * 规则与后端 app/hierarchy_utils.py 的 check_tree_assignment 完全一致，
 * 让管理页在选入门店时就能即时给出「哪个区域的哪个子节点、差在哪里」；
 * 后端在 POST /api/hierarchy/trees 时会用同一套规则再校验一次。
 *
 * 规则：
 *  - 至少一个区域，每个区域至少一个门店；
 *  - 同一门店在整棵树里只能出现一次；
 *  - 所有叶子的季节周期、周数、起止周与逐周日期网格必须完全一致。
 */
import type { AttachmentIssue } from "../api/types";

export interface AssignableSeries {
  id: number;
  name: string;
  period: number;
  dates: string[];
}

export interface RegionDraft {
  name: string;
  series_ids: number[];
}

export interface AssignmentCheck {
  ok: boolean;
  issues: AttachmentIssue[];
}

function alignedReason(a: AssignableSeries, b: AssignableSeries): string | null {
  if (a.period !== b.period) {
    return `季节周期不一致：${a.name} 为 ${a.period}，${b.name} 为 ${b.period}`;
  }
  if (a.dates.length !== b.dates.length) {
    return `周数不一致：${a.name} 有 ${a.dates.length} 周，${b.name} 有 ${b.dates.length} 周`;
  }
  if (a.dates[0] !== b.dates[0]) {
    return `起始周不一致：${a.name} 从 ${a.dates[0]} 开始，${b.name} 从 ${b.dates[0]} 开始`;
  }
  if (a.dates[a.dates.length - 1] !== b.dates[b.dates.length - 1]) {
    return `结束周不一致：${a.name} 止于 ${a.dates[a.dates.length - 1]}，${b.name} 止于 ${b.dates[b.dates.length - 1]}`;
  }
  for (let k = 0; k < a.dates.length; k++) {
    if (a.dates[k] !== b.dates[k]) {
      return `第 ${k + 1} 周对不上：${a.name} 是 ${a.dates[k]}，${b.name} 是 ${b.dates[k]}`;
    }
  }
  return null;
}

export function checkAssignment(
  regions: RegionDraft[],
  seriesById: Map<number, AssignableSeries>
): AssignmentCheck {
  const issues: AttachmentIssue[] = [];
  if (regions.length === 0) {
    return {
      ok: false,
      issues: [
        {
          region_name: "(全网)",
          series_id: -1,
          series_name: "(无)",
          reason: "至少需要一个区域。",
        },
      ],
    };
  }

  const seen = new Map<number, string>();
  const allLeaves: Array<{ regionName: string; series: AssignableSeries }> = [];

  for (const region of regions) {
    if (region.series_ids.length === 0) {
      issues.push({
        region_name: region.name,
        series_id: -1,
        series_name: "(无)",
        reason: `区域「${region.name}」下至少需要一个门店。`,
      });
    }
    for (const sid of region.series_ids) {
      const series = seriesById.get(sid);
      if (!series) {
        issues.push({
          region_name: region.name,
          series_id: sid,
          series_name: "(缺失)",
          reason: `门店序列 ${sid} 不存在或已被删除。`,
        });
        continue;
      }
      const priorRegion = seen.get(sid);
      if (priorRegion !== undefined) {
        issues.push({
          region_name: region.name,
          series_id: sid,
          series_name: series.name,
          reason: `门店「${series.name}」已挂在区域「${priorRegion}」下，同一门店不能重复挂接。`,
        });
      } else {
        seen.set(sid, region.name);
        allLeaves.push({ regionName: region.name, series });
      }
    }
  }

  if (allLeaves.length > 0) {
    const base = allLeaves[0];
    for (const { regionName, series } of allLeaves.slice(1)) {
      const reason = alignedReason(base.series, series);
      if (reason) {
        issues.push({
          region_name: regionName,
          series_id: series.id,
          series_name: series.name,
          reason,
        });
      }
    }
  }

  return { ok: issues.length === 0, issues };
}

/** 从校验问题里取出可读的一行（管理页顶部用）。 */
export function summarizeIssues(issues: AttachmentIssue[]): string {
  return issues.map((i) => `【${i.region_name}】${i.series_name}：${i.reason}`).join("\n");
}
