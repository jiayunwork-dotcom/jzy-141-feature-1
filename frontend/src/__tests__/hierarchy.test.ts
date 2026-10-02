import { describe, expect, it } from "vitest";
import type { AssignableSeries } from "../utils/hierarchy";
import { checkAssignment, summarizeIssues } from "../utils/hierarchy";

function series(id: number, name: string, overrides: Partial<AssignableSeries> = {}): AssignableSeries {
  const dates = overrides.dates ?? weeklyDates("2022-01-03", 10);
  return {
    id,
    name,
    period: overrides.period ?? 52,
    dates,
  };
}

function weeklyDates(start: string, n: number): string[] {
  const d0 = Date.parse(start + "T00:00:00Z");
  const out: string[] = [];
  for (let k = 0; k < n; k++) {
    out.push(new Date(d0 + k * 7 * 864e5).toISOString().slice(0, 10));
  }
  return out;
}

function mapOf(...items: AssignableSeries[]): Map<number, AssignableSeries> {
  return new Map(items.map((s) => [s.id, s]));
}

describe("checkAssignment - 通过情形", () => {
  it("对齐的两区域树通过", () => {
    const a = series(1, "店1");
    const b = series(2, "店2");
    const c = series(3, "店3");
    const check = checkAssignment(
      [
        { name: "区域甲", series_ids: [1, 2] },
        { name: "区域乙", series_ids: [3] },
      ],
      mapOf(a, b, c)
    );
    expect(check.ok).toBe(true);
    expect(check.issues).toEqual([]);
  });
});

describe("checkAssignment - 拒绝并指出子节点与差异", () => {
  it("周期不一致时点名区域、门店与具体周期", () => {
    const a = series(1, "店1");
    const b = series(2, "店2", { period: 12 });
    const check = checkAssignment(
      [{ name: "区域甲", series_ids: [1, 2] }],
      mapOf(a, b)
    );
    expect(check.ok).toBe(false);
    expect(check.issues).toHaveLength(1);
    const issue = check.issues[0];
    expect(issue.region_name).toBe("区域甲");
    expect(issue.series_id).toBe(2);
    expect(issue.series_name).toBe("店2");
    expect(issue.reason).toContain("周期");
    expect(issue.reason).toContain("52");
    expect(issue.reason).toContain("12");
  });

  it("起始周不一致被点名", () => {
    const a = series(1, "店1");
    const b = series(2, "店2", { dates: weeklyDates("2022-01-10", 10) });
    const check = checkAssignment(
      [{ name: "区域甲", series_ids: [1, 2] }],
      mapOf(a, b)
    );
    expect(check.ok).toBe(false);
    expect(check.issues[0].reason).toContain("起始周");
  });

  it("周数（结束周）不一致被点名", () => {
    const a = series(1, "店1", { dates: weeklyDates("2022-01-03", 10) });
    const b = series(2, "店2", { dates: weeklyDates("2022-01-03", 9) });
    const check = checkAssignment(
      [{ name: "区域甲", series_ids: [1, 2] }],
      mapOf(a, b)
    );
    expect(check.ok).toBe(false);
    expect(check.issues[0].reason).toContain("周数");
  });

  it("同长同起止但中间周网格不同被点名（缺周差异）", () => {
    const datesA = weeklyDates("2022-01-03", 4);
    const datesB = [...weeklyDates("2022-01-03", 4)];
    datesB[2] = weeklyDates("2022-01-03", 5)[3]; // 第三个点错成第四周
    const a = series(1, "店1", { dates: datesA });
    const b = series(2, "店2", { dates: datesB });
    const check = checkAssignment(
      [{ name: "区域甲", series_ids: [1, 2] }],
      mapOf(a, b)
    );
    expect(check.ok).toBe(false);
    expect(check.issues[0].reason).toContain("第 3 周对不上");
  });

  it("同一门店跨区域重复挂接被拒绝", () => {
    const a = series(1, "店1");
    const b = series(2, "店2");
    const check = checkAssignment(
      [
        { name: "区域甲", series_ids: [1] },
        { name: "区域乙", series_ids: [1, 2] },
      ],
      mapOf(a, b)
    );
    expect(check.ok).toBe(false);
    expect(check.issues.some((i) => i.reason.includes("重复挂接"))).toBe(true);
    const dup = check.issues.find((i) => i.series_id === 1);
    expect(dup?.region_name).toBe("区域乙");
  });

  it("空区域被拒绝", () => {
    const a = series(1, "店1");
    const check = checkAssignment(
      [{ name: "空区域", series_ids: [] }],
      mapOf(a)
    );
    expect(check.ok).toBe(false);
    expect(check.issues[0].reason).toContain("至少需要一个门店");
  });

  it("引用了不存在的序列被点名", () => {
    const a = series(1, "店1");
    const check = checkAssignment(
      [{ name: "区域甲", series_ids: [1, 99] }],
      mapOf(a)
    );
    expect(check.ok).toBe(false);
    expect(check.issues.some((i) => i.series_id === 99)).toBe(true);
  });

  it("没有任何区域直接判失败", () => {
    const check = checkAssignment([], new Map());
    expect(check.ok).toBe(false);
    expect(check.issues[0].reason).toContain("至少需要一个区域");
  });

  it("多个问题节点会逐条列出", () => {
    const a = series(1, "店1");
    const b = series(2, "店2", { period: 12 });
    const c = series(3, "店3", { dates: weeklyDates("2022-01-03", 8) });
    const check = checkAssignment(
      [{ name: "区域甲", series_ids: [1, 2, 3] }],
      mapOf(a, b, c)
    );
    expect(check.issues).toHaveLength(2);
  });
});

describe("summarizeIssues", () => {
  it("把每条问题格式化为区域/门店/原因", () => {
    const text = summarizeIssues([
      {
        region_name: "区域甲",
        series_id: 2,
        series_name: "店2",
        reason: "季节周期不一致",
      },
    ]);
    expect(text).toContain("区域甲");
    expect(text).toContain("店2");
    expect(text).toContain("季节周期不一致");
  });
});
