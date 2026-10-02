import { describe, expect, it } from "vitest";
import {
  assertWeeklyGrid,
  AttachCandidate,
  HierarchyAttachError,
  MAX_HIERARCHY_DEPTH,
  validateAttachment,
  validateAttachments,
} from "../utils/hierarchy";

function dates(n: number, start = "2022-01-03"): string[] {
  const out: string[] = [];
  const d = new Date(`${start}T00:00:00Z`);
  for (let i = 0; i < n; i++) {
    out.push(
      new Date(d.getTime() + i * 7 * 86_400_000).toISOString().slice(0, 10)
    );
  }
  return out;
}

const ref: AttachCandidate = { name: "门店一", period: 52, dates: dates(164) };

describe("validateAttachment", () => {
  it("accepts the first child and aligned siblings", () => {
    expect(validateAttachment(ref, [])).toBe(true);
    expect(
      validateAttachment(
        { name: "门店二", period: 52, dates: dates(164) },
        [ref]
      )
    ).toBe(true);
  });

  it("rejects period mismatch and names child + both periods", () => {
    const child = { name: "门店二", period: 12, dates: dates(164) };
    expect(() => validateAttachment(child, [ref])).toThrow(
      HierarchyAttachError
    );
    try {
      validateAttachment(child, [ref]);
      throw new Error("should have thrown");
    } catch (e) {
      const msg = (e as Error).message;
      expect(msg).toContain("门店二");
      expect(msg).toContain("12");
      expect(msg).toContain("52");
      expect(msg).toContain("周期");
    }
  });

  it("rejects start-week mismatch and names child + dates", () => {
    const child = {
      name: "门店二",
      period: 52,
      dates: dates(164, "2022-01-10"),
    };
    expect(() => validateAttachment(child, [ref])).toThrow(/起始周/);
    try {
      validateAttachment(child, [ref]);
      throw new Error("should have thrown");
    } catch (e) {
      expect((e as Error).message).toContain("2022-01-10");
      expect((e as Error).message).toContain("2022-01-03");
    }
  });

  it("rejects end-week mismatch (different length) and names child", () => {
    const child = { name: "门店二", period: 52, dates: dates(160) };
    try {
      validateAttachment(child, [ref]);
      throw new Error("should have thrown");
    } catch (e) {
      const msg = (e as Error).message;
      expect(msg).toContain("门店二");
      expect(msg).toContain("截止周");
    }
  });

  it("rejects non-Monday week starts", () => {
    const child: AttachCandidate = {
      name: "歪周门店",
      period: 52,
      dates: ["2022-01-04", ...dates(163, "2022-01-10")],
    };
    expect(() => validateAttachment(child, [])).toThrow(/周一/);
  });

  it("rejects gaps inside the child series", () => {
    const ds = dates(10);
    ds.splice(5, 1); // drop one week -> 14-day gap
    const child: AttachCandidate = {
      name: "缺周门店",
      period: 52,
      dates: ds,
    };
    expect(() => validateAttachment(child, [])).toThrow(/缺周或重周/);
  });

  it("rejects empty history", () => {
    expect(() =>
      validateAttachment({ name: "空门店", period: 52, dates: [] }, [])
    ).toThrow(/没有任何历史周/);
  });

  it("batch validation stops at first offending child", () => {
    const good: AttachCandidate = {
      name: "门店二",
      period: 52,
      dates: dates(164),
    };
    const bad: AttachCandidate = {
      name: "坏门店",
      period: 52,
      dates: dates(164, "2023-01-02"),
    };
    try {
      validateAttachments([ref, good, bad]);
      throw new Error("should have thrown");
    } catch (e) {
      expect((e as Error).message).toContain("坏门店");
    }
  });

  it("exposes the three-level limit", () => {
    expect(MAX_HIERARCHY_DEPTH).toBe(3);
  });
});

describe("assertWeeklyGrid", () => {
  it("rejects unparseable dates", () => {
    expect(() =>
      assertWeeklyGrid({ name: "x", period: 52, dates: ["not-a-date"] })
    ).toThrow(HierarchyAttachError);
  });
});
