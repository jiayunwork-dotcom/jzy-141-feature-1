/**
 * Hierarchy attachment validation (frontend mirror of the backend rules).
 *
 * Leaves (store series) can only be attached to a tree when every series is
 * aligned on the seasonal period and covers exactly the same weeks:
 *   - equal period
 *   - equal first week (start)
 *   - equal last week (end) — equal start+count implies equal count too
 *
 * The backend re-validates on save; this module gives the planner the
 * offending child and the concrete mismatch immediately in the UI / unit
 * tests.
 */

export interface AttachCandidate {
  /** node label / series name, used verbatim in rejection messages */
  name: string;
  period: number;
  /** ISO week-start dates, ordered ascending */
  dates: string[];
}

export interface AttachReference {
  name: string;
  period: number;
  dates: string[];
}

export class HierarchyAttachError extends Error {}

const MONDAY = 1; // ISO weekday

function isoWeekday(d: Date): number {
  // getDay(): Sunday=0; convert so Monday=1..Sunday=7
  return d.getDay() === 0 ? 7 : d.getDay();
}

function parseIso(s: string): Date {
  const d = new Date(`${s}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) {
    throw new HierarchyAttachError(`无法解析日期：${s}`);
  }
  return d;
}

/** Every date must be a Monday and consecutive 7-day steps. */
export function assertWeeklyGrid(c: AttachCandidate): void {
  if (c.dates.length === 0) {
    throw new HierarchyAttachError(`子节点『${c.name}』没有任何历史周。`);
  }
  for (let i = 0; i < c.dates.length; i++) {
    const d = parseIso(c.dates[i]);
    if (isoWeekday(d) !== MONDAY) {
      throw new HierarchyAttachError(
        `子节点『${c.name}』的周起始 ${c.dates[i]} 不是周一。`
      );
    }
    if (i > 0) {
      const prev = parseIso(c.dates[i - 1]);
      const gapDays = (d.getTime() - prev.getTime()) / 86_400_000;
      if (gapDays !== 7) {
        throw new HierarchyAttachError(
          `子节点『${c.name}』在 ${c.dates[i - 1]} 与 ${c.dates[i]} 之间缺周或重周（间隔 ${gapDays} 天）。`
        );
      }
    }
  }
}

/**
 * Validate that a new child can be attached next to the already attached
 * reference leaves.  Throws HierarchyAttachError naming the child and the
 * difference; returns true when compatible.
 */
export function validateAttachment(
  child: AttachCandidate,
  attached: AttachReference[]
): boolean {
  assertWeeklyGrid(child);
  if (attached.length === 0) return true;
  const ref = attached[0];
  if (child.period !== ref.period) {
    throw new HierarchyAttachError(
      `子节点『${child.name}』周期 ${child.period} 与已挂接的『${ref.name}』（周期 ${ref.period}）不一致，挂接被拒绝。`
    );
  }
  if (child.dates[0] !== ref.dates[0]) {
    throw new HierarchyAttachError(
      `子节点『${child.name}』起始周 ${child.dates[0]} 与已挂接的『${ref.name}』起始周 ${ref.dates[0]} 对不上，挂接被拒绝。`
    );
  }
  if (child.dates[child.dates.length - 1] !==
      ref.dates[ref.dates.length - 1]) {
    throw new HierarchyAttachError(
      `子节点『${child.name}』截止周 ${child.dates[child.dates.length - 1]} 与已挂接的『${ref.name}』截止周 ${ref.dates[ref.dates.length - 1]} 对不上，挂接被拒绝。`
    );
  }
  if (child.dates.length !== ref.dates.length) {
    throw new HierarchyAttachError(
      `子节点『${child.name}』周数 ${child.dates.length} 与『${ref.name}』（${ref.dates.length} 周）不一致，挂接被拒绝。`
    );
  }
  return true;
}

/** Validate a whole ordered batch; the first mismatch throws. */
export function validateAttachments(
  candidates: AttachCandidate[]
): boolean {
  const attached: AttachReference[] = [];
  for (const c of candidates) {
    validateAttachment(c, attached);
    attached.push(c);
  }
  return true;
}

export const MAX_HIERARCHY_DEPTH = 3;
