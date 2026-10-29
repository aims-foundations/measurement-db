import type { AlignedManifest, Counts, Interval, TestGroup } from "./types";

const BLOCK = 256;

/** A byte per cell, plus small block sums; no millions of JS record objects. */
export class ScoreMatrix {
  private readonly stride: number;
  private readonly passed: Uint32Array;
  private readonly failed: Uint32Array;

  constructor(
    readonly manifest: AlignedManifest,
    readonly values: Uint8Array,
  ) {
    const { columns, subjects } = manifest;
    if (values.length !== columns * subjects.length)
      throw new Error("Score dimensions do not match the test index.");
    this.stride = Math.ceil(columns / BLOCK) + 1;
    this.passed = new Uint32Array(subjects.length * this.stride);
    this.failed = new Uint32Array(subjects.length * this.stride);
    let totalPassed = 0,
      totalFailed = 0;
    for (let row = 0; row < subjects.length; row++) {
      let passed = 0,
        failed = 0;
      const base = row * this.stride;
      for (let column = 0; column < columns; column++) {
        const value = values[row * columns + column];
        if (value > 2) throw new Error("Unknown binary score code.");
        passed += Number(value === 2);
        failed += Number(value === 1);
        if ((column + 1) % BLOCK === 0) {
          this.passed[base + (column + 1) / BLOCK] = passed;
          this.failed[base + (column + 1) / BLOCK] = failed;
        }
      }
      totalPassed += passed;
      totalFailed += failed;
    }
    if (
      totalPassed !== manifest.passed ||
      totalFailed !== manifest.failed ||
      totalPassed + totalFailed !== manifest.observations
    )
      throw new Error("Displayed scores differ from the source totals.");
  }

  private rawCounts(row: number, start: number, end: number): Counts {
    let passed = 0,
      failed = 0;
    const first = Math.ceil(start / BLOCK),
      last = Math.floor(end / BLOCK);
    if (first < last) {
      const base = row * this.stride;
      passed = this.passed[base + last] - this.passed[base + first];
      failed = this.failed[base + last] - this.failed[base + first];
    }
    const add = (a: number, b: number) => {
      for (let column = a; column < b; column++) {
        const value = this.values[row * this.manifest.columns + column];
        passed += Number(value === 2);
        failed += Number(value === 1);
      }
    };
    if (first < last) {
      add(start, first * BLOCK);
      add(last * BLOCK, end);
    } else add(start, end);
    return { passed, failed, missing: end - start - passed - failed };
  }

  counts(row: number, range: Interval, grader = ""): Counts {
    if (!grader) return this.rawCounts(row, range.start, range.end);
    let passed = 0,
      failed = 0;
    for (const group of this.manifest.groups) {
      const start = Math.max(range.start, group.start);
      const end = Math.min(range.end, group.start + group.count);
      if (start >= end || group.setups[group.rowSetups[row]]?.grader !== grader)
        continue;
      const result = this.rawCounts(row, start, end);
      passed += result.passed;
      failed += result.failed;
    }
    return {
      passed,
      failed,
      missing: range.end - range.start - passed - failed,
    };
  }

  value(row: number, column: number, grader = ""): number {
    const group = groupAt(this.manifest.groups, column);
    return grader && group.setups[group.rowSetups[row]]?.grader !== grader
      ? 0
      : this.values[row * this.manifest.columns + column];
  }
}

export function groupAt(groups: TestGroup[], column: number): TestGroup {
  const group = groups.find(
    (g) => column >= g.start && column < g.start + g.count,
  );
  if (!group) throw new Error("Test column is outside the matrix.");
  return group;
}

export const groupLabel = (group: TestGroup) =>
  group.id.replace(/\.[a-f\d]{7,}$/, "").replace("__", " / ");

export function fitRange(start: number, span: number, total: number): Interval {
  span = Math.min(total, Math.max(1, Math.round(span)));
  start = Math.min(total - span, Math.max(0, Math.round(start)));
  return { start, end: start + span };
}

export function zoomRange(
  range: Interval,
  factor: number,
  total: number,
  anchor = 0.5,
): Interval {
  const span = range.end - range.start;
  const next = Math.max(1, Math.round(span * factor));
  return fitRange(range.start + (span - next) * anchor, next, total);
}

/** Disjoint bins cover every visible test exactly once, even at subpixel scale. */
export function pixelBins(range: Interval, width: number): Interval[] {
  const span = range.end - range.start;
  const count = Math.min(span, Math.max(1, Math.floor(width)));
  return Array.from({ length: count }, (_, i) => ({
    start: range.start + Math.floor((i * span) / count),
    end: range.start + Math.floor(((i + 1) * span) / count),
  }));
}
