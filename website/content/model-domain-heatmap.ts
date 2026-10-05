// ---------------------------------------------------------------------------
//  Model × domain score heatmap. Each cell is a model's AVERAGE SCORE
//  (accuracy) over all items in all BINARY benchmarks tagged with a domain —
//  a micro (item-weighted) mean in [0,1]. Composed at build time from
//  chart-marginals.json (see chart-marginals.ts): a cell is Σsum/Σn over the
//  shelf's binary benchmarks listing that domain, so each section's heatmap
//  covers exactly that section's benchmarks and follows any shelf change
//  automatically. A benchmark counts toward EVERY domain it lists (same as
//  domain-readiness.ts), so a math+coding benchmark feeds both columns.
// ---------------------------------------------------------------------------

import { marginalCompany, marginalsFor } from "@/content/chart-marginals";
import {
  benchmarks,
  benchmarkCategories,
  type BenchmarkCategoryId,
} from "@/content/measurement-db";

// Minimum binary responses behind a cell before we publish its mean. A model
// measured on a handful of a domain's items yields a noisy accuracy that
// would read as signal; below this the cell stays empty rather than shown.
const MIN_CELL_N = 30;

const domainsBySlug = new Map(benchmarks.map((b) => [b.slug, b.categories]));

export type HeatmapColumn = {
  id: BenchmarkCategoryId;
  label: string;
  color: string;
  /** This column's index into every row's `cells` array. Cells stay in the
   *  canonical domain order; only the COLUMN display order is sorted, so each
   *  column keeps a pointer back to its cell slot. */
  index: number;
};

export type HeatmapCell = {
  /** Mean score (accuracy) in [0,1], or null when the model was never
   *  measured in this domain (below MIN_CELL_N). */
  mean: number | null;
  /** Binary responses behind the mean (0 when null). */
  n: number;
  /** Sequential light→dark Palo Alto background for the cell (null → gray). */
  color: string;
};

export type HeatmapRow = {
  model: string;
  /** Model name with the redundant company prefix dropped ("Meta Llama 3 8B" →
   *  "Llama 3 8B"), since the company logo already carries the brand. Kept
   *  whole when the remainder would be a bare size ("Mistral 7B" stays). */
  short: string;
  company: string;
  /** Total binary responses across all domains (coverage). */
  n: number;
  /** Breadth-aware overall used to tie-break the default row order: the mean
   *  of the model's per-domain scores with unmeasured domains imputed to the
   *  global mean cell, so missing coverage neither rewards nor punishes. */
  overall: number;
  /** One cell per column in `columns`, aligned by index. */
  cells: ReadonlyArray<HeatmapCell>;
};

export type ModelDomainHeatmapData = {
  columns: ReadonlyArray<HeatmapColumn>;
  rows: ReadonlyArray<HeatmapRow>;
};

/** Drop the company prefix from a model name so the row label reads cleanly
 *  under its logo. Mirrors flagshipLabel() in model-leaderboard.ts. */
function shortName(company: string, model: string): string {
  const prefix = `${company} `;
  if (model.startsWith(prefix)) {
    const rest = model.slice(prefix.length);
    if (rest && !/^\d/.test(rest)) return rest;
  }
  return model;
}

// Sequential ramp over [0,1]. Accuracy is a plain magnitude — 50% is not a
// meaningful midpoint — so one hue, light→dark, replaces the earlier
// red→fog→blue diverging ramp, which made the sparse post-filter grid read
// as alarm colors. The hue is Stanford Palo Alto, the same green that
// numbers the sections, stepped through a lightened midpoint so mid-range
// scores keep some chroma instead of washing gray.
const LOW = [237, 243, 241];
const MID = [111, 162, 151];
const HIGH = [7, 72, 63];
const GRAY = "rgb(238,238,238)";

function lerp(a: number[], b: number[], t: number): string {
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Light→dark Palo Alto for a score in [0,1] (dark = high); gray for null. */
export function cellColor(mean: number | null): string {
  if (mean === null) return GRAY;
  const t = Math.min(1, Math.max(0, mean));
  return t < 0.5 ? lerp(LOW, MID, t * 2) : lerp(MID, HIGH, (t - 0.5) * 2);
}

/** The model × domain heatmap for one shelf's benchmarks. */
export function makeModelDomainHeatmap(
  slugs: readonly string[],
): ModelDomainHeatmapData {
  // (model, domain) -> [sum, n] over the shelf's binary benchmarks.
  const cell = new Map<string, [number, number]>();
  const modelN = new Map<string, number>();
  for (const m of marginalsFor(slugs)) {
    if (m.sum === undefined || m.n === undefined) continue; // non-binary
    modelN.set(m.model, (modelN.get(m.model) ?? 0) + m.n);
    for (const d of domainsBySlug.get(m.slug) ?? []) {
      const key = `${m.model} ${d}`;
      const c = cell.get(key) ?? [0, 0];
      c[0] += m.sum;
      c[1] += m.n;
      cell.set(key, c);
    }
  }

  // Domain columns that actually cleared MIN_CELL_N for someone. `index` is
  // the cell-array slot (canonical category order); the display order puts
  // the highest-scoring domain on the left, mirroring the row ordering.
  const present = benchmarkCategories.filter((cat) =>
    [...modelN.keys()].some((m) => {
      const c = cell.get(`${m} ${cat.id}`);
      return c !== undefined && c[1] >= MIN_CELL_N;
    }),
  );
  const columnMean = (id: BenchmarkCategoryId): number => {
    let sum = 0;
    let count = 0;
    for (const m of modelN.keys()) {
      const c = cell.get(`${m} ${id}`);
      if (c !== undefined && c[1] >= MIN_CELL_N) {
        sum += c[0] / c[1];
        count += 1;
      }
    }
    return count ? sum / count : 0;
  };
  const columns: HeatmapColumn[] = present
    .map((cat, index) => ({
      id: cat.id,
      label: cat.label,
      color: cat.color,
      index,
    }))
    .sort((a, b) => columnMean(b.id) - columnMean(a.id));

  // Global mean cell — the neutral score an unmeasured domain is imputed to.
  let priorSum = 0;
  let priorN = 0;
  for (const [, [s, n]] of cell) {
    if (n >= MIN_CELL_N) {
      priorSum += s;
      priorN += n;
    }
  }
  const prior = priorN ? priorSum / priorN : 0.5;

  const rows: HeatmapRow[] = [];
  for (const [model, n] of modelN) {
    // Cells align to `present` (canonical order) — the slot each column's
    // `index` points into — NOT to the display-sorted `columns`.
    const cells: HeatmapCell[] = present.map((cat) => {
      const c = cell.get(`${model} ${cat.id}`);
      const mean =
        c !== undefined && c[1] >= MIN_CELL_N
          ? Math.round((c[0] / c[1]) * 10000) / 10000
          : null;
      return {
        mean,
        n: mean === null ? 0 : c![1],
        color: cellColor(mean),
      };
    });
    if (!cells.some((c) => c.mean !== null)) continue;
    const company = marginalCompany[model] ?? "Unknown";
    const overall =
      cells.reduce((acc, c) => acc + (c.mean ?? prior), 0) / cells.length;
    rows.push({
      model,
      short: shortName(company, model),
      company,
      n,
      overall,
      cells,
    });
  }

  // Broadest coverage first, best performance within a tie. Ordering purely
  // by `overall` floats rows with two or three measured cells to the top, so
  // the visible viewport is mostly empty; coverage-first packs the top of the
  // grid with the models that actually have data, and the sparse tail scrolls.
  rows.sort((a, b) => {
    const am = a.cells.filter((c) => c.mean !== null).length;
    const bm = b.cells.filter((c) => c.mean !== null).length;
    return bm - am || b.overall - a.overall;
  });

  return { columns, rows };
}

export const heatmapMeta = {
  title: "AI Subject by Domain Scores",
  description:
    "Mean binary score for each AI subject and domain. Multi-domain benchmarks contribute to every applicable column; hatching means no qualifying measurement.",
} as const;
