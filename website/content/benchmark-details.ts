// ---------------------------------------------------------------------------
//  Per-benchmark detail data for the measurement-db gallery detail pages.
//
//  The JSON is generated, not hand-edited: it is extracted from the published
//  parquet files in github.com/aims-foundations/measurement-db by
//  `scripts/render_website/generate_benchmark_gallery.py` (which also writes the bare
//  response-matrix images under public/benchmarks/matrices/). Re-run
//  `generate_benchmark_gallery.py details --all` to refresh both. Stats here reflect the
//  actual published matrices.
// ---------------------------------------------------------------------------

import detailsJson from "@/content/generated/benchmark-details.json";

export type BenchmarkSubjectScore = {
  name: string;
  /** Mean response across the subject's items (accuracy / rate / rating);
   *  null when the benchmark has no per-item responses (granularity
   *  "aggregate"/"not_released"). */
  score: number | null;
};

/** A nominal response category (e.g. XSTest's full/partial refusal), with the
 *  exact color used to draw it in the matrix. */
export type MatrixCategory = {
  value: number;
  label: string;
  color: string;
};

/** One condition's binary matrix (e.g. a specific attack × category × judge).
 *  Each condition is sorted INDEPENDENTLY — its rows by its own row mean, its
 *  columns by its own column mean — over that slice's observed responses only.
 *  So conditions differ in axis ORDER and SIZE, not just color: a subject or item
 *  the slice never observed is dropped from its axis. */
export type MatrixCondition = {
  /** Selected value for each dimension, e.g. { attack, category, judge }. */
  sel: Record<string, string>;
  /** Path to this condition's binary matrix PNG. */
  matrix: string;
  matrixSize: [number, number];
  /** This condition's row axis, as indices into the parent's `matrixRows` /
   *  `matrixRowIds` (which are the union over all conditions). Indices, not ids,
   *  keep the page payload small across benchmarks with hundreds of slices.
   *  Null for legacy entries built before per-condition axes, where the parent's
   *  row axis applies as-is. */
  rowIdx?: number[] | null;
  /** Per-row subject rate for this condition, parallel to `rowIdx`. */
  rowScores: (number | null)[] | null;
  /** This condition's column count. The column axis itself (indices into the
   *  lazy file's `colIds`, plus per-column rates) is large, so it ships in
   *  `/benchmarks/data/<slug>.json.gz` under `slices` rather than here. Null for
   *  legacy entries, where the parent's column count applies. */
  nItems?: number | null;
  colP: (number | null)[] | null;
  observed: number;
};

/** Present when a benchmark stores multiple binary results per (subject, item)
 *  cell (trials / attack methods / judges). Each condition gets its own binary
 *  matrix, selected via one dropdown per dimension. */
export type BenchmarkConditions = {
  /** Ordered dimension names → one dropdown each (e.g. ["attack","category","judge"]). */
  dims: string[];
  /** Available values per dimension. Includes the literal "n/a" for a dimension
   *  only some conditions carry — that is a real, selectable value, since the
   *  matrices for those conditions are keyed with it. */
  values: Record<string, string[]>;
  /** Default selection shown first. Names EVERY dim in `dims`. */
  default: Record<string, string>;
  /** Set to the merged dimension's name when the benchmark's test_condition
   *  strings shared no universal dim and were collapsed into one dimension
   *  holding the raw condition string; null for the normal cross-product case.
   *  See generate_benchmark_gallery.py's `collapse`. */
  collapsed?: string | null;
  /** All conditions, keyed by "dim=val;dim=val;…" in `dims` order. */
  matrices: Record<string, MatrixCondition>;
};

export type BenchmarkDetail = {
  slug: string;
  name: string;
  description: string | null;
  domain: string[];
  modality: string[];
  license: string | null;
  responseType: string | null;
  responseScale: string | null;
  sourceUrl: string | null;
  paperUrl: string | null;
  releaseDate: string | null;
  stats: {
    items: number;
    subjects: number;
    /** Fraction of (subject × item) cells that were actually evaluated. */
    observed: number;
    /** Mean response across all observed cells; null when the benchmark has
     *  no per-item responses (nothing observed). */
    meanResponse: number | null;
  };
  /** Path to the bare response-matrix heatmap under /public (rows = subjects,
   *  columns = items). */
  matrix: string;
  /** [width, height] of the matrix PNG in pixels. */
  matrixSize: [number, number];
  /** Present when the benchmark has multiple binary conditions per cell; the
   *  viewer then shows one binary matrix per condition with a dropdown per
   *  dimension. Absent/null for single-matrix benchmarks. */
  conditions?: BenchmarkConditions | null;
  /** Present when the response is a nominal category (not binary / not an ordinal
   *  score), e.g. XSTest. Drives a discrete categorical legend + cell labels. */
  categories?: MatrixCategory[] | null;
  /** Custom labels for a binary benchmark whose 1/0 isn't "correct/incorrect"
   *  (e.g. Do-Not-Answer: 1 = Harmful, 0 = Safe). */
  binaryLabels?: { zero: string; one: string } | null;
  /** One line about the released data itself, shown above the matrix. For a
   *  benchmark whose provider publishes only part of its bank, the matrix
   *  width otherwise reads as the whole thing. */
  note?: string | null;
  /** Number of distinct adversarial attack methods (safety benchmarks);
   *  null when the benchmark has no attack dimension. */
  attacks?: number | null;
  /** Total item-columns available for this benchmark. */
  matrixItemsTotal: number;
  /** Item-columns actually drawn (subsampled when the bank is item-rich, so
   *  cells stay near-square). */
  matrixItemsShown: number;
  /** True when matrixItemsShown < matrixItemsTotal. */
  matrixSampled: boolean;
  /** Subject display names in matrix row order (top→bottom, weakest→strongest)
   *  — used to map a hovered cell to its subject. */
  matrixRows: string[];
  /** Subject ids in matrix row order, parallel to matrixRows — used to fetch
   *  the clicked subject's answer shard (traces/<slug>/<id>.json.gz). */
  matrixRowIds: string[];
  /** Per-row subject overall score (accuracy / rate), parallel to matrixRows. */
  matrixRowScores: (number | null)[];
  /** Item ids in matrix column order. generate_benchmark_gallery.py sorts by ascending solve
   *  rate, so this is left→right HARDEST→easiest (an earlier version of this
   *  comment had the direction backwards). Null for
   *  very item-rich benchmarks where the arrays would bloat the payload. */
  matrixColIds: string[] | null;
  /** Per-column item difficulty (solve rate), parallel to matrixColIds. */
  matrixColP: (number | null)[] | null;
  /** Whether per-(model, item) answer traces are published — when true, the
   *  matrix viewer offers the clicked model's answer, lazily fetched from the
   *  per-subject shards under public/benchmarks/traces/<slug>/. */
  hasTraces: boolean;
  /** Whether this benchmark ships per-cell call recordings under
   *  public/benchmarks/audio/<slug>/. Clips are named
   *  <subject_id>/<sanitize(item_id__slice_key)>.opus, the same key the viewer
   *  builds on click, so no extra index is needed. Not every cell has one
   *  (tau_voice's text baselines do not), so the player hides on a 404.
   *  Optional for back-compat with entries generated before audio existed. */
  hasAudio?: boolean;
  audio?: {
    path: string;
    subjects: Record<string, string>;
    items: Record<string, string>;
  };
  /** Whether joined/<slug>.json.gz exists — the per-cell payload behind the
   *  joined matrix (every response at once, no condition to select). Optional
   *  for back-compat with entries generated before it existed. */
  hasJoined?: boolean;
  /** How many leading hex chars of item_id the answer shards are bucketed by
   *  (0 = one flat shard per subject). Item-rich benchmarks use >0 so a click
   *  fetches a small chunk instead of a model's whole answer set. Optional for
   *  back-compat with entries generated before chunking existed. */
  traceChunkPrefix?: number;
  /** True when every response is exactly 0 or 1 (correctness is meaningful). */
  isBinary: boolean;
  /** [min, max] of the response scale as observed in the data. */
  valueRange: [number, number];
  /** Human-readable description of the response scale (e.g. "1 = correct ·
   *  0 = incorrect", or "GPT-4 judge rating, 1 (worst) to 10 (best)"). */
  scaleLabel: string;
  /** False when the upstream benchmark is gated and its question text can't be
   *  redistributed here (e.g. HLE); the clicked cell then shows no prompt. */
  questionsAvailable: boolean;
  subjects: BenchmarkSubjectScore[];
};

const details = detailsJson as unknown as Record<string, BenchmarkDetail>;

export function getBenchmarkDetail(slug: string): BenchmarkDetail | undefined {
  return details[slug];
}
