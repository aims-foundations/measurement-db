// ---------------------------------------------------------------------------
//  Per-benchmark Rasch (1PL) IRT fit.
//
//  The JSON is generated, not hand-edited, by
//  `scripts/analyze_measurements/fit_rasch_models.py --all --emit-web`, which fits
//
//      p(correct) = sigmoid( theta[subject] - z[item] + c[test_condition] )
//
//  on an 80/20 split over observations. The condition term is dropped when a
//  benchmark has a single test_condition, so those are the plain
//  sigmoid(theta - z).
//
//  ONLY benchmarks whose responses are natively 0/1 appear here. Graded ones
//  (arena_hard, writingbench, mtbench, ...) are absent on purpose — thresholding
//  them would invent a dichotomy the benchmark never defined — so every consumer
//  must treat a missing slug as "no IRT panel", not as an error.
// ---------------------------------------------------------------------------

import irtJson from "@/content/generated/benchmark-irt.json";

export type BenchmarkIrt = {
  /** The fitted link, e.g. "sigmoid(theta - z)" — shown verbatim on the page. */
  model: string;
  /** Area under the ROC curve on the fitted rows and on the held-out rows.
   *  Null only in the degenerate case where a split has a single class. */
  aucTrain: number | null;
  aucTest: number | null;
  /** Total log-likelihood (not per-observation) on each split. */
  llTrain: number | null;
  llTest: number | null;
  /** Subjects (items) whose responses are ALL right or ALL wrong. Their
   *  maximum-likelihood theta (z) is unbounded, so the fit only pushes it out
   *  until the logit saturates — those values are "off the scale", not
   *  measurements. Optional: absent from payloads written before this was
   *  recorded. */
  separatedSubjects?: number;
  separatedItems?: number;
  /** Share of HELD-OUT responses the thresholded prediction gets right. */
  accTest: number;
  nTrain: number;
  nTest: number;
  nSubjects: number;
  nItems: number;
  /** Distinct test_condition values; 1 means no condition term was fitted. */
  nConditions: number;
  /** Mean response over every fitted row — the base rate the AUC sits above. */
  meanResponse: number;
  iterations: number;
  /** False when the fit stopped at --max-iter instead of settling. */
  converged: boolean;
  seconds: number;
  /** subject_id -> ability. Keyed by id (not name) because that is how the
   *  matrix identifies its rows (`matrixRowIds`), and a slice's row axis is a
   *  subset in a different order. Centered at 0 across subjects. */
  theta: Record<string, number>;
  /** Fitted difficulty by source item ID; independent of chart order. */
  zByItem: Record<string, number>;
};

const irt = irtJson as unknown as Record<string, BenchmarkIrt>;

export function getBenchmarkIrt(slug: string): BenchmarkIrt | undefined {
  return irt[slug];
}
