export type ScaleType = "Binary" | "Graded" | "Categorical";

export type BenchmarkSearchHit = {
  slug: string;
  name: string;
  description: string;
  categories: string[];
  license: string | null;
  releaseYear: number | null;
  items: number;
  models: number;
  resultCount: number;
  itemResponses: boolean;
  scaleType: ScaleType;
};

export type ModelResultSearchHit = {
  id: string;
  benchmarkSlug: string;
  benchmarkName: string;
  modelName: string;
  score: number | null;
  scaleLabel: string;
  scaleType: ScaleType;
  isBinary: boolean;
  categories: string[];
  releaseYear: number | null;
};

/** The compact catalog bundled with the page for graceful offline fallback. */
export type LocalBenchmarkSearchDocument = BenchmarkSearchHit & {
  searchText: string;
};
