/** The canonical table fields used by the viewer. Extra source fields stay intact. */
import type { RecordPreview } from "./record-preview";
export type SourceRecord = Record<string, unknown>;

export type Subject = SourceRecord & {
  subject_id: string;
  display_name: string;
};

export type Item = SourceRecord & {
  item_id: string;
  testlet_id?: string | null;
  content: string | null;
  grading_criterion: string;
  asset_manifest: string | null;
};

export type Observation = SourceRecord & {
  response_id: string;
  attempt_id?: string | null;
  subject_id: string;
  item_id: string;
  trial: number;
  test_condition: string | null;
  interactors: string | null;
  response: number | null;
};

export type Trace = SourceRecord & { response_id: string; trace: string };
export type Attempt = SourceRecord & {
  attempt_id: string;
  subject_id: string;
  raw_attempt_id: string;
  trace: string | null;
};
export type Asset = SourceRecord & {
  asset_id: string;
  data_url?: string;
  data?: Uint8Array;
};
export type AssetLink = {
  asset_id: string;
  path: string;
  media_type: string;
  role: string;
};

export type TableRows = {
  subjects: Subject;
  items: Item;
  responses: Observation;
  traces: Trace;
  attempts: Attempt;
  assets: Asset;
};
export type TableName = keyof TableRows;
export type TableDescriptor = {
  key: string;
  rows: number;
  pages: number;
  bySubject?: string;
  parquet?: { url: string; bytes: number; download?: string };
};

export type BenchmarkMetadata = SourceRecord & {
  benchmark_id: string;
  name: string;
  description: string;
  response_type: string;
  response_scale: string;
  domain: string[];
  modality: string[];
  license: string;
  granularity: string;
};

export type Manifest = {
  testlets?: string;
  previews?: Partial<Record<TableName, RecordPreview>>;
  compressed?: boolean;
  pack?: { index: string };
  windows?: {
    testlets?: string;
    pages: number;
    rows: number;
    maxRows: number;
    maxItems: number;
    maxBytes: number;
  };
  benchmark: BenchmarkMetadata;
  tables: Partial<Record<TableName, TableDescriptor>>;
};
export type GalleryEntry = { base: string; manifest: Manifest };
export type GalleryRecords = {
  subjects: Subject[];
  allSubjects?: Subject[];
  itemIds: string[];
  observations: Observation[];
  partialItemIds?: string[];
  testlets?: TestletWindow[];
};

export type TestletWindow = {
  testletId: string;
  itemIds: string[];
  totalItems: number;
  continuedBefore: boolean;
  continuedAfter: boolean;
};
export type TestletIndex = Record<
  string,
  { pages: [number, number]; items: number }
>;

export type Direction = "higher_is_better" | "lower_is_better" | "unordered";
export type ConcreteScale =
  | {
      kind: "discrete";
      values: number[];
      meanings?: Record<string, string>;
      direction?: Direction;
    }
  | {
      kind: "interval";
      min: number | null;
      max: number | null;
      direction?: Direction;
    };
export type ResponseScale = ConcreteScale | { kind: "mixed" };
export type GradingCriterion = {
  reference_answer: string | null;
  rule: string | null;
  response_scale?: ConcreteScale;
};
