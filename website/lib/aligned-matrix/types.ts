import type { Subject } from "../benchmark-records/types";

export type Setup = {
  grader: string;
  contentHash: string;
  promptHash: string;
  image: string;
  cwd: string;
};
export type TestGroup = {
  id: string;
  start: number;
  count: number;
  setups: Setup[];
  rowSetups: number[];
};
export type AlignedManifest = {
  format: "aligned-test-preview-v1";
  benchmark: string;
  sourceRevision: string;
  columns: number;
  canonicalItems: number;
  observations: number;
  passed: number;
  failed: number;
  subjects: Subject[];
  groups: TestGroup[];
  pageSize: number;
  pack: { index: string };
  scores: {
    file: string;
    bytes: number;
    expandedBytes: number;
    sha256: string;
  };
};
export type TestRecord = { name: string; items: (string | null)[] };
export type TestPage = { start: number; tests: TestRecord[] };
export type Interval = { start: number; end: number };
export type SelectedCell = { row: number; column: number };
export type Counts = { passed: number; failed: number; missing: number };
