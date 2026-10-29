import { readJson } from "../benchmark-records/json";
import type { AlignedManifest, TestPage, TestRecord } from "./types";
import { ScoreMatrix, groupAt } from "./model";

const MAX_SCORE_BYTES = 12 * 1024 * 1024;

export async function loadMatrix(
  base: string,
  revision: string,
  signal: AbortSignal,
) {
  const response = await fetch(`${base}/manifest.json`, { signal });
  if (!response.ok)
    throw new Error(`Could not load test index (${response.status}).`);
  const manifest = (await response.json()) as AlignedManifest;
  if (
    manifest.format !== "aligned-test-preview-v1" ||
    manifest.sourceRevision !== revision ||
    manifest.columns < 1 ||
    manifest.scores.expandedBytes !==
      manifest.columns * manifest.subjects.length ||
    manifest.scores.expandedBytes > MAX_SCORE_BYTES ||
    manifest.scores.file !== "scores.bin.gz"
  )
    throw new Error("Invalid or mismatched test preview index.");
  let offset = 0;
  for (const group of manifest.groups) {
    if (
      group.start !== offset ||
      group.count < 1 ||
      group.rowSetups.length !== manifest.subjects.length ||
      group.rowSetups.some((i) => i < -1 || i >= group.setups.length)
    )
      throw new Error("Invalid testlet membership.");
    offset += group.count;
  }
  if (offset !== manifest.columns) throw new Error("Incomplete testlet index.");
  const scores = await fetch(`${base}/${manifest.scores.file}`, { signal });
  if (!scores.ok) throw new Error(`Could not load scores (${scores.status}).`);
  const compressed = await scores.arrayBuffer();
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", compressed)),
    (value) => value.toString(16).padStart(2, "0"),
  ).join("");
  if (
    compressed.byteLength !== manifest.scores.bytes ||
    hash !== manifest.scores.sha256
  )
    throw new Error("Score integrity check failed.");
  const raw = await new Response(
    new Blob([compressed])
      .stream()
      .pipeThrough(new DecompressionStream("gzip")),
  ).arrayBuffer();
  signal.throwIfAborted();
  return new ScoreMatrix(manifest, new Uint8Array(raw));
}

/** Small native-test pages only. Original items and answers use RecordSource. */
export class TestSource {
  private readonly pages = new Map<string, TestPage>();
  constructor(
    readonly base: string,
    readonly manifest: AlignedManifest,
  ) {}

  async test(column: number, signal: AbortSignal): Promise<TestRecord> {
    const group = groupAt(this.manifest.groups, column);
    const local = column - group.start;
    const page = Math.floor(local / this.manifest.pageSize);
    const path = `tests/${this.manifest.groups.indexOf(group)}/${page}.json`;
    let records = this.pages.get(path);
    if (!records) {
      records = await readJson<TestPage>(
        this.base,
        path,
        this.manifest,
        signal,
      );
      signal.throwIfAborted();
      this.pages.set(path, records);
      if (this.pages.size > 4)
        this.pages.delete(this.pages.keys().next().value!);
    }
    const test = records.tests[local - records.start];
    if (!test || test.items.length !== group.setups.length)
      throw new Error("Missing native test or setup mapping.");
    return test;
  }
}
