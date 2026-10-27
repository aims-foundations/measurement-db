import type {
  AssetLink,
  GalleryRecords,
  Manifest,
  TableName,
  TableRows,
  Subject,
} from "./types";
import { readParquetRecord } from "./parquet";
import { readJson } from "./json";

type Fetch = typeof fetch;
type PageIndex = Record<string, number>;

/** HTTP and a bounded cache only. This class has no React or layout decisions. */
export class RecordSource {
  private readonly cache = new Map<string, unknown>();

  constructor(
    readonly base: string,
    readonly manifest: Manifest,
    private readonly request: Fetch = (...args) => fetch(...args),
  ) {}

  private async json<T>(path: string, signal: AbortSignal): Promise<T> {
    signal.throwIfAborted();
    if (this.cache.has(path)) return this.cache.get(path) as T;
    const value = await readJson<T>(
      this.base,
      path,
      this.manifest,
      signal,
      this.request,
    );
    signal.throwIfAborted();
    // Cache completed requests only: aborting one selection cannot poison another.
    if (!path.startsWith("windows/")) this.cache.set(path, value);
    if (this.cache.size > 8) this.cache.delete(this.cache.keys().next().value!);
    return value;
  }

  private index(table: TableName, signal: AbortSignal): Promise<PageIndex> {
    return this.manifest.tables[table]
      ? this.json(`${table}/index.json`, signal)
      : Promise.resolve({});
  }

  private async all<T extends TableName>(
    table: T,
    signal: AbortSignal,
    selectedPages?: number[],
    include?: (record: TableRows[T]) => boolean,
  ): Promise<TableRows[T][]> {
    const result: TableRows[T][] = [];
    const pages =
      selectedPages ??
      Array.from(
        { length: this.manifest.tables[table]?.pages ?? 0 },
        (_, i) => i,
      );
    // A small concurrency bound avoids a request burst for large benchmarks.
    for (let start = 0; start < pages.length; start += 4) {
      const batch = await Promise.all(
        pages
          .slice(start, start + 4)
          .map((page) =>
            this.json<Record<string, TableRows[T]>>(
              `${table}/${page}.json`,
              signal,
            ),
          ),
      );
      for (const page of batch)
        for (const record of Object.values(page))
          if (!include || include(record)) result.push(record);
    }
    return result;
  }

  subjects(signal: AbortSignal): Promise<Subject[]> {
    return this.all("subjects", signal);
  }

  async window(page: number, signal: AbortSignal): Promise<GalleryRecords> {
    const [subjects, window] = await Promise.all([
      this.subjects(signal),
      this.json<Omit<GalleryRecords, "subjects">>(
        `windows/${page}.json`,
        signal,
      ),
    ]);
    return { ...window, subjects };
  }

  async windowForItem(
    id: string,
    signal: AbortSignal,
  ): Promise<[number, number] | null> {
    const hash = new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(id)),
    );
    const index = await this.json<Record<string, [number, number]>>(
      `windows/index-${hash[0].toString(16).padStart(2, "0")}.json`,
      signal,
    );
    return Object.hasOwn(index, id) ? index[id] : null;
  }

  async load(
    signal: AbortSignal,
    subjectIds?: string[],
  ): Promise<GalleryRecords> {
    const descriptor = this.manifest.tables.responses;
    const bySubject =
      subjectIds && descriptor?.bySubject
        ? await this.json<Record<string, number[]>>(
            `responses/${descriptor.bySubject}`,
            signal,
          )
        : undefined;
    const pages =
      bySubject &&
      [...new Set(subjectIds!.flatMap((id) => bySubject[id] ?? []))].sort(
        (a, b) => a - b,
      );
    const selected = subjectIds && new Set(subjectIds);
    const [subjects, items, observations] = await Promise.all([
      this.all("subjects", signal),
      this.index("items", signal),
      this.all(
        "responses",
        signal,
        pages,
        (record) => !selected || selected.has(record.subject_id),
      ),
    ]);
    return {
      ...(selected ? { allSubjects: subjects } : {}),
      subjects: selected
        ? subjects.filter((s) => selected.has(s.subject_id))
        : subjects,
      itemIds: Object.keys(items),
      observations,
    };
  }

  async get<T extends TableName>(
    table: T,
    id: string,
    signal: AbortSignal,
  ): Promise<TableRows[T] | null> {
    const parquet = this.manifest.tables[table]?.parquet;
    if (parquet)
      return readParquetRecord<TableRows[T]>(this.base, parquet, id, signal, {
        table,
        key: this.manifest.tables[table]!.key,
        ...this.manifest,
      });
    const index = await this.index(table, signal);
    if (!Object.hasOwn(index, id)) return null;
    const page = await this.json<Record<string, TableRows[T]>>(
      `${table}/${index[id]}.json`,
      signal,
    );
    if (!Object.hasOwn(page, id))
      throw new Error(`Missing ${table} record ${id}.`);
    return page[id];
  }

  async asset(
    link: AssetLink,
    signal: AbortSignal,
  ): Promise<{ url: string; blob: Blob | null }> {
    const asset = await this.get("assets", link.asset_id, signal);
    if (!asset) throw new Error(`Missing attachment ${link.path}.`);
    if (asset.data instanceof Uint8Array) {
      return {
        url: "",
        blob: new Blob([new Uint8Array(asset.data)], { type: link.media_type }),
      };
    }
    if (!asset.data_url)
      throw new Error(`Missing attachment bytes for ${link.path}.`);
    const url = `${this.base}/assets/${encodeURIComponent(asset.data_url)}`;
    if (!link.media_type.startsWith("image/")) return { url, blob: null };
    const response = await this.request(url, { signal });
    if (!response.ok)
      throw new Error(`Could not load attachment (${response.status}).`);
    return {
      url,
      blob: new Blob([await response.arrayBuffer()], {
        type: link.media_type,
      }),
    };
  }
}
