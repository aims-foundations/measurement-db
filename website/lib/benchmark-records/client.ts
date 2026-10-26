import type {
  AssetLink,
  GalleryRecords,
  Manifest,
  TableName,
  TableRows,
} from "./types";

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
    const response = await this.request(`${this.base}/${path}`, { signal });
    if (!response.ok)
      throw new Error(`Could not load benchmark data (${response.status}).`);
    const value = (await response.json()) as T;
    signal.throwIfAborted();
    // Cache completed requests only: aborting one selection cannot poison another.
    this.cache.set(path, value);
    if (this.cache.size > 32)
      this.cache.delete(this.cache.keys().next().value!);
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
  ): Promise<TableRows[T][]> {
    const result: TableRows[T][] = [];
    const pages = this.manifest.tables[table]?.pages ?? 0;
    // A small concurrency bound avoids a request burst for large benchmarks.
    for (let start = 0; start < pages; start += 4) {
      const batch = await Promise.all(
        Array.from({ length: Math.min(4, pages - start) }, (_, i) =>
          this.json<Record<string, TableRows[T]>>(
            `${table}/${start + i}.json`,
            signal,
          ),
        ),
      );
      for (const page of batch) result.push(...Object.values(page));
    }
    return result;
  }

  async load(signal: AbortSignal): Promise<GalleryRecords> {
    const [subjects, items, observations] = await Promise.all([
      this.all("subjects", signal),
      this.index("items", signal),
      this.all("responses", signal),
    ]);
    return { subjects, itemIds: Object.keys(items), observations };
  }

  async get<T extends TableName>(
    table: T,
    id: string,
    signal: AbortSignal,
  ): Promise<TableRows[T] | null> {
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
  ): Promise<{ url: string; image: Blob | null }> {
    const asset = await this.get("assets", link.asset_id, signal);
    if (!asset) throw new Error(`Missing attachment ${link.path}.`);
    const url = `${this.base}/assets/${encodeURIComponent(asset.data_url)}`;
    if (!link.media_type.startsWith("image/")) return { url, image: null };
    const response = await this.request(url, { signal });
    if (!response.ok)
      throw new Error(`Could not load attachment (${response.status}).`);
    return {
      url,
      image: new Blob([await response.arrayBuffer()], {
        type: link.media_type,
      }),
    };
  }
}
