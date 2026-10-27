import {
  asyncBufferFromUrl,
  parquetMetadata,
  parquetReadObjects,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { readJson, type JsonTransport } from "./json";
import { budgetSource } from "./source-budget";
import {
  readPreviewRecord,
  PagedRecordError,
  type RecordPreview,
} from "./record-preview";

type PageIndex = {
  footer: string;
  pages: Record<string, [number, number, number][]>;
};
async function json<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Could not load source index (${response.status}).`);
  return response.json();
}

/** Return the exact source row. Never extract, truncate, or re-label its answer. */
self.onmessage = async ({
  data,
}: MessageEvent<
  {
    base: string;
    parquet: { url: string; bytes: number };
    id: string;
    table?: string;
    key?: string;
    preview?: RecordPreview;
    previewBase?: string;
  } & JsonTransport
>) => {
  try {
    const { base, parquet, id, table, key = "response_id" } = data;
    if (data.preview && data.previewBase) {
      const value = await readPreviewRecord(
        data.previewBase,
        data.preview,
        id,
        key,
      );
      if (value) {
        self.postMessage({ value });
        return;
      }
    }
    const indexJson = <T>(path: string) =>
      table
        ? readJson<T>(base, `${table}/${path}`, {
            compressed: true,
            pack: data.pack,
          })
        : json<T>(`${base}/${path}.gz`);
    const hash = new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(id)),
    );
    const bucket = hash[0].toString(16).padStart(2, "0");
    const positions = await indexJson<Record<string, [number, number]>>(
      `index-${bucket}.json`,
    );
    if (!Object.hasOwn(positions, id)) {
      self.postMessage({ value: null });
      return;
    }
    const [group, row] = positions[id];
    const index = await indexJson<PageIndex>(`group-${group}.json`);
    const file = await asyncBufferFromUrl({
      url: parquet.url,
      byteLength: parquet.bytes,
    });
    const metadata = parquetMetadata(
      Uint8Array.from(atob(index.footer), (c) => c.charCodeAt(0)).buffer,
    );
    const pageLocationsByGroup = [
      Object.fromEntries(
        Object.entries(index.pages).map(([name, pages]) => [
          name,
          pages.map(([offset, size, firstRow]) => ({
            offset: BigInt(offset),
            compressed_page_size: size,
            first_row_index: BigInt(firstRow),
          })),
        ]),
      ),
    ];
    const options = {
      ...budgetSource(file, compressors),
      metadata,
      utf8: false,
      rowStart: row,
      rowEnd: row + 1,
      pageLocationsByGroup,
    };
    const rows = await parquetReadObjects(options);
    if (rows.length !== 1 || rows[0][key] !== id)
      throw new Error("Source row does not match the requested record ID.");
    const value = Object.fromEntries(
      Object.entries(rows[0]).map(([key, value]) => {
        if (typeof value !== "bigint") return [key, value];
        const number = Number(value);
        if (!Number.isSafeInteger(number))
          throw new Error(
            `Source integer ${key} cannot be represented exactly.`,
          );
        return [key, number];
      }),
    );
    self.postMessage({ value });
  } catch (error) {
    self.postMessage({
      ...(error instanceof PagedRecordError ? { paged: error.record } : {}),
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
