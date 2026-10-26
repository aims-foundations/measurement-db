import {
  asyncBufferFromUrl,
  parquetMetadata,
  parquetReadObjects,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";

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
}: MessageEvent<{
  base: string;
  parquet: { url: string; bytes: number };
  id: string;
}>) => {
  try {
    const { base, parquet, id } = data;
    const hash = new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(id)),
    );
    const bucket = hash[0].toString(16).padStart(2, "0");
    const positions = await json<Record<string, [number, number]>>(
      `${base}/index-${bucket}.json.gz`,
    );
    if (!Object.hasOwn(positions, id)) {
      self.postMessage({ value: null });
      return;
    }
    const [group, row] = positions[id];
    const index = await json<PageIndex>(`${base}/group-${group}.json.gz`);
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
      file,
      metadata,
      compressors,
      utf8: false,
      rowStart: row,
      rowEnd: row + 1,
      pageLocationsByGroup,
    };
    const rows = await parquetReadObjects(options);
    if (rows.length !== 1 || rows[0].response_id !== id)
      throw new Error("Source row does not match the requested response ID.");
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
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
