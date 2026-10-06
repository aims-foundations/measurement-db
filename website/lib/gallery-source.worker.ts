import {
  asyncBufferFromUrl,
  parquetMetadata,
  parquetReadObjects,
} from "hyparquet";
import { compressors } from "hyparquet-compressors";

type RowPosition = [number, number];
type Source = {
  dynamicItems: boolean;
  tables: Record<string, { size: number; columns: string[] }>;
};
type PageIndex = {
  footer: string;
  pages: Record<string, [number, number, number][]>;
};
type Item = { content: string | null; answer: string | null };
const INLINE_IMAGE = /data:image\/[\w.+-]+;base64,[A-Za-z0-9+/=]+/g;
const TRACE_MAX_CHARS = 50000;

async function json<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Could not load gallery data (${response.status}).`);
  return response.json();
}

async function lookup<T>(
  base: string,
  kind: string,
  key: unknown,
): Promise<T | undefined> {
  const encoded = JSON.stringify(key);
  const hash = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(encoded),
  );
  const bucket = new Uint8Array(hash)[0].toString(16).padStart(2, "0");
  const response = await fetch(`${base}/${kind}/${bucket}.json.gz`);
  if (response.status === 404) return undefined;
  if (!response.ok)
    throw new Error(`Could not load ${kind} data (${response.status}).`);
  return ((await response.json()) as Record<string, T>)[encoded];
}

async function readRow(
  base: string,
  source: Source,
  table: string,
  position: RowPosition,
) {
  const index = await lookup<PageIndex>(
    base,
    `${table}-pages`,
    String(position[0]),
  );
  if (!index) throw new Error(`Missing ${table} page index.`);
  const info = source.tables[table];
  const slug = base.split("/").at(-1)!;
  const prefix = base.slice(0, base.indexOf("/benchmark-data/"));
  const file = await asyncBufferFromUrl({
    url: `${prefix}/api/gallery-source/${encodeURIComponent(slug)}/${table}`,
    byteLength: info.size,
  });
  const metadata = parquetMetadata(
    Uint8Array.from(atob(index.footer), (c) => c.charCodeAt(0)).buffer,
  );
  const pageLocationsByGroup = [
    Object.fromEntries(
      Object.entries(index.pages).map(([name, pages]) => [
        name,
        pages.map(([offset, size, row]) => ({
          offset: BigInt(offset),
          compressed_page_size: size,
          first_row_index: BigInt(row),
        })),
      ]),
    ),
  ];
  const options = {
    file,
    metadata,
    compressors,
    utf8: false,
    columns: info.columns,
    rowStart: position[1],
    rowEnd: position[1] + 1,
    pageLocationsByGroup,
  };
  const rows = await parquetReadObjects(options);
  if (rows.length !== 1)
    throw new Error(`Could not locate the requested ${table} row.`);
  return rows[0];
}

function itemContent(row: Record<string, unknown>): Item {
  let answer =
    "reference_answer" in row ? row.reference_answer : row.correct_answer;
  if (row.grading_criterion != null)
    answer = JSON.parse(String(row.grading_criterion)).reference_answer;
  return {
    content: row.content == null ? null : String(row.content),
    answer: answer == null ? null : String(answer),
  };
}

async function sourceItem(base: string, source: Source, id: string) {
  const position = await lookup<RowPosition>(base, "item-rows", id);
  if (!position) throw new Error("Could not locate the requested item.");
  return readRow(base, source, "items", position);
}

async function loadItem(
  base: string,
  source: Source,
  id: string,
): Promise<Item | undefined> {
  if (!source.dynamicItems) return lookup<Item>(base, "item", id);
  const row = await sourceItem(base, source, id);
  const item = itemContent(row);
  const path = base.slice(base.indexOf("/benchmark-data/"));
  for (const field of ["content", "answer"] as const) {
    let index = 0;
    item[field] =
      item[field]?.replace(
        INLINE_IMAGE,
        () =>
          `${path}/image/item/${encodeURIComponent(id)}/${field}/${index++}`,
      ) ?? null;
  }
  const links = JSON.parse(String(row.asset_manifest || "[]")) as {
    role: string;
    media_type: string;
    asset_id: string;
    ordinal: number;
  }[];
  for (const link of links) {
    if (
      ["input", "input_image"].includes(link.role) &&
      link.media_type.startsWith("image/")
    )
      item.content =
        (item.content ?? "") +
        `\n\n![Image ${link.ordinal}](${path}/image/asset/${encodeURIComponent(link.asset_id)}?type=${encodeURIComponent(link.media_type)})`;
  }
  return item;
}

async function loadImage(base: string, source: Source, key: string) {
  const url = new URL(key, self.location.origin);
  const parts = url.pathname
    .split("/image/")[1]
    .split("/")
    .map(decodeURIComponent);
  if (parts[0] === "asset") {
    const position = await lookup<RowPosition>(base, "asset-rows", parts[1]);
    if (!position) throw new Error("Could not locate the requested image.");
    const row = await readRow(base, source, "assets", position);
    if (!(row.data instanceof Uint8Array))
      throw new Error("Invalid image bytes.");
    return {
      data: row.data,
      type: url.searchParams.get("type") ?? "application/octet-stream",
    };
  }
  const row = await sourceItem(base, source, parts[1]);
  const item = itemContent(row);
  const value = item[parts[2] as keyof Item];
  const encoded = value?.match(INLINE_IMAGE)?.[Number(parts[3])];
  if (!encoded) throw new Error("Could not locate the embedded image.");
  return {
    data: Uint8Array.from(atob(encoded.slice(encoded.indexOf(",") + 1)), (c) =>
      c.charCodeAt(0),
    ),
    type: encoded.slice(5, encoded.indexOf(";")),
  };
}

self.onmessage = async ({
  data,
}: MessageEvent<{ base: string; kind: string; key: unknown }>) => {
  try {
    const { base, kind, key } = data;
    const source = await json<Source>(`${base}/source.json.gz`);
    let value: unknown;
    if (kind === "item") value = await loadItem(base, source, String(key));
    else if (kind === "image")
      value = await loadImage(base, source, String(key));
    else {
      const position = await lookup<RowPosition>(base, "answer", key);
      const row = position
        ? await readRow(base, source, "traces", position)
        : null;
      let trace = row?.trace == null ? null : String(row.trace);
      if (trace != null && Array.from(trace).length > TRACE_MAX_CHARS) {
        const chars = Array.from(trace).slice(0, TRACE_MAX_CHARS);
        const space = chars.lastIndexOf(" ");
        trace =
          chars
            .slice(0, space > TRACE_MAX_CHARS * 0.6 ? space : undefined)
            .join("")
            .trimEnd() + " …";
      }
      value = { trace };
    }
    self.postMessage({ value });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({
      error: /\b429\b/.test(message)
        ? "Hugging Face is busy. Please try again shortly."
        : message,
    });
  }
};
