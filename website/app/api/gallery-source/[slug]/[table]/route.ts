import { open } from "node:fs/promises";
import path from "node:path";
import { serveHuggingFaceRange } from "@/lib/server/huggingface-range";
import sources from "@/content/generated/gallery-sources.json";

type Source = {
  revision: string;
  directory: string;
  tables: Record<string, { size: number }>;
};
const catalog = sources as Record<string, Source>;
/* Development only: serve tables from <dir>/<slug>/<table>.parquet instead of HF,
   to preview a locally rebuilt benchmark before it is uploaded. */
const LOCAL_DIR =
  process.env.NODE_ENV === "development"
    ? process.env.GALLERY_SOURCE_DIR
    : undefined;

async function localRange(file: string, range: string | null) {
  const match = range?.match(/^bytes=(\d+)-(\d+)$/);
  const handle = await open(file).catch(() => null);
  if (!handle) return new Response("Unknown local table.", { status: 404 });
  try {
    const size = (await handle.stat()).size;
    const start = Number(match?.[1]),
      end = Number(match?.[2]);
    if (!match || start > end || end >= size)
      return new Response("A valid byte range is required.", { status: 416 });
    const bytes = new Uint8Array(end - start + 1);
    await handle.read(bytes, 0, bytes.length, start);
    return new Response(bytes, {
      status: 206,
      headers: {
        "Content-Type": "application/octet-stream",
        "Cache-Control": "no-store",
        "Content-Range": `bytes ${start}-${end}/${size}`,
      },
    });
  } finally {
    await handle.close();
  }
}

export async function GET(
  request: Request,
  context: { params: Promise<{ slug: string; table: string }> },
) {
  const { slug, table } = await context.params;
  const source = Object.hasOwn(catalog, slug) ? catalog[slug] : undefined;
  const info =
    source && Object.hasOwn(source.tables, table)
      ? source.tables[table]
      : undefined;
  if (!source || !info)
    return new Response("Unknown source table.", { status: 404 });
  const range = request.headers.get("range");
  // slug and table are catalog keys here, so the path cannot leave LOCAL_DIR.
  if (LOCAL_DIR)
    return localRange(path.join(LOCAL_DIR, slug, `${table}.parquet`), range);
  const url = `https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/${source.revision}/${source.directory}/${table}.parquet`;
  return serveHuggingFaceRange(request, url, info.size);
}
