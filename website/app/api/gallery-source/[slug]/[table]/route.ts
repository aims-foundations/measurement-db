import { serveHuggingFaceRange } from "@/lib/server/huggingface-range";
import sources from "@/content/generated/gallery-sources.json";

type Source = {
  revision: string;
  directory: string;
  tables: Record<string, { size: number }>;
};
const catalog = sources as Record<string, Source>;

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
  const url = `https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/${source.revision}/${source.directory}/${table}.parquet`;
  return serveHuggingFaceRange(request, url, info.size);
}
