import previewsJson from "@/content/generated/record-previews.json";
import type { RecordPreview } from "@/lib/benchmark-records/record-preview";
import { serveHuggingFaceRange } from "@/lib/server/huggingface-range";
import { hiddenBenchmarkSet } from "@/content/curated/hidden-benchmarks";
const previews = previewsJson as unknown as Record<
  string,
  Record<string, RecordPreview>
>;

export async function GET(
  request: Request,
  context: {
    params: Promise<{ slug: string; table: string; file: string }>;
  },
) {
  const { slug, table, file } = await context.params;
  const preview =
    Object.hasOwn(previews, slug) && Object.hasOwn(previews[slug], table)
      ? previews[slug][table]
      : undefined;
  if (!preview || hiddenBenchmarkSet.has(slug))
    return new Response("Unknown record source.", { status: 404 });
  const pack = /^pack-(\d+)\.bin$/.exec(file);
  const bytes = pack ? preview.packs[Number(pack[1])] : undefined;
  if (!bytes) return new Response("Unknown record file.", { status: 404 });
  return serveHuggingFaceRange(
    request,
    `https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/${preview.revision}/${preview.directory}/${file}`,
    bytes,
  );
}
