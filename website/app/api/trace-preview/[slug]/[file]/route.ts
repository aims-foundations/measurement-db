import previewsJson from "@/content/generated/trace-previews.json";
import type { TracePreview } from "@/lib/benchmark-records/trace-preview";
import { serveHuggingFaceRange } from "@/lib/server/huggingface-range";
import { hiddenBenchmarkSet } from "@/content/curated/hidden-benchmarks";

const previews = previewsJson as unknown as Record<string, TracePreview>;

export async function GET(
  request: Request,
  context: {
    params: Promise<{ slug: string; file: string }>;
  },
) {
  const { slug, file } = await context.params;
  const preview = Object.hasOwn(previews, slug) ? previews[slug] : undefined;
  const pack = /^pack-(\d+)\.bin$/.exec(file);
  const bytes = preview && pack ? preview.packs[Number(pack[1])] : undefined;
  if (!preview || !bytes || hiddenBenchmarkSet.has(slug))
    return new Response("Unknown trace display file.", { status: 404 });
  return serveHuggingFaceRange(
    request,
    `https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/${preview.revision}/${preview.directory}/${file}`,
    bytes,
  );
}
