import { benchmarkDataOrigin } from "@/lib/benchmark-data";

const kinds = new Set(["item", "answer"]);
const fields = ["key", "item_id"];

export async function GET(
  request: Request,
  { params }: { params: Promise<{ slug: string }> },
) {
  const { slug } = await params;
  const origin = benchmarkDataOrigin(slug);
  const query = new URL(request.url).searchParams;
  const kind = query.get("kind") ?? "";
  if (!origin || !kinds.has(kind)) return new Response(null, { status: 404 });
  const target = new URL(`${origin}/${slug}/${kind}`);
  for (const field of fields) {
    const value = query.get(field);
    if (value !== null) target.searchParams.set(field, value);
  }
  try {
    const response = await fetch(target, {
      cache: "no-store",
      signal: AbortSignal.timeout(120_000),
    });
    if (!response.ok)
      return new Response("Source data unavailable", {
        status: response.status,
      });
    return new Response(await response.text(), {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new Response("Source data unavailable", { status: 502 });
  }
}
