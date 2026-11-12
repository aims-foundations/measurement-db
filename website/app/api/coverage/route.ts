import { coverageNetwork } from "@/content/coverage-network";
import { benchmarkCategories } from "@/content/measurement-db";

// A build-time snapshot shared by the graph and matrix; no Parquet reads here.
export const dynamic = "force-static";
export function GET() {
  return Response.json({
    data: coverageNetwork,
    categories: benchmarkCategories.map(({ id, label }) => ({ id, label })),
  });
}
