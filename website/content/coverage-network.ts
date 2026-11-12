import subjects from "@/content/generated/model-timeline.json";
import { benchmarks } from "@/content/measurement-db";
import { marginalsFor } from "@/content/chart-marginals";
import { buildCoverageNetwork } from "@/lib/coverage-network/model";
import { positionCoverageNetwork } from "@/lib/coverage-network/layout";

// Uses the same snapshot, canonical identities, and visibility rules as the
// catalog totals. marginalsFor also rejects incomplete benchmark manifests.
export const coverageNetwork = positionCoverageNetwork(
  buildCoverageNetwork(
    benchmarks,
    subjects,
    marginalsFor(benchmarks.map((b) => b.slug)),
  ),
);
