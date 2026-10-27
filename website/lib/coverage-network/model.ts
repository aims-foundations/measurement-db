export type CoverageNode = {
  id: string;
  kind: "benchmark" | "subject";
  label: string;
  slug?: string;
  provider?: string;
  categories: string[];
  degree: number;
  x: number;
  y: number;
};

export type CoverageEdge = {
  id: string;
  source: string;
  target: string;
  items: number;
};

export type CoverageNetwork = {
  nodes: CoverageNode[];
  edges: CoverageEdge[];
};

type BenchmarkInput = {
  slug: string;
  name: string;
  categories: readonly string[];
};
type SubjectInput = {
  name: string;
  provider: string | null;
  benchmarks: readonly string[];
};
type MarginalInput = { slug: string; model: string; items: number };

/** An edge requires recorded items, including records whose grade is null.
 * Canonical names merge identities across benchmarks, not configurations.
 * The caller supplies the visible catalog, so hidden benchmarks never leak in.
 */
export function buildCoverageNetwork(
  benchmarks: readonly BenchmarkInput[],
  subjects: readonly SubjectInput[],
  marginals: readonly MarginalInput[],
): CoverageNetwork {
  const visible = new Set(benchmarks.map((b) => b.slug));
  const nodes: CoverageNode[] = benchmarks.map((b) => ({
    id: `benchmark:${b.slug}`,
    kind: "benchmark",
    label: b.name,
    slug: b.slug,
    categories: [...b.categories],
    degree: 0,
    x: 0,
    y: 0,
  }));
  for (const s of subjects) {
    if (!s.benchmarks.some((slug) => visible.has(slug))) continue;
    nodes.push({
      id: `subject:${s.name}`,
      kind: "subject",
      label: s.name,
      provider: s.provider || "Unknown",
      categories: [],
      degree: 0,
      x: 0,
      y: 0,
    });
  }
  const byId = new Map(nodes.map((n) => [n.id, n]));
  if (byId.size !== nodes.length)
    throw new Error("Duplicate coverage node identity");
  const edges: CoverageEdge[] = [];
  const seen = new Set<string>();
  for (const row of marginals) {
    if (!visible.has(row.slug)) continue;
    if (!Number.isSafeInteger(row.items) || row.items < 0)
      throw new Error("Invalid coverage item count");
    if (!row.items) continue;
    const source = `subject:${row.model}`;
    const target = `benchmark:${row.slug}`;
    const subject = byId.get(source),
      benchmark = byId.get(target);
    if (!subject || !benchmark)
      throw new Error(`Unmapped coverage edge: ${row.model} / ${row.slug}`);
    const id = JSON.stringify([source, target]);
    if (seen.has(id))
      throw new Error(`Duplicate coverage edge: ${row.model} / ${row.slug}`);
    seen.add(id);
    subject.degree++;
    benchmark.degree++;
    edges.push({ id, source, target, items: row.items });
  }
  return { nodes, edges };
}

export function filterCoverageNetwork(
  data: CoverageNetwork,
  filters: { domain: string; provider: string; showUnmapped: boolean },
) {
  const eligible = new Set(
    data.nodes
      .filter((node) =>
        node.kind === "benchmark"
          ? (!filters.domain || node.categories.includes(filters.domain)) &&
            (filters.showUnmapped || node.degree > 0)
          : !filters.provider || node.provider === filters.provider,
      )
      .map((n) => n.id),
  );
  const edges = data.edges.filter(
    (e) => eligible.has(e.source) && eligible.has(e.target),
  );
  const nodes = new Set(edges.flatMap((e) => [e.source, e.target]));
  // Preserve true catalog isolates; a provider filter cannot attribute them.
  if (!filters.provider) {
    for (const n of data.nodes) {
      if (n.kind === "benchmark" && eligible.has(n.id)) nodes.add(n.id);
    }
  }
  return { nodes, edges };
}
