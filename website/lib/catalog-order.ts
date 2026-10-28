import { agenticFirst, laterInCatalog } from "../content/curated/catalog-order";

type CatalogEntry = { slug: string; name: string; year: number | null };

const priority = new Map<string, number>(
  agenticFirst.map((slug, index) => [slug, index]),
);
const later = new Set<string>(laterInCatalog);

function rank(slug: string): number {
  return priority.get(slug) ?? agenticFirst.length + Number(later.has(slug));
}

// Sort the full filtered catalog before pagination. Unlisted benchmarks appear
// between the two curated lists, newest first, without requiring a new entry.
export function compareCatalogOrder(a: CatalogEntry, b: CatalogEntry): number {
  return (
    rank(a.slug) - rank(b.slug) ||
    (b.year ?? 0) - (a.year ?? 0) ||
    a.name.localeCompare(b.name) ||
    a.slug.localeCompare(b.slug)
  );
}
