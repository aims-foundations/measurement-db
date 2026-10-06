import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptsDirectory = dirname(fileURLToPath(import.meta.url));
const websiteDirectory = dirname(scriptsDirectory);
const searchConfig = readJson("meilisearch.config.json");

export const BENCHMARKS_INDEX_UID = searchConfig.benchmarksIndex;
export const RESULTS_INDEX_UID = searchConfig.resultsIndex;

const CATEGORY_SEARCH_TEXT = {
  safety: "safety adversarial robustness harmful behavior refusal",
  cybersecurity: "cybersecurity cyber security",
  medicine: "medicine medical clinical healthcare",
  law: "law legal",
  finance: "finance financial economics markets",
  mathematics: "mathematics math mathematical",
  software_engineering:
    "software engineering coding code programming program synthesis",
  ml_engineering: "machine learning engineering ML research automation systems",
  agents_and_tool_use: "agents agentic tool use computer use autonomous",
  science: "science scientific",
  multilingual: "multilingual languages cross-lingual",
  cultural: "cultural culture region norms",
  education: "education teaching learning assessment",
  preference: "preference human preference judged response quality",
  reward_modeling: "reward modeling reward model judge",
  nlp_task: "natural language processing NLP extraction classification QA",
  knowledge: "knowledge factual world knowledge",
  reasoning: "reasoning problem solving multi-step",
  general: "general broad general-purpose",
};

export const BENCHMARKS_INDEX_SETTINGS = {
  searchableAttributes: [
    "name",
    "description",
    "fullDescription",
    "categoryText",
    "modality",
    "scaleLabel",
    "license",
    "institutionNames",
    "searchTerms",
  ],
  displayedAttributes: [
    "slug",
    "name",
    "description",
    "categories",
    "license",
    "releaseYear",
    "items",
    "models",
    "resultCount",
    "itemResponses",
    "scaleType",
  ],
  filterableAttributes: [
    "categories",
    "modality",
    "scaleType",
    "itemResponses",
    "saturation",
    "releaseYear",
    "license",
  ],
  sortableAttributes: ["releaseYear", "items", "models", "attacks", "observed"],
};

export const RESULTS_INDEX_SETTINGS = {
  searchableAttributes: [
    "modelName",
    "benchmarkName",
    "benchmarkDescription",
    "benchmarkFullDescription",
    "categoryText",
    "scaleLabel",
    "searchTerms",
  ],
  displayedAttributes: [
    "id",
    "benchmarkSlug",
    "benchmarkName",
    "modelName",
    "score",
    "scaleLabel",
    "scaleType",
    "isBinary",
    "categories",
    "releaseYear",
  ],
  filterableAttributes: [
    "benchmarkSlug",
    "categories",
    "scaleType",
    "isBinary",
    "releaseYear",
  ],
  // Scores are only comparable within the same benchmark. The public search
  // UI deliberately leaves relevance ordering intact and does not use this
  // sort globally; it is available for a future benchmark-scoped view.
  sortableAttributes: ["score"],
};

function readJson(relativePath) {
  return JSON.parse(readFileSync(join(websiteDirectory, relativePath), "utf8"));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function unique(values, label) {
  const seen = new Set();
  for (const value of values) {
    assert(!seen.has(value), `Duplicate ${label}: ${value}`);
    seen.add(value);
  }
}

function scaleType(detail) {
  if (Array.isArray(detail.categories) && detail.categories.length > 0) {
    return "Categorical";
  }
  return detail.isBinary ? "Binary" : "Graded";
}

function releaseYear(releaseDate) {
  const year = Number(String(releaseDate ?? "").slice(0, 4));
  return Number.isInteger(year) && year >= 1900 && year <= 2200 ? year : null;
}

function categoryText(categories) {
  return categories
    .map(
      (category) =>
        CATEGORY_SEARCH_TEXT[category] ?? category.replaceAll("_", " "),
    )
    .join(" ");
}

function saturationValue(value) {
  return value === null ? "unknown" : value ? "yes" : "no";
}

/**
 * Build the two intentionally small, public search corpora.
 *
 * Visibility is applied before joining details. This is important because the
 * generated card/detail files also contain temporarily hidden competition data
 * and unpublished benchmarks that must never reach the public search service.
 */
export function buildSearchDocuments() {
  const cards = readJson("content/generated/benchmark-cards.json");
  const details = readJson("content/generated/benchmark-details.json");
  const hiddenSlugs = readJson("content/curated/hidden-benchmarks.json");

  assert(Array.isArray(cards), "benchmark-cards.json must contain an array");
  assert(
    details && typeof details === "object" && !Array.isArray(details),
    "benchmark-details.json must contain an object keyed by slug",
  );
  assert(
    Array.isArray(hiddenSlugs),
    "hidden-benchmarks.json must contain an array",
  );
  unique(hiddenSlugs, "hidden benchmark slug");

  const hidden = new Set(hiddenSlugs);
  const visibleCards = cards.filter((card) => !hidden.has(card.slug));
  unique(
    visibleCards.map((card) => card.slug),
    "visible benchmark slug",
  );

  const benchmarks = [];
  const results = [];

  for (const card of visibleCards) {
    const detail = details[card.slug];
    assert(detail, `Missing benchmark detail for visible slug: ${card.slug}`);
    assert(
      Array.isArray(detail.matrixRows) &&
        Array.isArray(detail.matrixRowIds) &&
        Array.isArray(detail.matrixRowScores),
      `Invalid result arrays for ${card.slug}`,
    );
    assert(
      detail.matrixRows.length === detail.matrixRowIds.length &&
        detail.matrixRows.length === detail.matrixRowScores.length,
      `Result arrays have different lengths for ${card.slug}`,
    );

    const categories = Array.isArray(card.categories) ? card.categories : [];
    const type = scaleType(detail);
    const year = releaseYear(detail.releaseDate ?? card.releaseDate);
    const institutions = Array.isArray(card.institutions)
      ? card.institutions.map((institution) => institution.name).filter(Boolean)
      : [];
    const fullDescription = detail.description ?? card.description;

    benchmarks.push({
      slug: card.slug,
      name: card.name,
      description: card.description,
      fullDescription,
      categories,
      categoryText: categoryText(categories),
      modality: Array.isArray(detail.modality) ? detail.modality : [],
      license: card.license ?? detail.license ?? null,
      releaseYear: year,
      items: card.items,
      models: card.models,
      responses: card.responses,
      resultCount: detail.matrixRows.length,
      attacks: detail.attacks ?? null,
      observed: detail.stats?.observed ?? null,
      meanResponse: detail.stats?.meanResponse ?? null,
      itemResponses: Boolean(card.itemResponses),
      saturation: saturationValue(card.saturation),
      scaleType: type,
      scaleLabel: detail.scaleLabel,
      institutionNames: institutions,
      searchTerms: "benchmark evaluation performance results scores use case",
    });

    for (let index = 0; index < detail.matrixRows.length; index += 1) {
      const rowId = String(detail.matrixRowIds[index]);
      const id = `${card.slug}_${rowId}`;
      const score = detail.matrixRowScores[index];
      assert(
        score === null || (typeof score === "number" && Number.isFinite(score)),
        `Invalid score for ${id}`,
      );
      assert(
        /^[A-Za-z0-9_-]+$/.test(id),
        `Meilisearch result id contains unsupported characters: ${id}`,
      );

      results.push({
        id,
        benchmarkSlug: card.slug,
        benchmarkName: card.name,
        benchmarkDescription: card.description,
        benchmarkFullDescription: fullDescription,
        modelName: String(detail.matrixRows[index]),
        score,
        scaleLabel: detail.scaleLabel,
        scaleType: type,
        isBinary: Boolean(detail.isBinary),
        categories,
        categoryText: categoryText(categories),
        releaseYear: year,
        searchTerms: "model benchmark performance result score evaluation",
      });
    }
  }

  unique(
    results.map((result) => result.id),
    "result id",
  );

  const visibleSlugs = new Set(benchmarks.map((benchmark) => benchmark.slug));
  for (const result of results) {
    assert(
      visibleSlugs.has(result.benchmarkSlug),
      `Result references a non-public benchmark: ${result.id}`,
    );
  }
  for (const slug of hidden) {
    assert(
      !visibleSlugs.has(slug),
      `Hidden benchmark was included in search documents: ${slug}`,
    );
  }

  return { benchmarks, results, hiddenSlugs };
}
