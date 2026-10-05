// ---------------------------------------------------------------------------
//  Source-backed credit metadata for benchmark detail pages.
//
//  The JSON is generated, not hand-edited, by
//  scripts/render_website/generate_benchmark_attributions.py. It deliberately
//  distinguishes papers from provider-published datasets, named people from
//  organization authors, and an available citation from one the producer did
//  not supply. Keep those distinctions in consumers rather than filling gaps
//  with inferred bibliographic data.
// ---------------------------------------------------------------------------

import attributionJson from "@/content/generated/benchmark-attributions.json";
import { benchmarks } from "@/content/measurement-db";

export const LEAD_AUTHOR_AFFILIATION_SCOPE =
  "first_two_authors_first_listed_institutions" as const;

type AttributionAuthorBase = {
  /** Display-ready author name. */
  name: string;
  /** Exact author token retained from the source BibTeX. */
  bibtexName: string;
};

export type AttributionPersonAuthor = AttributionAuthorBase & {
  kind: "person";
};

export type AttributionOrganizationAuthor = AttributionAuthorBase & {
  kind: "organization";
};

export type AttributionAuthor =
  | AttributionPersonAuthor
  | AttributionOrganizationAuthor;

/** A named party credited for a non-paper reference. Roles are deliberately
 * source-specific (for example, dataset producer or measurement release). */
export type AttributionCredit = {
  name: string;
  role: string;
  url: string;
};

type AttributionReferenceBase = {
  title: string;
  url: string;
  /** Provider publication year, retained as YYYY text. */
  year: string;
  authors: readonly AttributionAuthor[];
  producers: readonly AttributionCredit[];
  contributors: readonly AttributionCredit[];
  measurementSources: readonly AttributionCredit[];
};

export type PaperAttributionReference = AttributionReferenceBase & {
  kind: "paper";
};

/** A provider-published dataset with explicit producer/contributor credit. It
 * is not presented as a paper when the provider supplied no formal paper. */
export type DatasetAttributionReference = AttributionReferenceBase & {
  kind: "dataset";
};

export type BenchmarkAttributionReference =
  | PaperAttributionReference
  | DatasetAttributionReference;

export type AvailableBenchmarkCitation = {
  status: "available";
  entryType: string;
  key: string;
  /** Exact, copy-ready source BibTeX. */
  bibtex: string;
  source: "local_metadata" | "curated_primary_source_override";
  evidenceUrl: string;
};

/** The producer did not publish a citation template. Null fields stay null so
 * consumers cannot accidentally offer an invented citation. */
export type ProducerCitationNotProvided = {
  status: "not_provided";
  entryType: null;
  key: null;
  bibtex: null;
  source: "producer_did_not_supply";
  evidenceUrl: string;
};

export type BenchmarkCitation =
  | AvailableBenchmarkCitation
  | ProducerCitationNotProvided;

export type LeadAuthorInstitution = {
  name: string;
};

export type AvailableLeadAuthorAffiliations = {
  affiliationStatus: "available";
  affiliationScope: typeof LEAD_AUTHOR_AFFILIATION_SCOPE;
  /** Deduplicated first-listed institutions for only the first two authors. */
  leadAuthorInstitutions: readonly LeadAuthorInstitution[];
};

export type InapplicableLeadAuthorAffiliations = {
  affiliationStatus: "not_applicable";
  affiliationScope: null;
  leadAuthorInstitutions: readonly [];
};

export type BenchmarkAttributionAffiliations =
  | AvailableLeadAuthorAffiliations
  | InapplicableLeadAuthorAffiliations;

export type AttributionEvidence = {
  evidenceUrl: string;
  reason: string;
};

export type AttributionAliasProvenance = AttributionEvidence & {
  /** Metadata-only source slug; never a measurement-data alias. */
  metadataSlug: string;
};

export type AttributionAuthorVerification = AttributionEvidence & {
  expectedCount: number;
  organizationAuthors: readonly string[];
};

export type BenchmarkAttributionProvenance = {
  alias: AttributionAliasProvenance | null;
  referenceOverride: AttributionEvidence | null;
  citationOverride: AttributionEvidence | null;
  authorVerification: AttributionAuthorVerification | null;
  affiliationOverride: AttributionEvidence | null;
  /** Curated institution-registry path, absent when affiliations do not apply. */
  affiliations: string | null;
  affiliationRegistrySource: string | null;
  affiliationEvidenceUrl: string | null;
};

type BenchmarkAttributionBase = {
  /** Slug whose benchmark metadata supplied the reference. */
  metadataSlug: string;
  metadataPath: string;
  benchmarkName: string;
  reference: BenchmarkAttributionReference;
  citation: BenchmarkCitation;
  provenance: BenchmarkAttributionProvenance;
};

export type BenchmarkAttribution = BenchmarkAttributionBase &
  BenchmarkAttributionAffiliations;

export type BenchmarkAttributionCoverage = {
  benchmarks: number;
  withCitation: number;
  withoutProducerCitation: readonly string[];
  unreviewed?: readonly string[];
};

type BenchmarkAttributionManifest = {
  schemaVersion: 1;
  affiliationScope: typeof LEAD_AUTHOR_AFFILIATION_SCOPE;
  coverage: BenchmarkAttributionCoverage;
  benchmarks: Readonly<Record<string, BenchmarkAttribution | null>>;
};

const manifest = attributionJson as unknown as BenchmarkAttributionManifest;

// benchmark-attributions.json is generated for the visible catalog, not for
// every benchmark ever curated. Match measurement-db.ts (the site's canonical
// hidden-benchmark filter) and fail the build loudly if either generated file
// has moved ahead of the other.
const visibleSlugs = benchmarks.map((benchmark) => benchmark.slug);
const visibleSlugSet = new Set(visibleSlugs);
const manifestSlugs = Object.keys(manifest.benchmarks);
const missing = visibleSlugs.filter((slug) => !(slug in manifest.benchmarks));
const nonVisible = manifestSlugs.filter((slug) => !visibleSlugSet.has(slug));

if (
  missing.length > 0 ||
  nonVisible.length > 0 ||
  manifest.coverage.benchmarks !== visibleSlugs.length
) {
  const issues = [
    missing.length > 0 ? `missing: ${missing.join(", ")}` : null,
    nonVisible.length > 0 ? `non-visible: ${nonVisible.join(", ")}` : null,
    manifest.coverage.benchmarks !== visibleSlugs.length
      ? `reported ${manifest.coverage.benchmarks}, expected ${visibleSlugs.length}`
      : null,
  ].filter((issue): issue is string => issue !== null);
  throw new Error(
    "benchmark-attributions.json does not match the visible benchmark pages " +
      `(${issues.join("; ")}) — re-run ` +
      "scripts/render_website/generate_benchmark_attributions.py.",
  );
}

export function getBenchmarkAttribution(
  slug: string,
): BenchmarkAttribution | undefined {
  return manifest.benchmarks[slug] ?? undefined;
}
