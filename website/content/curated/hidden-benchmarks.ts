import hiddenBenchmarkSlugsJson from "@/content/curated/hidden-benchmarks.json";

// Benchmarks temporarily hidden from the site. Their data stays in the bank,
// in generated/benchmark-details.json and in the gallery asset repo untouched — a slug
// here is filtered out of the card list (content/measurement-db.ts), which
// removes its card, its contribution to the headline stats and its detail page
// (app/[slug]/page.tsx 404s any slug without a card), and denied by the asset
// proxy (app/benchmarks/r/[rev]/[...path]/route.ts), which closes the only
// public route to its traces, matrices, data payloads and images. To un-hide,
// delete the slug from this list and redeploy.
//
// Hand-maintained on purpose: the generated JSONs are rewritten wholesale by
// generate_benchmark_gallery.py, so a flag there would not survive a rebuild — same
// reasoning as judgedBenchmarkSlugs in content/measurement-db.ts.
//
// Current entries: the ten benchmarks whose data was released in the
// prediction competition dataset (hf.co/datasets/aims-foundations/
// predictive_eval_competition), hidden 2026-08-19 while the competition runs.
// Listing a slug with no published card (chi_bench, and the five below the
// n_subjects gate today) is harmless and keeps them hidden if a future
// rebuild publishes them.
// The values live in JSON so the website and the Meilisearch sync script read
// the exact same visibility list. Keep the explanatory policy here; edit the
// adjacent JSON file when a benchmark should be hidden or restored.
export const hiddenBenchmarkSlugs: readonly string[] = hiddenBenchmarkSlugsJson;

export const hiddenBenchmarkSet: ReadonlySet<string> = new Set(
  hiddenBenchmarkSlugs,
);
