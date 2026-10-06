import { BenchmarkSearchClient } from "@/components/benchmark-search-client";
import { buildBenchmarkSearchData } from "@/lib/benchmark-search-data";

export function BenchmarkSearch({
  variant = "section",
  slugs,
  scopeLabel = "the Measurement Data Bank",
}: {
  variant?: "section" | "hero";
  slugs?: readonly string[];
  scopeLabel?: string;
} = {}) {
  const { categoryLabels, localDocuments } = buildBenchmarkSearchData(slugs);

  const search = (
    <BenchmarkSearchClient
      localDocuments={localDocuments}
      categoryLabels={categoryLabels}
      resultLayout={variant === "hero" ? "single" : "responsive"}
      scopeLabel={scopeLabel}
    />
  );

  if (variant === "hero") {
    return (
      <section
        className="rd-measurement-hero-search"
        aria-labelledby="search-measurement-evidence"
      >
        <p className="rd-mono opacity-60">Search the exchange</p>
        <h2
          id="search-measurement-evidence"
          className="mt-2 text-xl font-medium leading-tight tracking-tight sm:text-2xl"
        >
          Search measurements
        </h2>
        <div className="mt-4">{search}</div>
      </section>
    );
  }

  return (
    <section
      className="rd-white-band rd-band"
      aria-labelledby="search-the-bank"
    >
      <div className="rd-container grid min-w-0 grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(230px,0.55fr)_minmax(0,1.45fr)] lg:gap-16">
        <div className="min-w-0">
          <p className="rd-mono opacity-60">Find a benchmark</p>
          <h2 id="search-the-bank" className="rd-h2 mt-4">
            Search the bank
          </h2>
          <p className="rd-prose mt-5 text-base font-light leading-relaxed opacity-70">
            Look up a use case, benchmark, or AI subject. Results link to the
            response matrix and report aggregate means on the benchmark&rsquo;s
            original scale.
          </p>
        </div>
        {search}
      </div>
    </section>
  );
}
