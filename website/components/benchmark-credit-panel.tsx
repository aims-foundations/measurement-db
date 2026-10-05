import { ArrowUpRight, CaretDown } from "@phosphor-icons/react/dist/ssr";
import { ActionLink } from "@/components/action-link";
import { CitationBlock } from "@/components/citation-block";
import type {
  AttributionAuthor,
  AttributionCredit,
  BenchmarkAttribution,
} from "@/content/benchmark-attributions";

type BenchmarkCreditPanelProps = {
  attribution: BenchmarkAttribution;
  originalSourceHref: string;
  /** Number of author names retained in the compact summary. */
  authorPreviewCount?: number;
  className?: string;
};

function CreditLink({ credit }: { credit: AttributionCredit }) {
  const content = (
    <>
      <span className="font-medium">{credit.name}</span>
      <span className="text-xs opacity-65">{credit.role}</span>
    </>
  );

  if (!credit.url) {
    return <span className="flex min-w-0 flex-col gap-0.5">{content}</span>;
  }

  return (
    <a
      href={credit.url}
      rel="noopener noreferrer"
      target="_blank"
      className="group/credit flex min-w-0 flex-col gap-0.5 underline decoration-black/20 underline-offset-4 transition-colors hover:text-[var(--cardinal)] hover:decoration-[var(--cardinal)] focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[var(--digital-blue)]"
    >
      <span className="inline-flex items-center gap-1.5 font-medium">
        {credit.name}
        <ArrowUpRight
          aria-hidden="true"
          className="shrink-0 transition-transform group-hover/credit:-translate-y-0.5 group-hover/credit:translate-x-0.5"
          size={13}
        />
      </span>
      <span className="text-xs opacity-65">{credit.role}</span>
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}

function CreditGroup({
  className,
  credits,
  label,
}: {
  className: string;
  credits: readonly AttributionCredit[];
  label: string;
}) {
  if (credits.length === 0) return null;

  return (
    <div className={className}>
      <dt className="rd-mono !text-[0.65rem] opacity-60">{label}</dt>
      <dd className="mt-2 grid min-w-0 gap-2">
        {credits.map((credit) => (
          <CreditLink key={`${credit.role}-${credit.name}`} credit={credit} />
        ))}
      </dd>
    </div>
  );
}

function Authors({
  authors,
  previewCount,
}: {
  authors: readonly AttributionAuthor[];
  previewCount: number;
}) {
  if (authors.length === 0) return null;

  const compactAuthors = authors.slice(0, previewCount);
  const remaining = authors.length - compactAuthors.length;

  if (remaining <= 0) {
    return (
      <div className="benchmark-credit-authors">
        <p className="rd-mono !text-[0.65rem] opacity-60">
          {authors.length === 1 ? "Author" : "Authors"}
        </p>
        <p className="mt-2 text-sm leading-relaxed sm:text-base">
          {authors.map((author) => author.name).join(", ")}
        </p>
      </div>
    );
  }

  return (
    <div className="benchmark-credit-authors">
      <p className="rd-mono !text-[0.65rem] opacity-60">Authors</p>
      <details className="group mt-1 border-b border-black/15">
        <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between gap-3 py-2 text-sm leading-relaxed marker:hidden focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)] sm:text-base [&::-webkit-details-marker]:hidden">
          <span className="min-w-0 group-open:hidden">
            {compactAuthors.map((author) => author.name).join(", ")}
            <span className="whitespace-nowrap opacity-60">
              {` +${remaining} more`}
            </span>
          </span>
          <span className="hidden min-w-0 group-open:inline">
            All {authors.length.toLocaleString("en-US")} authors
          </span>
          <span className="inline-flex shrink-0 items-center gap-1.5 text-xs font-medium">
            <span className="group-open:hidden">Show all</span>
            <span className="hidden group-open:inline">Hide</span>
            <CaretDown
              aria-hidden="true"
              className="transition-transform duration-200 group-open:rotate-180"
              size={14}
            />
          </span>
        </summary>
        <ol className="benchmark-credit-author-list grid min-w-0 gap-y-1.5 border-t border-black/10 py-3 text-sm leading-relaxed">
          {authors.map((author, index) => (
            <li key={`${author.bibtexName ?? author.name}-${index}`}>
              <span className="mr-2 font-mono text-[0.65rem] tabular-nums opacity-40">
                {(index + 1).toString().padStart(2, "0")}
              </span>
              {author.name}
              {author.kind === "organization" ? (
                <span className="ml-2 text-xs opacity-55">organization</span>
              ) : null}
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}

/**
 * Compact credit card for a benchmark detail page. Native `<details>` keeps
 * long author lists usable without hiding any credit from keyboard or screen
 * reader users; the nested CitationBlock owns the clipboard interaction.
 */
export function BenchmarkCreditPanel({
  attribution,
  originalSourceHref,
  authorPreviewCount = 4,
  className = "",
}: BenchmarkCreditPanelProps) {
  const { citation, reference } = attribution;
  const previewCount = Math.max(1, Math.floor(authorPreviewCount));
  const isPaper = reference.kind === "paper";
  const hasNonPaperCredits =
    reference.producers.length > 0 ||
    reference.contributors.length > 0 ||
    reference.measurementSources.length > 0;

  return (
    <article
      aria-label={`${isPaper ? "Paper" : "Source"} and credit for ${attribution.benchmarkName}`}
      className={`benchmark-credit-panel rd-card gap-6 sm:gap-7 ${className}`.trim()}
    >
      <header className="benchmark-credit-reference min-w-0">
        <div className="flex items-center justify-between gap-3">
          <p className="benchmark-credit-kicker rd-mono opacity-60">
            {isPaper ? "Paper and credit" : "Source and credit"}
          </p>
          <span className="shrink-0 font-mono text-xs tabular-nums opacity-60">
            {reference.year}
          </span>
        </div>
        <h2 className="benchmark-credit-title rd-h3 mt-3 max-w-[42ch] break-words">
          <a
            href={reference.url}
            rel="noopener noreferrer"
            target="_blank"
            className="group/title inline underline decoration-[var(--cardinal)]/30 decoration-1 underline-offset-4 transition-colors hover:text-[var(--cardinal)] hover:decoration-[var(--cardinal)] focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[var(--digital-blue)]"
          >
            {reference.title}
            <ArrowUpRight
              aria-hidden="true"
              className="ml-1.5 inline shrink-0 align-baseline transition-transform group-hover/title:-translate-y-0.5 group-hover/title:translate-x-0.5"
              size={16}
            />
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        </h2>
        <div className="benchmark-credit-original-source mt-5">
          <ActionLink
            href={originalSourceHref}
            variant="primary"
            external
            className="min-h-11"
          >
            Original source
          </ActionLink>
        </div>
      </header>

      <Authors authors={reference.authors} previewCount={previewCount} />

      {hasNonPaperCredits ? (
        <dl className="benchmark-credit-nonpaper grid min-w-0 gap-5 border-t border-black/15 pt-5">
          <CreditGroup
            className="benchmark-credit-producers"
            credits={reference.producers}
            label="Producer"
          />
          <CreditGroup
            className="benchmark-credit-contributors"
            credits={reference.contributors}
            label="Contributor"
          />
          <CreditGroup
            className="benchmark-credit-measurement-sources"
            credits={reference.measurementSources}
            label="Measurement source"
          />
        </dl>
      ) : null}

      <div className="benchmark-credit-citation border-t border-black/15 pt-5 sm:pt-6">
        <p className="rd-mono mb-3 !text-[0.65rem] opacity-60">Citation</p>
        {citation.status === "available" && citation.bibtex ? (
          <details className="benchmark-credit-citation-details group">
            <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between gap-3 border-y border-black/15 px-1 text-sm marker:hidden focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)] [&::-webkit-details-marker]:hidden">
              <span className="font-medium">BibTeX citation</span>
              <span className="inline-flex shrink-0 items-center gap-1.5 text-xs font-medium">
                <span className="group-open:hidden">Show</span>
                <span className="hidden group-open:inline">Hide</span>
                <CaretDown
                  aria-hidden="true"
                  className="transition-transform duration-200 group-open:rotate-180"
                  size={14}
                />
              </span>
            </summary>
            <div className="pt-3">
              <CitationBlock bibtex={citation.bibtex} />
            </div>
          </details>
        ) : (
          <div
            className="rounded-md bg-black/[0.035] px-4 py-3 text-sm leading-relaxed"
            role="note"
          >
            <p className="font-medium">No citation template was provided.</p>
            <p className="mt-1 opacity-65">
              The benchmark producer does not currently supply a formal
              citation, so measurement-db does not invent one. Please use the
              linked source and credit information above.
            </p>
          </div>
        )}
      </div>
    </article>
  );
}
