import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { ActionLink } from "@/components/action-link";
import { AnnotationMindmap } from "@/components/annotation-mindmap";
import {
  annotations,
  claims,
  getAnnotation,
  type Annotation,
  type ClaimAssessment,
} from "@/content/annotations";

type AnnotationPageProps = {
  params: Promise<{ slug: string }>;
};

// The annotated set is a fixed, hand-authored list, so every route is known at
// build time and prerenders.
export function generateStaticParams() {
  return annotations.map(({ slug }) => ({ slug }));
}

export async function generateMetadata({
  params,
}: AnnotationPageProps): Promise<Metadata> {
  const { slug } = await params;
  const annotation = getAnnotation(slug);
  if (!annotation) return { title: "Benchmark annotations" };

  const title = `${annotation.name} · verifiability annotation`;
  const description = `What public evidence lets outsiders check the ${annotation.name} benchmark's validity, integrity and neutrality claims.`;
  return {
    title,
    description,
    alternates: { canonical: `/measurement-db/annotations/${slug}` },
    openGraph: {
      title,
      description,
      url: `/measurement-db/annotations/${slug}`,
    },
    twitter: { card: "summary_large_image", title, description },
  };
}

/** operator · domain · version, skipping whatever is not yet coded. */
function eyebrowParts(annotation: Annotation): string[] {
  return [annotation.operator, annotation.domain, annotation.version].filter(
    (part): part is string => Boolean(part),
  );
}

/** A labelled fact from the context items (what is hidden, who runs it, when
 *  protected material is released, whether anyone outside checked it). */
function MetaFact({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span className="rd-mono !text-[0.6rem] opacity-50">{label}</span>
      <span className="text-sm font-light">{children}</span>
    </div>
  );
}

/** One table cell. An uncoded cell says so rather than reading as "no findings". */
function CellList({
  entries,
  tone,
}: {
  entries: readonly string[];
  tone: "strength" | "concern" | "channel";
}) {
  if (entries.length === 0) {
    return (
      <span className="text-sm font-light italic opacity-40">
        Not yet coded
      </span>
    );
  }
  const markerClass = {
    strength: "bg-[var(--rd-sage-dark)]",
    concern: "border border-[var(--rd-cardinal)]",
    channel: "bg-[var(--rd-accent)]",
  }[tone];

  return (
    <ul className="flex flex-col gap-2">
      {entries.map((entry) => (
        <li key={entry} className="flex gap-2.5 text-sm font-light leading-snug">
          <span
            className={`mt-[0.45rem] size-2 shrink-0 ${markerClass}`}
            aria-hidden
          />
          <span>{entry}</span>
        </li>
      ))}
    </ul>
  );
}

const columns = [
  { key: "strengths", label: "Strengths", tone: "strength" },
  { key: "concerns", label: "Concerns", tone: "concern" },
  { key: "channels", label: "What would make it checkable", tone: "channel" },
] as const;

export default async function AnnotationReportPage({
  params,
}: AnnotationPageProps) {
  const { slug } = await params;
  const annotation = getAnnotation(slug);
  if (!annotation) notFound();

  const eyebrow = eyebrowParts(annotation);

  return (
    <main id="main-content">
      {/* ---------------------------------------------------------- header.
          Dark band so the record reads as a self-contained dossier; the
          summary table below sits on white for legibility. */}
      <section className="rd-dark-band pb-14 pt-[clamp(128px,13vw,168px)]">
        <div className="rd-container">
          <div className="min-w-0">
            {eyebrow.length > 0 ? (
              <p className="rd-mono opacity-55">{eyebrow.join(" · ")}</p>
            ) : null}
            <h1 className="rd-page-title mt-4">{annotation.name}</h1>
          </div>

          <div className="mt-7 flex flex-wrap gap-x-10 gap-y-3">
            {annotation.hidden.length > 0 ? (
              <MetaFact label="Hidden">{annotation.hidden.join(", ")}</MetaFact>
            ) : null}
            {annotation.runBy ? (
              <MetaFact label="Run by">{annotation.runBy}</MetaFact>
            ) : null}
            {annotation.release ? (
              <MetaFact label="Release">{annotation.release}</MetaFact>
            ) : null}
            {/* Only shown when an outside check was actually located, and then
                the verifiers' own pages are the value. A "none located" line
                here would duplicate the Verification row in the table below,
                and reads as a claim that nobody checked rather than that no
                check was found. */}
            {annotation.externalVerification &&
            annotation.externalVerification.sources.length > 0 ? (
              <MetaFact label="Verification">
                <span className="flex flex-wrap items-baseline gap-x-1">
                  {annotation.externalVerification.sources.map(
                    (source, index) => (
                      <span key={source.href}>
                        <a
                          href={source.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline underline-offset-4 decoration-white/40 hover:decoration-white"
                        >
                          {source.label}
                        </a>
                        {index <
                        annotation.externalVerification!.sources.length - 1
                          ? ","
                          : null}
                      </span>
                    ),
                  )}
                </span>
              </MetaFact>
            ) : null}
          </div>

          <div className="mt-9 max-w-3xl border-l-2 border-white/30 pl-5">
            {annotation.summary ? (
              <p className="rd-serif-title text-xl leading-relaxed sm:text-2xl">
                {annotation.summary}
              </p>
            ) : null}
            {/* DRAFTING AID — remove before publication. Kept beneath a written
                summary so the brief stays visible while the wording is being
                judged against it. */}
            <p
              className={`text-base font-light leading-relaxed opacity-45 ${
                annotation.summary ? "mt-5 border-t border-white/15 pt-4" : ""
              }`}
            >
              <span className="italic">
                {annotation.summary
                  ? "What this summary is for:"
                  : "Summary not yet written."}
              </span>{" "}
              Two or three sentences reading across all four claims at once:
              which of them the public evidence carries best, which is
              thinnest, and the one thing somebody should know before they rely
              on a score from this benchmark.
            </p>
          </div>
        </div>
      </section>

      {/* ----------------------------------------------------- claim summary */}
      <section className="rd-white-band rd-band">
        <div className="rd-container">
          <p className="rd-mono opacity-60">Summary</p>
          <h2 className="rd-h2 mt-4">By claim</h2>
          <p className="rd-prose-wide mt-4 text-base font-light leading-relaxed opacity-70">
            A rough summary &mdash; one row per claim, read off the coded items
            rather than reproducing them. These are judgments about available
            evidence, not about the benchmark&rsquo;s quality: a gap means
            nothing supporting it was found by the{" "}
            {annotation.cutoff ?? "corpus"} cutoff, not that the practice is
            absent. The item-by-item detail sits below.
          </p>

          <div className="mt-9 overflow-x-auto">
            <table className="w-full min-w-[46rem] border-collapse text-left">
              <thead>
                <tr className="border-b border-black/15">
                  <th scope="col" className="w-[22%] py-3 pr-5 align-bottom">
                    <span className="rd-mono !text-[0.62rem] opacity-60">
                      Claim
                    </span>
                  </th>
                  {columns.map((column) => (
                    <th
                      key={column.key}
                      scope="col"
                      className="w-[26%] py-3 pr-5 align-bottom"
                    >
                      <span className="rd-mono !text-[0.62rem] opacity-60">
                        {column.label}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {claims.map((claim) => {
                  const assessment: ClaimAssessment =
                    annotation.claims[claim.id];
                  return (
                    <tr
                      key={claim.id}
                      className="border-b border-black/10 align-top"
                    >
                      <th scope="row" className="py-6 pr-5 font-normal">
                        {/* Native <details>, so the definition needs no client
                            JS — same pattern as the chart accordions on the
                            landing page. */}
                        <details className="group">
                          <summary className="flex cursor-pointer list-none items-start gap-2 [&::-webkit-details-marker]:hidden">
                            <span className="min-w-0 flex-1">
                              <span className="rd-h3 block">{claim.label}</span>
                              <span className="mt-1.5 block text-sm font-light leading-snug opacity-60">
                                {claim.question}
                              </span>
                            </span>
                            <svg
                              width="12"
                              height="12"
                              viewBox="0 0 12 12"
                              fill="none"
                              aria-hidden
                              className="mt-2 shrink-0 opacity-45 transition-transform duration-200 group-open:rotate-180"
                            >
                              <path
                                d="M2 4.5 6 8.5l4-4"
                                stroke="currentColor"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </svg>
                          </summary>
                          <p className="mt-3 text-sm font-light leading-relaxed opacity-70">
                            {claim.definition}
                          </p>
                        </details>
                      </th>
                      {columns.map((column) => (
                        <td key={column.key} className="py-6 pr-5">
                          <CellList
                            entries={assessment[column.key]}
                            tone={column.tone}
                          />
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {annotation.readingScores ? (
            <p className="rd-prose-wide mt-8 text-sm font-light leading-relaxed">
              <span className="font-medium">Reading its scores: </span>
              <span className="opacity-75">{annotation.readingScores}</span>
            </p>
          ) : null}
        </div>
      </section>

      {/* ------------------------------------------------- fine detail (TODO)
          Shell only. The per-claim profiles and their per-item breakdowns go
          here; nothing is plotted yet. */}
      <section className="rd-sage-band rd-band">
        <div className="rd-container">
          <p className="rd-mono opacity-60">Fine detail</p>
          <h2 className="rd-h2 mt-4">By item</h2>
          <p className="rd-prose-wide mt-4 text-base font-light leading-relaxed opacity-70">
            Each claim above is coded from a set of individual items, every one
            scored on the strength of the evidence located for it. This section
            will carry a profile per claim, and open up to the item-level codes
            behind it. Claims are reported separately rather than combined into
            a single number.
          </p>
          {annotation.bands.length > 0 ? (
            <AnnotationMindmap
              benchmarkName={annotation.name}
              bands={annotation.bands}
              agreement={annotation.agreement}
            />
          ) : (
            <p className="mt-8 border-t border-black/10 pt-6 text-sm font-light italic opacity-45">
              Not yet coded.
            </p>
          )}

          {/* WORKING NOTE — remove before publication, like the summary brief
              above. Kept in view so the open conflicts between the tree, the
              table and the tallies stay visible while being settled. */}
          {annotation.reconcile && annotation.reconcile.length > 0 ? (
            <div className="mt-12 border border-dashed border-black/30 bg-black/[0.035] p-6 sm:p-7">
              <p className="rd-mono !text-[0.62rem] uppercase tracking-wider opacity-60">
                Working notes
              </p>
              <h3 className="rd-h3 mt-3">
                Open questions &mdash; {annotation.reconcile.length}
              </h3>
              <p className="rd-prose-wide mt-3 text-sm font-light leading-relaxed opacity-70">
                Where the tree, the table and the tallies don&rsquo;t line up
                yet, or where we haven&rsquo;t settled a counting rule. Not
                findings about the benchmark.
              </p>
              <ul className="mt-5 flex max-w-4xl flex-col gap-3">
                {annotation.reconcile.map((note) => (
                  <li
                    key={note}
                    className="flex gap-3 text-sm font-light leading-relaxed opacity-60"
                  >
                    <span
                      className="mt-[0.5rem] size-1.5 shrink-0 rotate-45 border border-current"
                      aria-hidden
                    />
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </section>

      {/* ------------------------------------------------------------ record */}
      <section className="rd-warm-band rd-band">
        <div className="rd-container">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div>
              <p className="rd-mono opacity-60">Sources</p>
              {annotation.links.length > 0 ? (
                <ol className="mt-5 flex flex-col gap-2">
                  {annotation.links.map((link, index) => (
                    <li
                      key={link.href}
                      className="flex gap-3 text-sm font-light leading-snug"
                    >
                      <span className="rd-mono !text-[0.6rem] tabular-nums opacity-40">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline decoration-black/25 underline-offset-4 hover:decoration-current"
                      >
                        {link.label}
                      </a>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="mt-5 text-sm font-light italic opacity-50">
                  No sources recorded yet.
                </p>
              )}
            </div>

            <div>
              <p className="rd-mono opacity-60">Annotation record</p>
              <dl className="mt-5 flex flex-col gap-3 text-sm font-light">
                <div className="flex flex-wrap gap-x-3">
                  <dt className="opacity-55">Annotators</dt>
                  <dd>
                    {annotation.annotators.length > 0
                      ? annotation.annotators.join(", ")
                      : "—"}
                  </dd>
                </div>
                <div className="flex flex-wrap gap-x-3">
                  <dt className="opacity-55">Corpus cutoff</dt>
                  <dd>{annotation.cutoff ?? "—"}</dd>
                </div>
              </dl>
            </div>
          </div>

          <p className="mt-10 border-t border-black/10 pt-6 text-sm font-light leading-relaxed opacity-60">
            <ActionLink href="/annotations" variant="ghost">
              All annotations
            </ActionLink>
          </p>
        </div>
      </section>
    </main>
  );
}
