import { getBenchmarkDetail } from "@/content/benchmark-details";
import { getBenchmarkIrt } from "@/content/benchmark-irt";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { ActionLink } from "@/components/action-link";
import { BenchmarkCreditPanel } from "@/components/benchmark-credit-panel";
import { SaturationTag } from "@/components/saturation-tag";
import { BenchmarkMatrix } from "@/components/matrix-viewer";
import { PageHero } from "@/components/page-hero";
import { ScrollReveal } from "@/components/scroll-reveal";
import { SectionHeading } from "@/components/section-heading";
import { getBenchmarkAttribution } from "@/content/benchmark-attributions";
import {
  getBenchmark,
  getBenchmarkCategory,
  measurementDb,
} from "@/content/measurement-db";

type DetailPageProps = {
  params: Promise<{ slug: string }>;
};

// Map a category to the PageHero eyebrow accent (subset of brand colors).
const categoryAccent: Record<
  string,
  "cardinal" | "lagunita" | "palo-alto" | "poppy" | "plum" | "default"
> = {
  safety: "cardinal",
  cybersecurity: "cardinal",
  software_engineering: "palo-alto",
  agents_and_tool_use: "poppy",
  nlp_task: "plum",
  knowledge: "lagunita",
  reasoning: "lagunita",
};

export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: DetailPageProps): Promise<Metadata> {
  const { slug } = await params;
  const benchmark = getBenchmark(slug);
  if (!benchmark) return { title: "measurement-db" };

  const title = `${benchmark.name} · measurement-db`;
  const description = benchmark.description;
  return {
    title,
    description,
    alternates: { canonical: `/measurement-db/${slug}` },
    openGraph: { title, description, url: `/measurement-db/${slug}` },
    twitter: { card: "summary_large_image", title, description },
  };
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="inline-flex items-baseline gap-1.5 rounded-full border border-black/15 px-3 py-1 text-xs">
      <span className="font-mono">{value}</span>
      <span className="rd-mono !text-[0.6rem] opacity-60">{label}</span>
    </div>
  );
}

export default async function BenchmarkDetailPage({ params }: DetailPageProps) {
  const { slug } = await params;
  const benchmark = getBenchmark(slug);
  if (!benchmark) notFound();
  const detail = getBenchmarkDetail(slug);
  const irt = getBenchmarkIrt(slug);
  const attribution = getBenchmarkAttribution(slug);

  if (!detail) {
    notFound();
  }

  const primaryCategory = benchmark.categories[0];
  const category = getBenchmarkCategory(primaryCategory);
  const accent = categoryAccent[primaryCategory] ?? "default";
  // Build script link is temporarily hidden — re-enable with the "Build script" ActionLink below.
  // const buildScriptHref = `${measurementDb.repoHref}/blob/main/${slug}/build.py`;

  return (
    <main id="main-content">
      <PageHero
        eyebrow={category?.label ?? "Benchmark"}
        eyebrowAccent={accent}
        title={benchmark.name}
        description={detail.description ?? benchmark.description}
        aside={
          attribution ? (
            <BenchmarkCreditPanel
              attribution={attribution}
              originalSourceHref={benchmark.code}
              className="min-w-0 !bg-white/90 !p-5 sm:!p-6"
            />
          ) : (
            <div className="rd-card min-w-0 gap-4 !bg-white/90 !p-5 sm:!p-6">
              <h2 className="rd-h3">Source</h2>
              <div className="flex flex-wrap gap-3">
                {benchmark.paper ? (
                  <ActionLink href={benchmark.paper} external>
                    Paper
                  </ActionLink>
                ) : null}
                {benchmark.code ? (
                  <ActionLink href={benchmark.code} external>
                    Original source
                  </ActionLink>
                ) : null}
              </div>
            </div>
          )
        }
      >
        <div className="flex min-w-0 flex-col gap-6">
          <div className="flex min-w-0 flex-wrap gap-1.5">
            <StatChip
              label="items"
              value={detail.stats.items.toLocaleString("en-US")}
            />
            <StatChip
              label="subjects"
              value={detail.stats.subjects.toLocaleString("en-US")}
            />
            {detail.license ? (
              <StatChip label="license" value={detail.license} />
            ) : null}
            {detail.domain.map((d) => (
              <StatChip key={d} label="domain" value={d} />
            ))}
            {detail.modality.map((m) => (
              <StatChip key={m} label="modality" value={m} />
            ))}
            {benchmark.itemResponses ? (
              <div className="inline-flex items-baseline rounded-full border border-[var(--palo-alto)]/40 bg-[var(--palo-alto)]/5 px-3 py-1 text-xs font-medium text-[var(--palo-alto)]">
                item-level responses released
              </div>
            ) : null}
            <SaturationTag saturation={benchmark.saturation} variant="detail" />
          </div>

          <div className="flex flex-wrap items-start gap-3">
            {/* Build script link — temporarily hidden. Uncomment to restore (also un-comment `buildScriptHref` above).
            <ActionLink href={buildScriptHref} variant="secondary" external>
              Build script
            </ActionLink>
            */}
            <ActionLink
              href={measurementDb.datasetHref}
              variant="secondary"
              className="min-h-11"
              external
            >
              Full data on Hugging Face
            </ActionLink>
            <ActionLink href="/" variant="secondary" className="min-h-11">
              Back to the gallery
            </ActionLink>
          </div>
        </div>
      </PageHero>

      {/* Response matrix */}
      <section className="rd-band rd-white-band">
        {/* No max-w-5xl here, unlike the prose sections. A matrix is a figure,
            not a column of text: its width is what the item axis gets, and at
            1024px a benchmark with many column blocks is squeezed to unreadable
            slivers (mmlu's 57 tasks come out ~12px each). rd-container's own
            1334px gives ~1042px to the blocks — 17 of them at a legible 60px,
            which covers lawbench and mme. The note below shares the figure's
            wide measure (rd-prose-wide): several curation notes run half a
            dozen sentences, and at the 48ch reading measure they broke into a
            ragged ten-line column beside empty space. */}
        <div className="rd-container">
          <ScrollReveal>
            <SectionHeading eyebrow="Response matrix" />
          </ScrollReveal>

          {detail.note ? (
            <ScrollReveal>
              <p className="rd-prose-wide mt-6 whitespace-pre-line text-sm font-light opacity-70">
                {detail.note}
              </p>
            </ScrollReveal>
          ) : null}

          <ScrollReveal>
            <figure className="rd-card mt-8 p-4 sm:p-6">
              <BenchmarkMatrix
                key={detail.slug}
                slug={detail.slug}
                name={benchmark.name}
                irt={irt}
              />
              <figcaption className="mt-4 space-y-2 text-xs font-light opacity-70">
                <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
                  {detail.stats.meanResponse ===
                  null ? null : detail.categories &&
                    detail.categories.length ? (
                    detail.categories.map((c) => (
                      <span
                        key={c.value}
                        className="inline-flex items-center gap-1.5"
                      >
                        <span
                          className="inline-block h-3 w-3 rounded-sm"
                          style={{ backgroundColor: c.color }}
                        />
                        {c.label}
                      </span>
                    ))
                  ) : detail.isBinary ? (
                    // Binary matrices are recoloured at display time (see
                    // MatrixViewer's COLOR_SWAP filter): the "1" cells show
                    // blue, the "0" cells show red. The swatches match.
                    <>
                      <span className="inline-flex items-center gap-1.5">
                        <span className="inline-block h-3 w-3 rounded-sm bg-[#1f77b4]" />
                        {detail.binaryLabels
                          ? `${detail.binaryLabels.one} (1)`
                          : "Correct (1)"}
                      </span>
                      <span className="inline-flex items-center gap-1.5">
                        <span className="inline-block h-3 w-3 rounded-sm bg-[#d62728]" />
                        {detail.binaryLabels
                          ? `${detail.binaryLabels.zero} (0)`
                          : "Incorrect (0)"}
                      </span>
                    </>
                  ) : (
                    <span className="inline-flex items-center gap-2">
                      <span>low</span>
                      <span
                        aria-hidden="true"
                        className="inline-block h-3 w-24 rounded-sm"
                        style={{
                          backgroundImage:
                            "linear-gradient(to right, #1f77b4, #f2f2f2, #d62728)",
                        }}
                      />
                      <span>high</span>
                    </span>
                  )}
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className={`inline-block h-3 w-3 rounded-sm ${
                        detail.isBinary &&
                        !(detail.categories && detail.categories.length)
                          ? "border border-[var(--line)] bg-white"
                          : "bg-[#9e9e9e]"
                      }`}
                    />
                    {detail.stats.meanResponse === null
                      ? "Grade unavailable"
                      : "Unobserved"}
                  </span>
                </div>
                <p>
                  <span className="rd-mono !text-[0.6rem] opacity-60">
                    Scale:
                  </span>{" "}
                  <span className="font-mono">{detail.scaleLabel}</span>
                </p>
              </figcaption>
            </figure>
          </ScrollReveal>
        </div>
      </section>
    </main>
  );
}
