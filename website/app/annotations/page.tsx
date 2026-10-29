import type { Metadata } from "next";
import type { Route } from "next";
import Link from "next/link";
import { PageHero } from "@/components/page-hero";
import { ScrollReveal } from "@/components/scroll-reveal";
import { SectionHeading } from "@/components/section-heading";
import { annotations, annotationStatusLabel } from "@/content/annotations";

const description =
  "Private benchmarks withhold the information outsiders would ordinarily use to scrutinize them. These records code what each benchmark hides, and what public evidence remains for checking its validity, integrity and neutrality claims.";

export const metadata: Metadata = {
  title: "Benchmark annotations | AIMS",
  description,
  alternates: { canonical: "/measurement-db/annotations" },
  openGraph: {
    title: "Benchmark annotations",
    description,
    url: "/measurement-db/annotations",
  },
  twitter: {
    card: "summary_large_image",
    title: "Benchmark annotations",
    description,
  },
};

export default function AnnotationsIndexPage() {
  return (
    <main id="main-content">
      <PageHero
        eyebrow="Measurement Data Bank"
        title="Benchmark annotations"
        description={description}
      />

      {/* ------------------------------------------------------- the records */}
      <section className="rd-white-band rd-band">
        <div className="rd-container">
          <SectionHeading
            eyebrow={`${annotations.length} benchmarks`}
            title="Annotated so far"
            description="Each record is coded by hand against a shared codebook, double-coded where possible, and tied to evidence available at a frozen corpus cutoff."
          />

          <ScrollReveal stagger>
            <ul className="mt-10 grid gap-5 sm:grid-cols-2">
              {annotations.map((annotation) => {
                const eyebrow = [annotation.operator, annotation.domain]
                  .filter(Boolean)
                  .join(" · ");
                return (
                  <li key={annotation.slug} className="reveal">
                    <Link
                      href={`/annotations/${annotation.slug}` as Route}
                      className="rd-card h-full gap-3"
                    >
                      {eyebrow ? (
                        <span className="rd-mono !text-[0.62rem] opacity-55">
                          {eyebrow}
                        </span>
                      ) : null}
                      <span className="rd-h3">{annotation.name}</span>
                      {annotation.hidden.length > 0 ? (
                        <span className="text-sm font-light leading-snug opacity-70">
                          Hidden: {annotation.hidden.join(", ")}
                        </span>
                      ) : null}
                      <span className="rd-chip mt-2 self-start">
                        {annotationStatusLabel[annotation.status]}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </ScrollReveal>

          <p className="rd-prose-wide mt-10 text-sm font-light leading-relaxed opacity-65">
            These codes describe the strength of publicly available evidence,
            not a benchmark&rsquo;s quality, fairness or validity. A gap means
            no supporting evidence was located after an adequate search &mdash;
            it does not establish that a practice is absent. Making a claim
            checkable is also not the same as someone having checked it.
          </p>
        </div>
      </section>
    </main>
  );
}
