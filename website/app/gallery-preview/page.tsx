import Link from "next/link";
import type { Route } from "next";
import { galleryRecords } from "@/content/gallery-records";
import { PageHero } from "@/components/page-hero";

export default function GalleryPreviewPage() {
  const entries = Object.values(galleryRecords);
  return (
    <main id="main-content">
      <PageHero
        eyebrow="Gallery preview"
        title="Explore the benchmark tables"
        description="Models, tasks, and every recorded attempt in one consistent view."
      />
      <section className="rd-band rd-white-band">
        <div className="rd-container grid gap-6 md:grid-cols-2">
          {entries.map(({ manifest }) => (
            <Link
              key={manifest.benchmark.benchmark_id}
              href={`/${manifest.benchmark.benchmark_id}` as Route}
              className="rd-card block p-6 hover:border-[var(--palo-alto)]"
            >
              <h2 className="rd-h3">{manifest.benchmark.name}</h2>
              <p className="mt-3 text-sm leading-relaxed text-[var(--muted)]">
                {manifest.benchmark.description}
              </p>
              <p className="mt-4 font-mono text-xs">
                {manifest.tables.subjects?.rows.toLocaleString() ?? 0} models ·{" "}
                {manifest.tables.items?.rows.toLocaleString() ?? 0} tasks
              </p>
              <span className="mt-5 inline-block text-sm font-medium text-[var(--palo-alto)]">
                Explore observations →
              </span>
            </Link>
          ))}
          {entries.length === 0 && (
            <p>No benchmark tables have been prepared for this preview.</p>
          )}
        </div>
      </section>
    </main>
  );
}
