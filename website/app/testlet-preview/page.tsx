import type { Metadata } from "next";
import { PageHero } from "@/components/page-hero";
import { ActionLink } from "@/components/action-link";
import { AlignedPreview } from "@/components/aligned-matrix/explorer";
import { getGalleryRecords } from "@/content/gallery-records";

export const metadata: Metadata = {
  title: "ProgramBench · Continuous matrix preview",
  robots: { index: false, follow: false },
};

export default function TestletPreviewPage() {
  const entry = getGalleryRecords("programbench");
  return (
    <main id="main-content">
      <PageHero
        eyebrow="Interactive preview"
        eyebrowAccent="palo-alto"
        title="ProgramBench"
        description="Explore every behavioral test in one continuous view. Zoom from the full benchmark into individual outcomes, with related tests grouped by program."
      >
        <ActionLink href="/programbench" variant="secondary">
          Compare with the current page
        </ActionLink>
      </PageHero>
      <section className="rd-band rd-white-band">
        <div className="rd-container">
          <figure className="rd-card p-3 sm:p-6">
            {entry ? (
              <AlignedPreview entry={entry} />
            ) : (
              <p>ProgramBench records are unavailable.</p>
            )}
          </figure>
        </div>
      </section>
    </main>
  );
}
