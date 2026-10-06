import type { ReactNode } from "react";
import { ScrollReveal } from "@/components/scroll-reveal";

/* ============================================================
   PageHero — shared subpage header.

   The canonical "page top" treatment used across /research,
   /cs321m, /workshop, /torch_measure, etc. Restyled to match the
   redesign language: a beige band that sits under the fixed white header,
   a mono eyebrow, an oversized semibold grotesk title, and an
   optional `children` slot for page-specific extras (CTAs, chips).
   Pages that need a compact companion panel can also opt into an
   `aside`, which stacks below the main content on smaller screens.

   The `eyebrowAccent` prop is preserved for API compatibility but the
   redesign uses a single restrained accent, so it's no longer color-mapped.
   ============================================================ */

type EyebrowAccent =
  | "default"
  | "cardinal"
  | "lagunita"
  | "palo-alto"
  | "poppy"
  | "plum";

type PageHeroProps = {
  eyebrow?: string;
  eyebrowAccent?: EyebrowAccent;
  title: string;
  description?: string;
  /** Optional content rendered below the description — CTAs, metadata, etc. */
  children?: ReactNode;
  /** Optional companion content rendered beside the main hero content on desktop. */
  aside?: ReactNode;
};

export function PageHero({
  eyebrow,
  title,
  description,
  children,
  aside,
}: PageHeroProps) {
  const mainContent = (
    <>
      {eyebrow ? <p className="rd-mono opacity-60">{eyebrow}</p> : null}
      <h1 className={`rd-page-title ${eyebrow ? "mt-5" : ""}`}>{title}</h1>
      {description ? (
        <p className="rd-lead rd-prose mt-6 opacity-80">{description}</p>
      ) : null}
      {children ? <div className="mt-9">{children}</div> : null}
    </>
  );

  return (
    <section className="rd-page-hero">
      <div className="rd-container relative">
        {aside ? (
          <ScrollReveal>
            <div className="grid min-w-0 gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(20rem,0.72fr)] lg:items-start lg:gap-12">
              <div className="min-w-0">{mainContent}</div>
              <aside className="min-w-0 lg:self-start">{aside}</aside>
            </div>
          </ScrollReveal>
        ) : (
          <ScrollReveal>{mainContent}</ScrollReveal>
        )}
      </div>
    </section>
  );
}
