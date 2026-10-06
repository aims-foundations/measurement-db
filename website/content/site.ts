// Trimmed copy of the main site's content/site.ts — only the pieces the shared
// header/footer need: siteConfig (footer links) and the primary navigation.
// primaryNavigation here is the build-time FALLBACK; the live menu is loaded via
// lib/nav.ts, which (from Phase 4) fetches the main site's published /nav.json so
// this sub-site's header stays in sync. Keep siteConfig in step with the main
// site when its footer URLs change.
const siteOrigin = "https://aimslab.stanford.edu";
const defaultTextbookUrl = `${siteOrigin}/textbook/`;

function ensureTrailingSlash(url: string) {
  try {
    const parsed = new URL(url, siteOrigin);

    if (!parsed.pathname.endsWith("/")) {
      parsed.pathname = `${parsed.pathname}/`;
    }

    return parsed.toString();
  } catch {
    return url.endsWith("/") ? url : `${url}/`;
  }
}

function normalizeTextbookUrl(url: string) {
  try {
    const parsed = new URL(url, siteOrigin);
    const pathname = parsed.pathname.replace(/\/+$/, "");

    if (parsed.origin === siteOrigin && pathname === "/cs321m/textbook") {
      parsed.pathname = "/textbook/";
    }

    return ensureTrailingSlash(parsed.toString());
  } catch {
    return ensureTrailingSlash(url);
  }
}

export const siteConfig = {
  name: "AIMS",
  fullName: "AI Measurement Science",
  url: siteOrigin,
  githubUrl: "https://github.com/aims-foundations/aimslab",
  description:
    "Stanford AIMS brings together research, teaching, and open resources for AI Measurement Science: the study of how AI systems are evaluated and understood.",
  shortDescription:
    "Building principled foundations for measuring AI systems through measurement theory, machine learning, and evaluation science.",
  textbookUrl: normalizeTextbookUrl(
    process.env.NEXT_PUBLIC_TEXTBOOK_URL || defaultTextbookUrl,
  ),
  textbookPdfUrl: process.env.NEXT_PUBLIC_TEXTBOOK_PDF_URL ?? "",
  syllabusUrl:
    "https://docs.google.com/document/d/1sE6VC7BZ0QK9gF8zD2904CJy-GRebW0dg3p0oi4jGEs/edit?tab=t.0",
  competitionUrl: "/competition",
  stanfordCourseUrl: "https://web.stanford.edu/class/cs321m/",
  discordUrl:
    process.env.NEXT_PUBLIC_DISCORD_INVITE_URL ??
    "https://discord.gg/UVseUQ983v",
} as const;

// A nav entry is either a leaf link ({ href, label }) or a dropdown group
// ({ label, children: [...] }). A `proxied` leaf is served by a Next rewrite on
// the main site rather than an app route, so it is always rendered as a plain
// full-navigation anchor.
export type NavLeaf = { href: string; label: string; proxied?: boolean };
export type NavGroup = { label: string; children: readonly NavLeaf[] };
export type NavItem = NavLeaf | NavGroup;

// Fallback navigation, mirrored from the main site's single source of truth.
// lib/nav.ts prefers the main site's published /nav.json and falls back to this.
export const primaryNavigation: readonly NavItem[] = [
  { href: "/research", label: "Research" },
  { href: "/measurement-db", label: "Data & Software", proxied: true },
  {
    label: "Education",
    children: [
      { href: siteConfig.textbookUrl, label: "Textbook" },
      { href: "/cs321m", label: "Course" },
    ],
  },
  {
    label: "Community",
    children: [
      { href: "/competition", label: "Competition" },
      { href: "/workshop", label: "Workshop" },
      { href: "/seminars", label: "Seminars" },
    ],
  },
  { href: "/blog", label: "Blog" },
];
