import { primaryNavigation, type NavItem } from "@/content/site";
import { MAIN_SITE_ORIGIN } from "@/lib/main-site-url";

// The main AIMS site publishes its primary navigation at <origin>/nav.json (a
// route handler derived from its single source of truth). We fetch that at build
// time so this sub-site's header auto-syncs: when the main site's menu changes,
// this menu follows on the next rebuild. NEXT_PUBLIC_MAIN_ORIGIN can override the
// origin (e.g. to point a preview deploy at a preview of the main site).
const NAV_URL = `${MAIN_SITE_ORIGIN}/nav.json`;

function isNavItemArray(value: unknown): value is NavItem[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (item) =>
        typeof item === "object" &&
        item !== null &&
        ("href" in item || "children" in item),
    )
  );
}

// Loads the primary navigation for the shared header. Prefers the main site's
// published manifest; falls back to the local copy (content/site.ts) if the
// fetch fails or returns an unexpected shape, so a build never breaks on it.
export async function getNavItems(): Promise<readonly NavItem[]> {
  try {
    // force-cache: fetched once during `next build` and baked into the static
    // output, so the menu tracks the main site as of this site's last build.
    const res = await fetch(NAV_URL, { cache: "force-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: unknown = await res.json();
    if (!isNavItemArray(data)) throw new Error("unexpected shape");
    return data;
  } catch (err) {
    console.warn(
      `[nav] using local fallback navigation (could not load ${NAV_URL}): ${String(
        err,
      )}`,
    );
    return primaryNavigation;
  }
}
