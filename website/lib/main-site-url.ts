import { siteConfig } from "@/content/site";

export const MAIN_SITE_ORIGIN = (
  process.env.NEXT_PUBLIC_MAIN_ORIGIN ?? siteConfig.url
).replace(/\/+$/, "");

/**
 * Qualify a main-site path so shared-shell links also work when this subsite is
 * previewed on its standalone origin instead of through the public proxy.
 */
export function mainSiteHref(href: string) {
  return href.startsWith("/") ? `${MAIN_SITE_ORIGIN}${href}` : href;
}
