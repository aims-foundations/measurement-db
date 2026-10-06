/* This site is served under a basePath and proxied from the main AIMS site.
   Footer links point to the parent site's pages (e.g. "/blog", "/cs321m")
   and are intentionally plain <a> tags so they escape basePath. next/link would
   wrongly prefix them with /measurement-db, where those routes don't exist. */
import { siteConfig } from "@/content/site";
import { mainSiteHref } from "@/lib/main-site-url";
import { RdLogo } from "./rd-logo";

const getInvolved = [
  { href: "/cs321m", label: "CS321M Course" },
  { href: siteConfig.textbookUrl, label: "Textbook" },
  { href: "/competition", label: "Competition" },
  { href: "/workshop", label: "Workshop" },
  { href: "/seminars", label: "Seminars" },
  { href: "/news", label: "News archive" },
  { href: "/blog", label: "Blog" },
];

const stayInTouch = [
  { href: siteConfig.discordUrl, label: "Discord" },
  { href: siteConfig.githubUrl, label: "GitHub" },
  { href: "/news/feed.xml", label: "Atom feed" },
];

const stanfordLinks = [
  { href: "https://www.stanford.edu", label: "Stanford Home" },
  { href: "https://visit.stanford.edu/plan/", label: "Maps & Directions" },
  { href: "/search", label: "Search AIMS" },
  { href: "https://www.stanford.edu/search/", label: "Search Stanford" },
  { href: "https://emergency.stanford.edu", label: "Emergency Info" },
];

const legalLinks = [
  {
    href: "https://www.stanford.edu/site/terms/",
    label: "Terms of Use",
  },
  {
    href: "https://www.stanford.edu/site/privacy/",
    label: "Privacy",
  },
  {
    href: "https://uit.stanford.edu/security/copyright-infringement",
    label: "Copyright",
  },
  {
    href: "https://adminguide.stanford.edu/chapters/guiding-policies/1-5-4-ownership-and-use-stanford-trademarks-and-images",
    label: "Trademarks",
  },
  {
    href: "https://www.stanford.edu/site/accessibility/",
    label: "Accessibility",
  },
];

export function RdFooter() {
  return (
    <footer>
      <div className="rd-darkred-band">
        <div className="rd-container grid gap-12 py-14 md:grid-cols-3">
          <div className="space-y-5">
            <div className="space-y-2">
              <RdLogo height={48} />
              <p className="text-xs font-light opacity-70">
                <a href="https://hai.stanford.edu/" className="rd-footer-link">
                  A center within Stanford HAI
                </a>
              </p>
            </div>
            <p className="text-sm font-light leading-relaxed opacity-85">
              Stanford University
              <br />
              353 Jane Stanford Way
              <br />
              Stanford, CA 94305
            </p>
          </div>

          <div className="space-y-4">
            <div className="rd-mono opacity-60">Get Involved</div>
            <ul className="space-y-2.5 text-sm font-light">
              {getInvolved.map((link) => (
                <li key={link.label}>
                  <a href={mainSiteHref(link.href)} className="rd-footer-link">
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-4">
            <div className="rd-mono opacity-60">Stay in Touch</div>
            <ul className="space-y-2.5 text-sm font-light">
              {stayInTouch.map((link) => (
                <li key={link.label}>
                  <a
                    href={mainSiteHref(link.href)}
                    className="rd-footer-link"
                    rel="noopener noreferrer"
                    target="_blank"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="rd-dark-band">
        <div className="rd-container space-y-4 py-8">
          <ul className="flex flex-wrap gap-x-6 gap-y-2 text-xs font-light">
            {stanfordLinks.map((link) => (
              <li key={link.label}>
                <a href={mainSiteHref(link.href)} className="rd-footer-link">
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
          <ul className="flex flex-wrap gap-x-6 gap-y-2 text-xs font-light opacity-70">
            {legalLinks.map((link) => (
              <li key={link.label}>
                <a href={link.href} className="rd-footer-link">
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
          <p className="text-xs font-light opacity-60">
            © Stanford University. Stanford, California 94305.
          </p>
          <p className="text-xs font-light opacity-60">
            Website designed by{" "}
            <a
              href="https://ai.stanford.edu/~nntruong/"
              className="rd-footer-link underline underline-offset-2"
              rel="noopener noreferrer"
              target="_blank"
            >
              Nhi Truong
            </a>
            .
          </p>
        </div>
      </div>
    </footer>
  );
}
