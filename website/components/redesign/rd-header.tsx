"use client";

/* This site is served under a basePath and proxied from the main AIMS site.
   The logo and nav links are intentionally plain <a> tags so they escape
   basePath and navigate to the parent site's pages (e.g. "/", "/research"),
   which are NOT routes of this sub-site. next/link would wrongly prefix them
   with /measurement-db. */

import { type MouseEvent, useEffect, useState } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { SiteSearchDialog } from "@/components/site-search-dialog";
import type { NavItem } from "@/content/site";
import { mainSiteHref } from "@/lib/main-site-url";

import { RdLogo } from "./rd-logo";

function RdCaret() {
  return (
    <svg
      className="rd-nav-caret"
      width="11"
      height="7"
      viewBox="0 0 11 7"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M1 1.5 5.5 6l4.5-4.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// navItems is loaded server-side (lib/nav.ts) and passed in, so this sub-site's
// menu can track the main site's published nav manifest (Phase 4).
export function RdHeader({ navItems }: { navItems: readonly NavItem[] }) {
  const [hidden, setHidden] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);

  useEffect(() => {
    let lastY = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      setScrolled(y > 24);
      setHidden(y > 120 && y > lastY);
      lastY = y;
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const openSearch = (event: MouseEvent<HTMLAnchorElement>) => {
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      typeof HTMLDialogElement === "undefined"
    ) {
      return;
    }

    event.preventDefault();
    setSearchOpen(true);
  };

  return (
    <>
      <header
        className={`rd-header rd-header--solid ${
          scrolled || mobileOpen || searchOpen ? "rd-header--scrolled" : ""
        } ${hidden && !mobileOpen && !searchOpen ? "rd-header--hidden" : ""}`}
      >
        <div className="rd-container flex items-center justify-between py-4">
          {/* Plain anchor (not next/link) so basePath is NOT prepended — the
              AIMS wordmark returns to the main site home. */}
          <a
            href={mainSiteHref("/")}
            className="rd-wordmark"
            aria-label="AIMS — AI Measurement Science home"
          >
            <RdLogo height={40} compactBelowSm />
          </a>

          <div className="hidden items-center gap-2 lg:flex">
            <nav aria-label="Primary" className="flex items-center gap-2">
              {navItems.map((item) =>
                "children" in item ? (
                  <div key={item.label} className="rd-nav-item">
                    <button type="button" className="rd-nav-link">
                      {item.label}
                      <RdCaret />
                    </button>
                    <div className="rd-dropdown">
                      {item.children.map((child) => (
                        <a key={child.label} href={mainSiteHref(child.href)}>
                          {child.label}
                        </a>
                      ))}
                    </div>
                  </div>
                ) : (
                  <a
                    key={item.label}
                    href={mainSiteHref(item.href)}
                    className="rd-nav-link"
                  >
                    {item.label}
                  </a>
                ),
              )}
            </nav>

            <a
              href={mainSiteHref("/search")}
              className="rd-search-trigger"
              aria-label="Search AIMS"
              aria-haspopup="dialog"
              aria-controls="rd-site-search-dialog"
              aria-expanded={searchOpen}
              title="Search AIMS"
              onClick={openSearch}
            >
              <MagnifyingGlass size={20} weight="bold" aria-hidden="true" />
            </a>
          </div>

          <button
            type="button"
            className="rd-mono inline-flex min-h-11 min-w-11 items-center justify-center lg:hidden"
            aria-expanded={mobileOpen}
            aria-controls="rd-mobile-navigation"
            aria-label={
              mobileOpen ? "Close navigation menu" : "Open navigation menu"
            }
            onClick={() => setMobileOpen((open) => !open)}
          >
            {mobileOpen ? "Close" : "Menu"}
          </button>
        </div>

        {mobileOpen && (
          <nav
            id="rd-mobile-navigation"
            aria-label="Mobile"
            className="rd-container max-h-[calc(100dvh-72px)] overflow-y-auto border-t border-black/10 pb-6 pt-2 lg:hidden"
          >
            <a
              href={mainSiteHref("/search")}
              className="flex min-h-11 items-center gap-3 border-b border-black/10 py-2 text-lg font-light"
              aria-haspopup="dialog"
              aria-controls="rd-site-search-dialog"
              aria-expanded={searchOpen}
              onClick={openSearch}
            >
              <MagnifyingGlass size={20} weight="bold" aria-hidden="true" />
              Search AIMS
            </a>
            {navItems.map((item) =>
              "children" in item ? (
                <div key={item.label} className="py-2">
                  <div className="rd-mono pb-1 opacity-60">{item.label}</div>
                  {item.children.map((child) => (
                    <a
                      key={child.label}
                      href={mainSiteHref(child.href)}
                      className="flex min-h-11 items-center pl-4 text-lg font-light"
                    >
                      {child.label}
                    </a>
                  ))}
                </div>
              ) : (
                <a
                  key={item.label}
                  href={mainSiteHref(item.href)}
                  className="flex min-h-11 items-center text-lg font-light"
                >
                  {item.label}
                </a>
              ),
            )}
          </nav>
        )}
      </header>
      <SiteSearchDialog
        open={searchOpen}
        onOpenChange={setSearchOpen}
        onNavigate={() => setMobileOpen(false)}
      />
    </>
  );
}
