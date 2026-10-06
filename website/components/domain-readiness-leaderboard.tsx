"use client";

import { useState } from "react";

import type { DomainReadinessEntry } from "@/content/domain-readiness";

const MOBILE_VISIBLE_ROWS = 8;

// Horizontal-bar leaderboard: benchmark domains ranked by the total number of
// distinct evaluation items across the shelf whose entries are passed in (see
// content/section-charts.ts). Bars are CSS widths scaled to the most-measured
// domain, tinted with each domain's brand accent. Shows the top `initialCount`
// by default and reveals the rest via a "Show more" toggle.
export function DomainReadinessLeaderboard({
  entries,
  initialCount = 15,
}: {
  entries: ReadonlyArray<DomainReadinessEntry>;
  initialCount?: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const [mobileExpanded, setMobileExpanded] = useState(false);
  const more = entries.length - initialCount;
  const mobileMore = entries.length - MOBILE_VISIBLE_ROWS;

  return (
    <div className="panel px-4 py-6 sm:px-8 sm:py-8">
      {/* Column header */}
      <div className="mb-4 flex items-center gap-3 border-b border-[var(--line)] pb-3 sm:gap-4">
        <span className="w-6 text-right font-mono text-xs uppercase tracking-wider text-[var(--muted)]">
          #
        </span>
        <span className="text-xs uppercase tracking-wider text-[var(--muted)]">
          Domain
        </span>
        <span className="ml-auto text-xs uppercase tracking-wider text-[var(--muted)]">
          Items
        </span>
      </div>

      <ol
        className={`space-y-3.5 ${
          expanded ? "" : "rd-domain-desktop-limited"
        } ${mobileExpanded ? "" : "rd-mobile-top-eight"}`}
      >
        {entries.map((entry) => (
          <li
            key={entry.id}
            className="flex min-w-0 items-center gap-2 sm:gap-4"
          >
            <span className="w-5 shrink-0 text-right font-mono text-sm text-[var(--muted)] sm:w-6">
              {entry.rank}
            </span>

            <span
              className="h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color }}
              aria-hidden
            />

            <div className="min-w-0 flex-1 leading-tight sm:w-44 sm:flex-none">
              <p className="truncate text-sm font-medium text-[var(--ink)]">
                {entry.label}
              </p>
              <p className="truncate text-xs text-[var(--muted)]">
                {entry.benchmarksLabel}
              </p>
            </div>

            {/* Bar track + fill, scaled to the most-measured domain. Flat
                domain hue — identity lives in the color, magnitude in the
                length; a gradient adds nothing to either. */}
            <div className="hidden h-2.5 flex-1 overflow-hidden rounded-full bg-black/[0.08] sm:block">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${entry.fillPct}%`,
                  backgroundColor: entry.color,
                }}
              />
            </div>

            <span className="ml-auto min-w-[3.25rem] shrink-0 text-right font-mono text-sm tabular-nums text-[var(--ink)] sm:ml-0 sm:w-20">
              {entry.itemsLabel}
            </span>
          </li>
        ))}
      </ol>

      {more > 0 ? (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          className="mt-5 hidden text-xs font-medium text-[var(--digital-blue)] underline decoration-[var(--digital-blue)]/30 underline-offset-[3px] transition-colors hover:text-[var(--lagunita)] sm:inline-flex"
        >
          {expanded ? "Show fewer" : `Show ${more} more domains`}
        </button>
      ) : null}
      {mobileMore > 0 ? (
        <div className="mt-4 flex items-center justify-between gap-3 text-xs text-[var(--muted)] sm:hidden">
          <span>
            Showing {mobileExpanded ? "all" : `top ${MOBILE_VISIBLE_ROWS}`} of{" "}
            {entries.length}
          </span>
          <button
            type="button"
            aria-expanded={mobileExpanded}
            onClick={() => setMobileExpanded((current) => !current)}
            className="min-h-11 shrink-0 font-medium text-[var(--lagunita)] underline underline-offset-2"
          >
            {mobileExpanded
              ? `Show top ${MOBILE_VISIBLE_ROWS}`
              : `Show ${mobileMore} more`}
          </button>
        </div>
      ) : null}
    </div>
  );
}
