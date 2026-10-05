"use client";

import { useState } from "react";

import { CompanyLogo } from "@/components/company-logo";
import type { LeaderboardEntry } from "@/content/model-leaderboard";

const MOBILE_VISIBLE_ROWS = 8;

// Horizontal-bar leaderboard: model providers ranked by how many distinct
// benchmark items each provider's most-measured model has been asked across
// the shelf whose entries are passed in (see content/section-charts.ts).
// Bars are CSS widths scaled to the top provider. Phones start with the top
// eight rows so one expanded chart does not consume several screens.
export function ModelLeaderboard({
  entries,
}: {
  entries: ReadonlyArray<LeaderboardEntry>;
}) {
  const [showAllOnMobile, setShowAllOnMobile] = useState(false);

  return (
    <div className="panel px-4 py-6 sm:px-8 sm:py-8">
      {/* Column header */}
      <div className="mb-4 flex items-center gap-3 border-b border-[var(--line)] pb-3 sm:gap-4">
        <span className="w-6 text-right font-mono text-xs uppercase tracking-wider text-[var(--muted)]">
          #
        </span>
        <span className="text-xs uppercase tracking-wider text-[var(--muted)]">
          <span className="sm:hidden">AI subject</span>
          <span className="hidden sm:inline">
            Provider · most-measured AI subject
          </span>
        </span>
        <span className="ml-auto text-xs uppercase tracking-wider text-[var(--muted)]">
          Items asked
        </span>
      </div>

      <ol
        className={`space-y-3.5 ${
          showAllOnMobile ? "" : "rd-mobile-top-eight"
        }`}
      >
        {entries.map((entry) => (
          <li
            key={entry.company}
            className="flex min-w-0 items-center gap-2 sm:gap-4"
          >
            <span className="w-5 shrink-0 text-right font-mono text-sm text-[var(--muted)] sm:w-6">
              {entry.rank}
            </span>

            <CompanyLogo company={entry.company} size={22} />

            <div className="min-w-0 flex-1 leading-tight sm:w-44 sm:flex-none">
              <p className="truncate text-sm font-medium text-[var(--ink)]">
                {entry.company}
              </p>
              <p className="truncate text-xs text-[var(--muted)]">
                {entry.flagship}
              </p>
            </div>

            {/* Bar track + fill, scaled to the top provider. One flat hue —
                the ranking is a single magnitude, so the bar carries no
                second encoding; cardinal ties the figure to the section's
                opening band. */}
            <div className="hidden h-2.5 flex-1 overflow-hidden rounded-full bg-black/[0.08] sm:block">
              <div
                className="h-full rounded-full bg-[var(--cardinal)]"
                style={{ width: `${entry.fillPct}%` }}
              />
            </div>

            <span className="ml-auto min-w-[3.25rem] shrink-0 text-right font-mono text-sm tabular-nums text-[var(--ink)] sm:ml-0 sm:w-20">
              {entry.itemsLabel}
            </span>
          </li>
        ))}
      </ol>

      {entries.length > MOBILE_VISIBLE_ROWS ? (
        <div className="mt-4 flex items-center justify-between gap-3 text-xs text-[var(--muted)] sm:hidden">
          <span>
            Showing {showAllOnMobile ? "all" : `top ${MOBILE_VISIBLE_ROWS}`} of{" "}
            {entries.length}
          </span>
          <button
            type="button"
            aria-expanded={showAllOnMobile}
            onClick={() => setShowAllOnMobile((current) => !current)}
            className="min-h-11 shrink-0 font-medium text-[var(--lagunita)] underline underline-offset-2"
          >
            {showAllOnMobile ? `Show top ${MOBILE_VISIBLE_ROWS}` : "Show all"}
          </button>
        </div>
      ) : null}
    </div>
  );
}
