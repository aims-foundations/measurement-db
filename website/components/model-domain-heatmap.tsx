"use client";

import { useMemo, useState } from "react";
import { createPortal } from "react-dom";

import { CompanyLogo } from "@/components/company-logo";
import {
  cellColor,
  type HeatmapRow,
  type ModelDomainHeatmapData,
} from "@/content/model-domain-heatmap";

// Sample points for the legend's color ramp, spanning the same [0,1] the cells
// use so the swatches are literally the cell colors, not an approximation.
const SCALE_STOPS = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1];

// Model × domain score heatmap. Rows are MODELS, columns are benchmark
// DOMAINS, each cell the model's average score (accuracy) across that domain's
// binary benchmarks — light (low) → dark Palo Alto (high), hatched when
// unmeasured. Interactive: filter models by name, sort model rows by any
// domain column (or by total coverage), and a cursor-following hover label
// per cell.
//
// The bank has ~800 models, so models are the SCROLLING axis: the grid scrolls
// vertically inside a capped viewport with a sticky domain header + sticky model
// column, so every model is reachable without a "show more" toggle.

const NAME_W = "var(--heatmap-name-width)"; // responsive sticky subject column
const COL_W = 48; // px per domain column
const HEADER_H = 132; // px reserved for the angled domain header
// The angled header labels rise up-and-right past the last column; without this
// the rightmost ones ("Cybersecurity", …) are clipped by the scroll container.
// Included in the grid's width so the panel (w-fit) grows to hold them.
const LABEL_PAD = 90; // px of headroom to the right of the last column
const MOBILE_VISIBLE_ROWS = 15;

/** Percent label, or "" for an unmeasured (null) cell. */
function pct(mean: number | null): string {
  return mean === null ? "" : `${Math.round(mean * 100)}`;
}

/** White text once the sequential ramp turns dark, ink on the light half. */
function textColor(mean: number | null): string {
  if (mean === null) return "transparent";
  return mean > 0.65 ? "#fff" : "var(--ink)";
}

// "Not measured" cells get a diagonal hatch instead of a flat gray, so they read
// as EMPTY rather than being mistaken for a mid-range (~50%) score, whose color
// sits near the ramp's near-white midpoint. Distinct in kind, not just in shade.
const EMPTY_CELL: React.CSSProperties = {
  backgroundColor: "#fbfbfb",
  backgroundImage:
    "repeating-linear-gradient(45deg, rgba(0,0,0,0.07) 0, rgba(0,0,0,0.07) 1px, transparent 1px, transparent 5px)",
};

export function ModelDomainHeatmap({
  data,
  metricLabel = "Mean binary score",
}: {
  data: ModelDomainHeatmapData;
  metricLabel?: string;
}) {
  // Destructured to the names the body has always used — every reference
  // below stays shelf-agnostic.
  const { rows: heatmapRows, columns: heatmapColumns } = data;
  const [query, setQuery] = useState("");
  // null → default order (broadest coverage first, see the content module);
  // otherwise sort model rows by that DOMAIN column's score, descending.
  const [sortDomain, setSortDomain] = useState<number | null>(null);
  // null → default column order (avg domain score); otherwise sort DOMAIN columns
  // by this model's per-domain scores, descending. Mirror of sortDomain.
  const [sortModel, setSortModel] = useState<string | null>(null);
  const [showAllOnMobile, setShowAllOnMobile] = useState(false);
  // Floating hover label: its text + the viewport coords to anchor it at.
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(
    null,
  );

  // The models that become rows, after filtering + optional sort.
  const models = useMemo(() => {
    const q = query.trim().toLowerCase();
    const base: HeatmapRow[] = q
      ? heatmapRows.filter(
          (m) =>
            m.model.toLowerCase().includes(q) ||
            m.company.toLowerCase().includes(q),
        )
      : [...heatmapRows];
    if (sortDomain !== null) {
      base.sort((a, b) => {
        const av = a.cells[sortDomain].mean;
        const bv = b.cells[sortDomain].mean;
        // Unmeasured cells sink to the bottom regardless.
        if (av === null) return bv === null ? 0 : 1;
        if (bv === null) return -1;
        return bv - av;
      });
    }
    // else: heatmapRows already arrives sorted coverage-first.
    return base;
  }, [heatmapRows, query, sortDomain]);

  // The domain columns, in display order: default (avg domain score, from the
  // content module) unless a model is selected, then sorted by that model's
  // per-domain scores. Each column keeps its `index` into the cells array, so
  // the body stays aligned however the columns are ordered.
  const columns = useMemo(() => {
    if (!sortModel) return heatmapColumns;
    const m = heatmapRows.find((r) => r.model === sortModel);
    if (!m) return heatmapColumns;
    return [...heatmapColumns].sort((a, b) => {
      const av = m.cells[a.index].mean;
      const bv = m.cells[b.index].mean;
      // Unmeasured domains sink to the right regardless.
      if (av === null) return bv === null ? 0 : 1;
      if (bv === null) return -1;
      return bv - av;
    });
  }, [heatmapRows, heatmapColumns, sortModel]);

  const searching = query.trim().length > 0;

  return (
    <div className="rd-heatmap panel w-fit max-w-full px-4 py-6 sm:px-6 sm:py-8">
      {/* Search + scale legend. Cells encode accuracy as color, so the ramp has
          to be decodable without hovering — a continuous scale with no legend
          leaves blue-vs-red unreadable. The hatch swatch is called out too,
          since "not measured" is a different kind of thing from a low score. */}
      <div className="flex flex-wrap items-center justify-between gap-x-8 gap-y-3">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search AI subjects…"
          aria-label="Search AI subjects"
          className="w-full max-w-xs rounded-md border border-[var(--line)] bg-[var(--white)] px-3 py-1.5 text-sm text-[var(--ink)] placeholder:text-[var(--muted)] focus:border-[var(--palo-alto)] focus:outline-none"
        />

        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="w-full text-[11px] leading-none text-[var(--muted)] sm:w-auto">
              {metricLabel}
            </span>
            <span className="flex items-center" aria-hidden>
              {SCALE_STOPS.map((s) => (
                <span
                  key={s}
                  className="h-3 w-3 sm:w-6"
                  style={{ backgroundColor: cellColor(s) }}
                />
              ))}
            </span>
            <span className="font-mono text-[11px] leading-none tabular-nums text-[var(--muted)]">
              0 – 100
            </span>
          </div>
          <span className="flex items-center gap-2">
            <span
              className="h-3 w-6 border border-[var(--line)]"
              style={EMPTY_CELL}
              aria-hidden
            />
            <span className="text-[11px] leading-none text-[var(--muted)]">
              Not measured
            </span>
          </span>
        </div>
      </div>

      <p className="mb-2 mt-3 text-[11px] leading-relaxed text-[var(--muted)] sm:hidden">
        Swipe sideways for more domains.
      </p>

      {/* Scrollable grid — rows scroll vertically, header + subject column stick */}
      <div
        className={`rd-heatmap-scroll overflow-auto ${
          showAllOnMobile ? "rd-heatmap-scroll-expanded" : ""
        }`}
      >
        <div
          style={{
            // size to the columns' content (+ headroom for the angled labels)
            // so rows/borders stop at the last column instead of stretching, and
            // no header label is clipped on the right
            width: "max-content",
            minWidth: `calc(${NAME_W} + ${
              heatmapColumns.length * COL_W + LABEL_PAD
            }px)`,
            paddingRight: LABEL_PAD,
          }}
        >
          {/* Header: domain names angled 45° (far more legible than vertical),
              each anchored at the bottom-left of its column so the label rises
              to the right above it. A colored dot sits at the anchor. Clicking a
              domain sorts the model rows by that domain's score. */}
          <div
            className="sticky top-0 z-20 flex items-end border-b border-[var(--line)] bg-[var(--white)]"
            style={{ height: HEADER_H }}
          >
            <div
              className="sticky left-0 z-10 shrink-0 self-end bg-[var(--white)] pb-2 text-xs uppercase tracking-wider text-[var(--muted)]"
              style={{ width: NAME_W }}
            >
              AI subject
            </div>
            {columns.map((domain) => {
              const active = sortDomain === domain.index;
              return (
                <button
                  key={domain.id}
                  type="button"
                  onClick={() => setSortDomain(active ? null : domain.index)}
                  title={`${domain.label}: click to sort AI subjects by this domain`}
                  className="relative shrink-0 cursor-pointer"
                  style={{ width: COL_W, height: HEADER_H }}
                >
                  <span className="absolute bottom-1 left-1 flex origin-bottom-left -rotate-45 items-center gap-1 whitespace-nowrap">
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{
                        backgroundColor: domain.color,
                        opacity: active ? 1 : 0.6,
                      }}
                    />
                    <span
                      className={`text-[11px] leading-none ${
                        active
                          ? "font-semibold text-[var(--ink)]"
                          : "text-[var(--muted)]"
                      }`}
                    >
                      {domain.label}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>

          {/* One row per model */}
          <ol
            className={
              showAllOnMobile ? undefined : "rd-heatmap-mobile-limited"
            }
          >
            {models.map((m) => (
              <li
                key={m.model}
                className="flex items-center border-b border-[var(--line)]/60"
              >
                <button
                  type="button"
                  onClick={() =>
                    setSortModel(sortModel === m.model ? null : m.model)
                  }
                  title={`${m.model}: click to sort domains by this AI subject`}
                  className="sticky left-0 z-10 flex shrink-0 cursor-pointer items-center gap-2 bg-[var(--white)] py-1.5 pr-2 text-left"
                  style={{ width: NAME_W }}
                >
                  <CompanyLogo company={m.company} size={16} />
                  <span
                    className={`truncate text-xs ${
                      sortModel === m.model
                        ? "font-semibold text-[var(--palo-alto)]"
                        : "font-medium text-[var(--ink)]"
                    }`}
                  >
                    {m.short}
                  </span>
                </button>
                {columns.map((domain) => {
                  const cell = m.cells[domain.index];
                  const text =
                    cell.mean === null
                      ? `${m.model} in ${domain.label}: not measured`
                      : `${m.model} in ${domain.label}: ${(cell.mean * 100).toFixed(1)}% over ${cell.n.toLocaleString()} responses`;
                  return (
                    <div
                      key={domain.id}
                      role="img"
                      aria-label={text}
                      className="flex h-9 shrink-0 items-center justify-center font-mono text-[11px] tabular-nums"
                      style={{
                        width: COL_W,
                        color: textColor(cell.mean),
                        ...(cell.mean === null
                          ? EMPTY_CELL
                          : { backgroundColor: cell.color }),
                      }}
                      onMouseEnter={(e) =>
                        setTip({ text, x: e.clientX, y: e.clientY })
                      }
                      onMouseMove={(e) =>
                        setTip({ text, x: e.clientX, y: e.clientY })
                      }
                      onMouseLeave={() => setTip(null)}
                    >
                      {pct(cell.mean)}
                    </div>
                  );
                })}
              </li>
            ))}
          </ol>
        </div>
      </div>

      {models.length > MOBILE_VISIBLE_ROWS ? (
        <div className="mt-3 flex items-center justify-between gap-3 text-xs text-[var(--muted)] sm:hidden">
          <span>
            Showing {showAllOnMobile ? "all" : `top ${MOBILE_VISIBLE_ROWS}`} of{" "}
            {models.length} AI subjects
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

      {searching ? (
        <p className="mt-4 text-xs text-[var(--muted)]">
          {models.length} AI subject{models.length === 1 ? "" : "s"} match “
          {query.trim()}”
        </p>
      ) : null}

      {/* Cursor-following hover label. Portaled to <body> so its `position:
          fixed` resolves against the viewport — an ancestor here (ScrollReveal)
          keeps a CSS transform, which would otherwise make `fixed` relative to
          THAT box and throw the label far from the pointer. Nudged down-right of
          the cursor, flipped left near the right edge, and pointer-events-none so
          it never eats the hover it describes. */}
      {tip && typeof document !== "undefined"
        ? createPortal(
            <div
              className="pointer-events-none fixed z-50 max-w-xs rounded-md bg-[var(--ink)] px-2.5 py-1.5 text-xs leading-snug text-[var(--white)] shadow-lg"
              style={{
                top: tip.y + 16,
                left: tip.x + 14,
                transform:
                  tip.x > window.innerWidth - 280
                    ? "translateX(calc(-100% - 28px))"
                    : undefined,
              }}
            >
              {tip.text}
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}
