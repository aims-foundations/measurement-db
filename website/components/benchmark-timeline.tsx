"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";

import { useMobileLatestScroll } from "@/components/use-mobile-latest-scroll";
import type {
  BenchmarkTimelineData,
  TimelineItem,
} from "@/content/benchmark-timeline";

// Dot-histogram release timeline, the same idiom as the model release
// timeline so the two charts read as a pair: every year gets the same width,
// split into 12 month columns, and each month-dated benchmark is one small
// dot stacked in its release month — column height reads as releases per
// month. Dots are colored by primary domain (legend above the chart, same
// hue as the gallery card), link to the benchmark's detail page, and carry a
// hover tooltip with the name, domain and release month. Derived entirely
// from benchmark-cards.json (see content/benchmark-timeline.ts).

// Same pitch as the model timeline so the two figures stay one idiom at two
// scales. Sized generously — the chart lives folded inside an accordion, so
// its footprint costs nothing until a reader asks for it.
const DOT = 7; // px dot diameter
const HIT = 12; // px pointer/focus target around the visual dot
const GAP = 1; // px between stacked dots
const PITCH = HIT + GAP;
const MONTH_W = HIT; // one non-overlapping target per month column
const YEAR_W = MONTH_W * 12;

type Tip = { item: TimelineItem; x: number; y: number };

export function BenchmarkTimeline({ data }: { data: BenchmarkTimelineData }) {
  const [tip, setTip] = useState<Tip | null>(null);
  const scrollRef = useMobileLatestScroll(data.years.length);

  const chartH = data.maxMonth * PITCH;

  return (
    <div className="panel w-fit max-w-full px-4 py-6 sm:px-6 sm:py-8">
      {/* Domain legend — gallery-card hues from content/measurement-db.ts */}
      <div className="mb-5 flex flex-wrap gap-x-4 gap-y-1.5">
        {data.legend.map((l) => (
          <span key={l.label} className="flex items-center gap-1.5">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: l.color }}
            />
            <span className="text-[11px] leading-none text-[var(--muted)]">
              {l.label}
            </span>
          </span>
        ))}
      </div>

      <p className="mb-2 text-[11px] leading-relaxed text-[var(--muted)] lg:hidden">
        Showing recent releases. Swipe for earlier years.
      </p>
      <div
        ref={scrollRef}
        className="snap-x snap-proximity overflow-x-auto focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--digital-blue)]"
        role="region"
        tabIndex={0}
        aria-label="Benchmark release timeline; recent years shown first on narrow screens"
      >
        <div className="flex" style={{ width: "max-content" }}>
          {data.years.map((y) => (
            <div
              key={y.label}
              className="snap-end border-l border-[var(--line)]"
              style={{ width: YEAR_W }}
            >
              {/* 12 month columns of bottom-anchored dot stacks */}
              <div className="flex items-end" style={{ height: chartH }}>
                {y.months.map((items, mi) => (
                  <div
                    key={mi}
                    className="flex flex-col-reverse items-center"
                    style={{ width: MONTH_W, gap: GAP }}
                  >
                    {items.map((it) => {
                      const move = (e: React.MouseEvent) =>
                        setTip({ item: it, x: e.clientX, y: e.clientY });
                      return (
                        <Link
                          key={it.slug}
                          href={`/${it.slug}`}
                          aria-label={`${it.name}, ${it.domain}, released ${it.released}`}
                          className="group flex shrink-0 items-center justify-center rounded-full focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--digital-blue)]"
                          style={{
                            width: HIT,
                            height: HIT,
                          }}
                          onMouseEnter={move}
                          onMouseMove={move}
                          onMouseLeave={() => setTip(null)}
                        >
                          <span
                            className="shrink-0 rounded-full opacity-80 group-hover:opacity-100"
                            style={{
                              width: DOT,
                              height: DOT,
                              backgroundColor: it.color,
                            }}
                            aria-hidden
                          />
                        </Link>
                      );
                    })}
                  </div>
                ))}
              </div>

              {/* Axis — the year + its count */}
              <div className="border-t border-[var(--line)] pt-1.5 text-center">
                <span className="font-mono text-[10px] leading-none text-[var(--muted)]">
                  {y.label} · {y.count}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Exclusions, said out loud: a chart that bounds its coverage and
          doesn't say so reads as complete. */}
      {data.excluded.length > 0 ? (
        <p className="mt-4 max-w-prose text-[11px] leading-relaxed text-[var(--muted)]">
          Not shown (no month-dated release): {data.excluded.join(", ")}.
        </p>
      ) : null}

      {/* Cursor-following tooltip, portaled to <body> so `position: fixed`
          resolves against the viewport despite transformed ancestors
          (ScrollReveal) — same trick as the heatmap. */}
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
              <p className="font-semibold">{tip.item.name}</p>
              <p className="mt-0.5 opacity-85">
                {tip.item.domain} · Released {tip.item.released}
              </p>
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}
