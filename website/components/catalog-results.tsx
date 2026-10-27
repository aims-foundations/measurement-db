"use client";

import { useId, useRef, useState, type ReactNode } from "react";

const PAGE_SIZE = 12;

/** One limit across year groups; search and filtering happen before this view. */
export function CatalogResults<T extends { slug: string }>({
  groups,
  renderItem,
}: {
  groups: { key: string; label: string | null; items: T[] }[];
  renderItem: (item: T) => ReactNode;
}) {
  const [limit, setLimit] = useState(PAGE_SIZE);
  const root = useRef<HTMLDivElement>(null);
  const resultsId = useId();
  const total = groups.reduce((sum, group) => sum + group.items.length, 0);
  const shown = Math.min(limit, total);
  const visibleGroups = groups.flatMap((group, index) => {
    const preceding = groups
      .slice(0, index)
      .reduce((sum, entry) => sum + entry.items.length, 0);
    const items = group.items.slice(0, Math.max(0, limit - preceding));
    return items.length ? [{ ...group, items, total: group.items.length }] : [];
  });

  function focusCard(index: number) {
    root.current
      ?.querySelectorAll<HTMLElement>("[data-catalog-item]")
      [index]?.querySelector<HTMLAnchorElement>("a")
      ?.focus({ preventScroll: true });
  }

  return (
    <div ref={root} className="scroll-mt-24">
      <div
        id={resultsId}
        className="space-y-14"
        role="region"
        aria-label="Benchmark catalog results"
      >
        {visibleGroups.map((group) => (
          <section key={group.key} aria-label={group.label || "Benchmarks"}>
            {group.label ? (
              <div className="mb-5 flex items-baseline gap-3 border-b border-[var(--line)] pb-3">
                <h3 className="type-h3 text-[1.25rem] text-[var(--ink)]">
                  {group.label}
                </h3>
                <span className="font-mono text-sm text-[var(--muted)]">
                  {group.items.length < group.total
                    ? `${group.items.length} of ${group.total} shown`
                    : group.total}
                </span>
              </div>
            ) : null}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {group.items.map((item) => (
                <div key={item.slug} data-catalog-item>
                  {renderItem(item)}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-[var(--line)] pt-5">
        <p
          className="text-sm text-[var(--muted)]"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          Showing {shown} of {total} benchmarks
        </p>
        <div className="flex items-center gap-5">
          {shown > PAGE_SIZE ? (
            <button
              type="button"
              aria-controls={resultsId}
              className="min-h-11 cursor-pointer text-sm text-[var(--muted)] underline underline-offset-4 focus-visible:outline-2 focus-visible:outline-offset-4"
              onClick={() => {
                setLimit(PAGE_SIZE);
                requestAnimationFrame(() => {
                  root.current?.scrollIntoView({ block: "start" });
                  focusCard(0);
                });
              }}
            >
              Show less
            </button>
          ) : null}
          {shown < total ? (
            <button
              type="button"
              aria-controls={resultsId}
              className="min-h-11 cursor-pointer rounded-full border border-[var(--rd-cardinal)] px-6 py-2 text-sm font-medium text-[var(--rd-cardinal)] transition-colors hover:bg-[var(--rd-cardinal)] hover:text-white focus-visible:outline-2 focus-visible:outline-offset-4"
              onClick={() => {
                setLimit(limit + PAGE_SIZE);
                requestAnimationFrame(() => focusCard(shown));
              }}
            >
              Show more
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
