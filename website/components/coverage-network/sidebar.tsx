"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import type {
  CoverageNetwork,
  CoverageNode,
  CoverageEdge,
} from "@/lib/coverage-network/model";
import { NODE_COLORS } from "@/lib/coverage-network/appearance";
import styles from "./network.module.css";

export type CoverageFilters = {
  domain: string;
  provider: string;
  showUnmapped: boolean;
};
const count = (n: number) => n.toLocaleString("en-US");
export function CoverageSidebar({
  data,
  visibleNodes,
  edges,
  categories,
  selected,
  onSelect,
  filters,
  onFiltersChange,
}: {
  data: CoverageNetwork;
  visibleNodes: CoverageNode[];
  edges: CoverageEdge[];
  categories: { id: string; label: string }[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  filters: CoverageFilters;
  onFiltersChange: (change: Partial<CoverageFilters>) => void;
}) {
  const { domain, provider, showUnmapped } = filters;
  const [query, setQuery] = useState("");
  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data]);
  const providers = useMemo(
    () =>
      [
        ...new Set(data.nodes.flatMap((n) => (n.provider ? [n.provider] : []))),
      ].sort(),
    [data],
  );
  const node = selected ? byId.get(selected) : undefined;
  const results = useMemo(() => {
    const term = query.trim().toLowerCase();
    return visibleNodes
      .filter(
        (n) =>
          !term ||
          `${n.label} ${n.provider || ""}`.toLowerCase().includes(term),
      )
      .sort((a, b) => b.degree - a.degree || a.label.localeCompare(b.label));
  }, [visibleNodes, query]);
  const neighbors = useMemo(
    () =>
      selected
        ? edges
            .filter((e) => e.source === selected || e.target === selected)
            .map((e) => ({
              node: byId.get(e.source === selected ? e.target : e.source)!,
              items: e.items,
            }))
            .sort(
              (a, b) =>
                b.items - a.items || a.node.label.localeCompare(b.node.label),
            )
        : [],
    [edges, byId, selected],
  );
  const categoryLabels = new Map(categories.map((c) => [c.id, c.label]));
  return (
    <aside className={styles.sidebar} aria-label="Explore network nodes">
      <div className={styles.filters}>
        <label htmlFor="coverage-search">Find a benchmark or subject</label>
        <input
          id="coverage-search"
          type="search"
          placeholder="Try GPT-4o or SWE-bench…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className={styles.selects}>
          <label>
            Domain
            <select
              aria-label="Benchmark domain"
              value={domain}
              onChange={(e) => {
                onFiltersChange({ domain: e.target.value });
              }}
            >
              <option value="">All domains</option>
              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Provider
            <select
              aria-label="Subject provider"
              value={provider}
              onChange={(e) => {
                onFiltersChange({ provider: e.target.value });
              }}
            >
              <option value="">All providers</option>
              {providers.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={showUnmapped}
            onChange={(e) => {
              onFiltersChange({ showUnmapped: e.target.checked });
            }}
          />
          Include benchmarks without mapped subjects
        </label>
      </div>

      {node ? (
        <section
          className={styles.details}
          aria-label="Selected node"
          aria-live="polite"
        >
          <div className={styles.detailHeader}>
            <span className={styles.kicker}>
              {node.kind === "benchmark" ? "Benchmark" : "AI subject"}
            </span>
            <button type="button" onClick={() => onSelect(null)}>
              Clear selection
            </button>
          </div>
          <h2>{node.label}</h2>
          <p className={styles.metadata}>
            {node.kind === "subject"
              ? node.provider
              : node.categories
                  .map((c) => categoryLabels.get(c) || c)
                  .join(" · ")}
          </p>
          <p className={styles.connectionCount}>
            {count(neighbors.length)}{" "}
            {node.kind === "subject" ? "benchmarks" : "mapped subjects"}
            {neighbors.length !== node.degree
              ? ` shown · ${count(node.degree)} in the bank`
              : " in the bank"}
          </p>
          {node.slug ? (
            <Link className={styles.benchmarkLink} href={`/${node.slug}`}>
              Open benchmark →
            </Link>
          ) : null}
          {node.kind === "benchmark" && !node.degree ? (
            <p className={styles.mappingNote}>
              This benchmark has no connections to the bank’s canonically named
              AI subjects in this snapshot. Its evaluation data may include
              subjects whose identities have not been mapped. Open the benchmark
              to inspect them.
            </p>
          ) : null}
          {neighbors.length ? (
            <>
              <div className={styles.listHeading}>
                <span>
                  Connected{" "}
                  {node.kind === "subject" ? "benchmarks" : "subjects"}
                </span>
                <span>Items evaluated</span>
              </div>
              <ul className={styles.connections}>
                {neighbors.map(({ node: neighbor, items }) => (
                  <li key={neighbor.id}>
                    <button type="button" onClick={() => onSelect(neighbor.id)}>
                      <span>{neighbor.label}</span>
                      <span>{count(items)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </section>
      ) : null}

      <section className={styles.browse} aria-label="Search results">
        <div className={styles.listHeading}>
          <span>
            {query.trim()
              ? `${count(results.length)} matches`
              : "Explore nodes"}
          </span>
          <span>Links</span>
        </div>
        {results.length ? (
          <ul className={styles.results}>
            {results.slice(0, 30).map((result) => (
              <li key={result.id}>
                <button
                  type="button"
                  aria-pressed={selected === result.id}
                  onClick={() => onSelect(result.id)}
                >
                  <i
                    data-kind={result.kind}
                    style={{ background: NODE_COLORS[result.kind] }}
                  />
                  <span>
                    {result.label}
                    <small>
                      {result.kind === "benchmark"
                        ? "Benchmark"
                        : result.provider}
                    </small>
                  </span>
                  <span>{count(result.degree)}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className={styles.empty}>
            No matches with these filters. Try another name or reset the
            filters.
          </p>
        )}
        {results.length > 30 ? (
          <p className={styles.resultsHint}>
            Showing the 30 most connected matches. Refine your search to find
            any node.
          </p>
        ) : null}
      </section>
    </aside>
  );
}
