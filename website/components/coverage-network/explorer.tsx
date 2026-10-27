"use client";

import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import {
  filterCoverageNetwork,
  type CoverageNetwork,
} from "@/lib/coverage-network/model";
import { CoverageCanvas, NODE_COLORS, type NetworkControls } from "./canvas";
import styles from "./network.module.css";

const count = (n: number) => n.toLocaleString("en-US");

export function CoverageExplorer({
  data,
  categories,
}: {
  data: CoverageNetwork;
  categories: { id: string; label: string }[];
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("");
  const [provider, setProvider] = useState("");
  const [showUnmapped, setShowUnmapped] = useState(true);
  const controls = useRef<NetworkControls>(null);
  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data]);
  const providers = useMemo(
    () =>
      [
        ...new Set(data.nodes.flatMap((n) => (n.provider ? [n.provider] : []))),
      ].sort(),
    [data],
  );
  const visible = useMemo(
    () => filterCoverageNetwork(data, { domain, provider, showUnmapped }),
    [data, domain, provider, showUnmapped],
  );
  const visibleNodes = useMemo(
    () => data.nodes.filter((n) => visible.nodes.has(n.id)),
    [data, visible],
  );
  const benchmarks = visibleNodes.filter((n) => n.kind === "benchmark").length;
  const unmapped = data.nodes.filter(
    (n) => n.kind === "benchmark" && !n.degree,
  ).length;
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
        ? visible.edges
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
    [visible, byId, selected],
  );
  const categoryLabels = new Map(categories.map((c) => [c.id, c.label]));

  function reset() {
    setSelected(null);
    setQuery("");
    setDomain("");
    setProvider("");
    setShowUnmapped(true);
    controls.current?.reset();
  }

  return (
    <div className={styles.explorer}>
      <div className={styles.toolbar}>
        <div className={styles.legend}>
          <span>
            <i style={{ background: NODE_COLORS.benchmark }} />
            {count(benchmarks)} benchmarks
          </span>
          <span>
            <i style={{ background: NODE_COLORS.subject }} />
            {count(visibleNodes.length - benchmarks)} AI subjects
          </span>
          <span className={styles.edgeCount}>
            {count(visible.edges.length)} links
          </span>
        </div>
        <div className={styles.tools}>
          <button
            type="button"
            aria-label="Zoom in"
            onClick={() => controls.current?.zoom(1 / 1.4)}
          >
            +
          </button>
          <button
            type="button"
            aria-label="Zoom out"
            onClick={() => controls.current?.zoom(1.4)}
          >
            −
          </button>
          <button
            type="button"
            onClick={() => {
              setSelected(null);
              controls.current?.reset();
            }}
          >
            Fit graph
          </button>
          <button type="button" onClick={reset}>
            Reset all
          </button>
        </div>
      </div>
      <div className={styles.workspace}>
        <CoverageCanvas
          data={data}
          visible={visible.nodes}
          selected={selected}
          onSelect={setSelected}
          controlsRef={controls}
        />
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
                    setDomain(e.target.value);
                    setSelected(null);
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
                    setProvider(e.target.value);
                    setSelected(null);
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
                  setShowUnmapped(e.target.checked);
                  setSelected(null);
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
                <button type="button" onClick={() => setSelected(null)}>
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
                  This benchmark has no connections to the bank’s canonically
                  named AI subjects in this snapshot. Its evaluation data may
                  include subjects whose identities have not been mapped. Open
                  the benchmark to inspect them.
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
                        <button
                          type="button"
                          onClick={() => setSelected(neighbor.id)}
                        >
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
                      onClick={() => setSelected(result.id)}
                    >
                      <i style={{ background: NODE_COLORS[result.kind] }} />
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
                Showing the 30 most connected matches. Refine your search to
                find any node.
              </p>
            ) : null}
          </section>
        </aside>
      </div>
      <div className={styles.notes}>
        <p>
          <strong>Reading the map.</strong> A link means the bank contains
          recorded evaluations for that subject–benchmark pair, including
          records with an unavailable grade. Larger nodes have more connections.
          Positions reflect evaluation overlap, not capability or performance.
        </p>
        <p>
          <strong>{count(unmapped)} benchmarks without mapped subjects.</strong>{" "}
          The separate block of benchmark nodes has no links to the current
          canonical subject identities. This does not mean those benchmarks have
          no evaluations. Subject identities combine aliases across benchmarks;
          they do not distinguish every harness or configuration.
        </p>
      </div>
    </div>
  );
}
