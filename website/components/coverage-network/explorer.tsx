"use client";

import dynamic from "next/dynamic";
import { useMemo, useRef, useState } from "react";
import {
  filterCoverageNetwork,
  type CoverageNetwork,
} from "@/lib/coverage-network/model";
import { CoverageSidebar, type CoverageFilters } from "./sidebar";
import { CoverageMatrix } from "./matrix";
import { NODE_COLORS } from "@/lib/coverage-network/appearance";
import type { NetworkControls } from "./canvas";
import { ViewTabs } from "./view-tabs";
import styles from "./network.module.css";

const CoverageCanvas = dynamic(
  () => import("./canvas").then((m) => m.CoverageCanvas),
  {
    ssr: false,
    loading: () => (
      <p className={styles.loading} role="status">
        Drawing the network…
      </p>
    ),
  },
);
const count = (n: number) => n.toLocaleString("en-US");
const defaults: CoverageFilters = {
  domain: "",
  provider: "",
  showUnmapped: true,
};

/** Graph and matrix share visibility, identity, and selection state. */
export function CoverageExplorer({
  data,
  categories,
}: {
  data: CoverageNetwork;
  categories: { id: string; label: string }[];
}) {
  const [view, setView] = useState<"graph" | "matrix">("graph");
  const [selected, setSelected] = useState<string | null>(null);
  const [filters, setFilters] = useState(defaults);
  const [resetCount, setResetCount] = useState(0);
  const controls = useRef<NetworkControls>(null);
  const visible = useMemo(
    () => filterCoverageNetwork(data, filters),
    [data, filters],
  );
  const visibleNodes = useMemo(
    () => data.nodes.filter((n) => visible.nodes.has(n.id)),
    [data, visible],
  );
  const benchmarks = visibleNodes.filter((n) => n.kind === "benchmark").length;
  const unmapped = visibleNodes.filter(
    (n) => n.kind === "benchmark" && !n.degree,
  );
  function reset() {
    setSelected(null);
    setFilters(defaults);
    setResetCount((n) => n + 1);
    controls.current?.reset();
  }
  return (
    <div className={styles.explorer}>
      <div className={styles.toolbar}>
        <div className={styles.legend}>
          <span>
            <i
              data-kind="benchmark"
              style={{ background: NODE_COLORS.benchmark }}
            />
            {count(benchmarks)} benchmarks
          </span>
          <span>
            <i
              data-kind="subject"
              style={{ background: NODE_COLORS.subject }}
            />
            {count(visibleNodes.length - benchmarks)} AI subjects
          </span>
          <span className={styles.edgeCount}>
            {count(visible.edges.length)} links
          </span>
        </div>
        <div className={styles.tools}>
          {view === "graph" ? (
            <>
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
            </>
          ) : null}
          <button type="button" onClick={reset}>
            Reset all
          </button>
        </div>
      </div>
      <div className={styles.viewBar}>
        <ViewTabs
          id="coverage-view"
          label="Coverage display"
          value={view}
          onChange={setView}
          options={[
            { value: "graph", label: "Network" },
            { value: "matrix", label: "Coverage matrix" },
          ]}
        />
        <span>
          {view === "graph"
            ? "Explore connections and shared coverage"
            : "Compare observed coverage, benchmark by subject"}
        </span>
      </div>
      <div className={styles.workspace}>
        <div
          id={`coverage-view-${view}-panel`}
          role="tabpanel"
          aria-labelledby={`coverage-view-${view}`}
          className={styles.plotPanel}
        >
          {view === "graph" ? (
            <>
              <CoverageCanvas
                data={data}
                visible={visible.nodes}
                selected={selected}
                onSelect={setSelected}
                controlsRef={controls}
              />
              {unmapped.length ? (
                <details className={styles.unmapped}>
                  <summary>
                    {count(unmapped.length)} benchmarks without mapped subjects{" "}
                    <span>Inspect benchmarks</span>
                  </summary>
                  <p>
                    These benchmarks have evaluation data, but no links to the
                    current canonical AI subjects.
                  </p>
                  <div>
                    {unmapped.map((n) => (
                      <button
                        key={n.id}
                        type="button"
                        aria-pressed={selected === n.id}
                        onClick={() => setSelected(n.id)}
                      >
                        {n.label}
                      </button>
                    ))}
                  </div>
                </details>
              ) : null}
            </>
          ) : (
            <CoverageMatrix
              key={`${filters.domain}/${filters.provider}/${filters.showUnmapped}/${resetCount}`}
              nodes={visibleNodes}
              edges={visible.edges}
              selected={selected}
              onSelect={setSelected}
            />
          )}
        </div>
        <CoverageSidebar
          key={resetCount}
          data={data}
          visibleNodes={visibleNodes}
          edges={visible.edges}
          categories={categories}
          selected={selected}
          onSelect={setSelected}
          filters={filters}
          onFiltersChange={(change) => {
            setFilters({ ...filters, ...change });
            setSelected(null);
          }}
        />
      </div>
      <div className={styles.notes}>
        <p>
          <strong>Reading coverage.</strong> A link or filled cell means
          recorded evaluations exist for that subject–benchmark pair, including
          records with unavailable grades. A blank cell means no mapped record
          in this snapshot; it is not a failure or a zero score.
        </p>
        <p>
          <strong>Reading the network.</strong> Squares are benchmarks; circles
          are AI subjects. Larger nodes have more connections, with a compressed
          size scale. Positions show coverage structure, not a performance
          scale. Subject identities combine aliases across benchmarks and may
          span several harnesses or configurations.
        </p>
      </div>
    </div>
  );
}
