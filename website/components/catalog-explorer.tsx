"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useState, type ReactNode } from "react";
import { ViewTabs } from "./coverage-network/view-tabs";
import styles from "./coverage-network/network.module.css";

const CoverageLoader = dynamic(
  () => import("./coverage-network/loader").then((m) => m.CoverageLoader),
  {
    loading: () => (
      <p className={styles.loading} role="status">
        Loading coverage explorer…
      </p>
    ),
  },
);

/** Keep catalog state when switching views. Coverage code and data load on demand. */
export function CatalogExplorer({ children }: { children: ReactNode }) {
  const [view, setView] = useState<"catalog" | "coverage">("catalog");
  return (
    <>
      <div className={styles.catalogHeading}>
        <div>
          <h2>Explore the data bank</h2>
          <p>
            Browse benchmarks or explore which AI subjects have been evaluated
            together.
          </p>
        </div>
        {view === "coverage" ? (
          <Link href="/network" prefetch={false}>
            Expand coverage ↗
          </Link>
        ) : null}
      </div>
      <ViewTabs
        id="catalog-view"
        label="Explore the data bank"
        value={view}
        onChange={setView}
        options={[
          { value: "catalog", label: "Catalog" },
          { value: "coverage", label: "Coverage network" },
        ]}
      />
      <div
        id="catalog-view-catalog-panel"
        role="tabpanel"
        aria-labelledby="catalog-view-catalog"
        hidden={view !== "catalog"}
      >
        {children}
      </div>
      {view === "coverage" ? (
        <div
          id="catalog-view-coverage-panel"
          role="tabpanel"
          aria-labelledby="catalog-view-coverage"
        >
          <CoverageLoader />
        </div>
      ) : null}
    </>
  );
}
