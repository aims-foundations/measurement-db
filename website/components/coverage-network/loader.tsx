"use client";

import { useEffect, useState } from "react";
import { withBase } from "@/lib/base-path";
import type { CoverageNetwork } from "@/lib/coverage-network/model";
import { CoverageExplorer } from "./explorer";
import styles from "./network.module.css";

type Snapshot = {
  data: CoverageNetwork;
  categories: { id: string; label: string }[];
};
let cached: Snapshot | undefined;

export function CoverageLoader() {
  const [snapshot, setSnapshot] = useState(cached);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (cached) return;
    const controller = new AbortController();
    fetch(withBase("/api/coverage"), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Coverage unavailable");
        const value: Snapshot = await response.json();
        if (
          !value?.data?.nodes ||
          !value.data.edges ||
          !Array.isArray(value.categories)
        )
          throw new Error("Invalid coverage snapshot");
        cached = value;
        setSnapshot(value);
      })
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      });
    return () => controller.abort();
  }, [attempt]);
  if (snapshot) return <CoverageExplorer {...snapshot} />;
  if (error)
    return (
      <div className={styles.loading} role="alert">
        <p>
          Coverage could not be loaded. The benchmark catalog is still
          available.
        </p>
        <button
          type="button"
          onClick={() => {
            setError(false);
            setAttempt(attempt + 1);
          }}
        >
          Try again
        </button>
      </div>
    );
  return (
    <p className={styles.loading} role="status">
      Loading coverage explorer…
    </p>
  );
}
