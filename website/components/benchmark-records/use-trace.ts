"use client";

import { useEffect, useState } from "react";
import type { RecordSource } from "@/lib/benchmark-records/client";
import type { TraceDisplay } from "@/lib/benchmark-records/trace";
import { PagedRecordError } from "@/lib/benchmark-records/record-preview";
import type { LoadState } from "./use-records";

export type TraceLoad = LoadState<TraceDisplay | null> & { retry: () => void };

/** Selection, cancellation and retries stay outside the answer presentation. */
export function useTrace(source: RecordSource, id: string | null): TraceLoad {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<{
    source: RecordSource;
    id: string;
    state: LoadState<TraceDisplay | null>;
  }>();
  useEffect(() => {
    if (id === null) return;
    const controller = new AbortController();
    source.trace(id, controller.signal).then(
      (value) => {
        if (!controller.signal.aborted)
          setResult({ source, id, state: { status: "ready", value } });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({
            source,
            id,
            state: {
              status: "error",
              message: error instanceof Error ? error.message : String(error),
              ...(error instanceof PagedRecordError
                ? { paged: error.record }
                : {}),
            },
          });
      },
    );
    return () => controller.abort();
  }, [source, id, attempt]);
  const state: LoadState<TraceDisplay | null> =
    id === null
      ? { status: "ready", value: null }
      : result?.source === source && result.id === id
        ? result.state
        : { status: "loading" };
  return {
    ...state,
    retry: () => {
      setResult(undefined);
      setAttempt((value) => value + 1);
    },
  };
}
