import { useEffect, useState } from "react";
import type { LoadState } from "../benchmark-records/use-records";

/** A changed selection cannot display a completed request for the previous one. */
export function useRequest<T>(load: (signal: AbortSignal) => Promise<T>) {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<{
    load: typeof load;
    state: LoadState<T>;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(
      (value) => {
        if (!controller.signal.aborted)
          setResult({ load, state: { status: "ready", value } });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({
            load,
            state: { status: "error", message: String(error.message ?? error) },
          });
      },
    );
    return () => controller.abort();
  }, [load, attempt]);
  return {
    state:
      result?.load === load
        ? result.state
        : ({ status: "loading" } as LoadState<T>),
    retry: () => {
      setResult(undefined);
      setAttempt((n) => n + 1);
    },
  };
}
