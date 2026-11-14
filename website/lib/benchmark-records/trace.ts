import { displayTrace, type Trace as StructuredTrace } from "../traces/display";
import type { Trace } from "./types";
import type { BoundedTrace } from "./trace-preview";

export type TraceDisplay =
  | { kind: "paged"; value: BoundedTrace }
  | { kind: "structured"; value: StructuredTrace }
  | { kind: "legacy"; value: Trace };

/** Prepare a separate display value; the canonical record remains available unchanged. */
export function prepareTraceRecord(record: Trace | null): TraceDisplay | null {
  if (!record) return null;
  try {
    const value: unknown = JSON.parse(record.trace);
    if ((value as { schema?: unknown } | null)?.schema === "trace/1")
      return { kind: "structured", value: displayTrace(value) };
  } catch {
    // Legacy text and malformed structured records retain their exact source view.
  }
  return { kind: "legacy", value: record };
}
