/** Bounded display transport. Full source records remain separate and unchanged. */
import type { Trace, TraceStep } from "../traces/display";
import {
  readPreviewPart,
  type PreviewRange,
  type RecordPreview,
} from "./record-preview";

export const TRACE_PAGE_BYTES = 1024 * 1024;
export type TracePreview = RecordPreview & {
  format: "trace-preview-v1";
  source_sha256: string;
  exporter: string;
  page_bytes: number;
};
export type TraceField = {
  step?: number;
  path: (string | number)[];
  characters: number;
  encoding?: "json";
  parts: PreviewRange[];
};
export type TracePageDescriptor = {
  start: number;
  end: number;
  turnStart: number;
  range: PreviewRange;
};
export type TraceHeader = {
  kind: "trace-preview-v1";
  response_id: string;
  source_trace_sha256: string;
  trace: Omit<Trace, "steps">;
  summary: {
    steps: number;
    modelTurns: number;
    toolResults: number;
    model?: string;
    hasReasoning: boolean;
  };
  pages: TracePageDescriptor[];
  fields: TraceField[];
  search: PreviewRange[];
};
export type TracePage = {
  response_id: string;
  start: number;
  steps: TraceStep[];
  fields: TraceField[];
};
export type BoundedTrace = {
  preview: TracePreview;
  header: TraceHeader;
  firstPage: TracePage;
};
type Transport = { base: string; request?: typeof fetch };

export function matchingTracePreview(
  source: string | undefined,
  preview: TracePreview | undefined,
): TracePreview | undefined {
  return preview?.format === "trace-preview-v1" && preview.source === source
    ? preview
    : undefined;
}

function checkDescriptor(preview: TracePreview) {
  if (
    preview.format !== "trace-preview-v1" ||
    !Number.isSafeInteger(preview.page_bytes) ||
    preview.page_bytes > TRACE_PAGE_BYTES ||
    preview.page_bytes < 1 ||
    preview.indexes.length !== 256 ||
    !/^[a-f0-9]{40}$/.test(preview.revision) ||
    !/^[a-f0-9]{64}$/.test(preview.source_sha256)
  )
    throw new Error("Invalid structured trace preview descriptor.");
}

async function part(
  preview: TracePreview,
  range: PreviewRange,
  signal: AbortSignal,
  transport: Transport,
): Promise<string> {
  checkDescriptor(preview);
  signal.throwIfAborted();
  if (range[2] > preview.page_bytes || range[3] > preview.page_bytes)
    throw new Error("Structured trace page exceeds its byte budget.");
  const value = await readPreviewPart(
    transport.base,
    preview,
    range,
    signal,
    transport.request,
  );
  signal.throwIfAborted();
  return value;
}

export async function readTracePreview(
  preview: TracePreview,
  id: string,
  signal: AbortSignal,
  transport: Transport,
): Promise<BoundedTrace | null> {
  checkDescriptor(preview);
  const bucket = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(id)),
  )[0];
  const index = JSON.parse(
    await part(preview, preview.indexes[bucket], signal, transport),
  ) as Record<string, PreviewRange>;
  if (!Object.hasOwn(index, id)) return null;
  const header = JSON.parse(
    await part(preview, index[id], signal, transport),
  ) as TraceHeader;
  if (
    header.kind !== "trace-preview-v1" ||
    header.response_id !== id ||
    header.trace.schema !== "trace/1" ||
    !Number.isSafeInteger(header.summary.steps) ||
    header.summary.steps < 0 ||
    !/^[a-f0-9]{64}$/.test(header.source_trace_sha256) ||
    Object.hasOwn(header.trace, "native")
  )
    throw new Error("Structured trace preview identity or format mismatch.");
  let end = 0;
  for (const page of header.pages) {
    if (
      page.start !== end ||
      !Number.isSafeInteger(page.end) ||
      page.end <= page.start
    )
      throw new Error("Structured trace step index is incomplete.");
    end = page.end;
  }
  if (end !== header.summary.steps)
    throw new Error("Structured trace step count mismatch.");
  const firstPage = header.pages.length
    ? await readTracePage(preview, header, 0, signal, transport)
    : { response_id: id, start: 0, steps: [], fields: [] };
  return { preview, header, firstPage };
}

export async function readTracePage(
  preview: TracePreview,
  header: TraceHeader,
  index: number,
  signal: AbortSignal,
  transport: Transport,
): Promise<TracePage> {
  const descriptor = header.pages[index];
  if (!descriptor) throw new Error("Unknown trace step page.");
  const value = JSON.parse(
    await part(preview, descriptor.range, signal, transport),
  ) as TracePage;
  if (
    value.response_id !== header.response_id ||
    value.start !== descriptor.start ||
    value.steps.length !== descriptor.end - descriptor.start ||
    value.steps.some((s) => Object.hasOwn(s, "native"))
  )
    throw new Error(
      "Structured trace page does not match the selected observation.",
    );
  return value;
}

export async function readTraceField(
  trace: BoundedTrace,
  field: TraceField,
  index: number,
  signal: AbortSignal,
  transport: Transport,
): Promise<string> {
  if (!field.parts[index]) throw new Error("Unknown complete-text page.");
  return part(trace.preview, field.parts[index], signal, transport);
}

export type TraceSearch = {
  matches: number[];
  total: number;
  complete: boolean;
};
const SEARCH_MATCHES = 10000;

/** Scan complete text in bounded pages. Native data is outside the declared search scope. */
export async function searchTrace(
  trace: BoundedTrace,
  query: string,
  signal: AbortSignal,
  transport: Transport,
  progress?: (done: number, total: number) => void,
): Promise<TraceSearch> {
  const q = query.trim().toLowerCase();
  if (!q || q.length > 256)
    throw new Error("Enter between 1 and 256 characters to search.");
  const matches: number[] = [];
  let previous = -1,
    tail = "",
    found = false,
    total = 0;
  for (let i = 0; i < trace.header.search.length; i++) {
    const value = JSON.parse(
      await part(trace.preview, trace.header.search[i], signal, transport),
    ) as {
      response_id: string;
      entries: [number, string, boolean][];
    };
    if (value.response_id !== trace.header.response_id)
      throw new Error("Trace search identity mismatch.");
    for (const [step, text, continuation] of value.entries) {
      if (
        !Number.isSafeInteger(step) ||
        step < 0 ||
        step >= trace.header.summary.steps ||
        (continuation ? step !== previous : step <= previous)
      )
        throw new Error("Trace search step order mismatch.");
      if (!continuation) {
        tail = "";
        found = false;
      }
      const lower = tail + text.toLowerCase();
      if (!found && lower.includes(q)) {
        found = true;
        total++;
        if (matches.length < SEARCH_MATCHES) matches.push(step);
      }
      tail = lower.slice(-Math.max(0, q.length - 1));
      if (q.length === 1) tail = "";
      previous = step;
    }
    progress?.(i + 1, trace.header.search.length);
  }
  return { matches, total, complete: total === matches.length };
}
