/* The trace/1 format (trace_schema.json at the repository root), as the website reads it, and
   the display caps the worker applies. The dataset stores full text; only the browser caps it. */

export type TraceText = string;

type Common = {
  agent?: string;
  /** Original length of each field the worker shortened for display. */
  truncated?: Partial<Record<CappedField, number>>;
};
type CappedField = "content" | "reasoning" | "output" | "arguments" | "error";

export type MessageStep = Common & {
  type: "message";
  label: "system" | "task" | "harness" | "user";
  participant?: string;
  content: TraceText;
};
export type ToolCall = { id?: string; name?: string; arguments: TraceText };
export type LlmStep = Common & {
  type: "llm";
  model?: string;
  content?: TraceText;
  reasoning?: TraceText;
  toolCalls?: ToolCall[];
  usage?: {
    prompt?: number;
    completion?: number;
    reasoning?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  cost?: number;
  start?: number;
  end?: number;
};
export type ToolStep = Common & {
  type: "tool";
  callId?: string;
  name?: string;
  arguments?: TraceText;
  output: TraceText;
  exitCode?: number;
  error?: string;
  start?: number;
  end?: number;
};
export type EventStep = Common & {
  type: "event";
  name: string;
  content?: TraceText;
  start?: number;
  end?: number;
};
export type TraceStep = MessageStep | LlmStep | ToolStep | EventStep;

export type Trace = {
  schema: "trace/1";
  converter: string;
  timing?: "recorded" | "partial" | "synthetic";
  steps: TraceStep[];
  run?: {
    status?: string;
    submission?: string;
    exception?: string;
    traceback?: string;
    cost?: number;
    apiCalls?: number;
    promptTokens?: number;
    completionTokens?: number;
    durationS?: number;
  };
  evaluation?: {
    errorCode?: string;
    errorDetails?: string;
    passed?: number;
    total?: number;
    notRun?: number;
    tests?: { name: string; status: string }[];
    /** Set by the worker when it kept only part of the test list. */
    testsTotal?: number;
    judge?: { content: string; reasoning?: string; participant?: string };
  };
};

export const FIELD_MAX_CHARS = 20000;
export const TESTS_MAX = 5000;

function cap<T extends Common>(
  step: T,
  field: CappedField,
  value: string | undefined,
  limit: number,
) {
  if (value == null || value.length <= limit) return value;
  step.truncated = { ...step.truncated, [field]: value.length };
  return value.slice(0, limit);
}

/** Prepare a parsed trace/1 value for display: drop `native` (not shown yet) and cap long
 *  fields, so the main thread never receives megabytes of a single tool output. */
export function displayTrace(value: unknown, limit = FIELD_MAX_CHARS): Trace {
  const trace = value as Trace & { native?: unknown };
  if (trace?.schema !== "trace/1" || !Array.isArray(trace.steps))
    throw new Error("Not a trace/1 value.");
  delete trace.native;
  for (const step of trace.steps as (TraceStep & { native?: unknown })[]) {
    delete step.native;
    if (step.type === "llm") {
      step.content = cap(step, "content", step.content, limit);
      step.reasoning = cap(step, "reasoning", step.reasoning, limit);
      for (const call of step.toolCalls ?? [])
        call.arguments =
          cap(step, "arguments", call.arguments, limit) ?? call.arguments;
    } else if (step.type === "tool") {
      step.output = cap(step, "output", step.output, limit) ?? "";
      step.arguments = cap(step, "arguments", step.arguments, limit);
      step.error = cap(step, "error", step.error, limit);
    } else {
      step.content = cap(step, "content", step.content, limit) ?? "";
    }
  }
  const tests = trace.evaluation?.tests;
  if (tests && tests.length > TESTS_MAX) {
    // Keep what a reader looks for first: the tests that did not pass.
    const ordered = [
      ...tests.filter((t) => t.status !== "passed"),
      ...tests.filter((t) => t.status === "passed"),
    ];
    trace.evaluation!.testsTotal = tests.length;
    trace.evaluation!.tests = ordered.slice(0, TESTS_MAX);
  }
  return trace;
}
