"use client";

/* ============================================================
   TraceViewer — renders one trace/1 value (see lib/traces/display.ts)
   as a transcript: model turns with their tool calls, tool results
   indented under them, harness messages, and the grader's result.
   Trace content is only ever rendered as text, never as HTML.
   ============================================================ */

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type {
  LlmStep,
  MessageStep,
  ToolStep,
  Trace,
  TraceStep,
} from "@/lib/traces/display";

type Kind = "llm" | "tool" | "tool-nz" | "tool-err" | "harness" | "context";
/* `color` marks a step's card. On the step strip only the kinds a reader hunts
   for (failures, harness interventions) are colored; model turns and tool calls
   alternate on agent runs, so coloring them too turns the strip into one band. */
const KIND: Record<Kind, { color: string; label: string; flagged?: boolean }> =
  {
    llm: { color: "var(--digital-blue)", label: "model turn" },
    tool: { color: "var(--stone)", label: "tool call" },
    "tool-nz": {
      color: "var(--poppy)",
      label: "non-zero exit",
      flagged: true,
    },
    "tool-err": {
      color: "var(--digital-red)",
      label: "tool error",
      flagged: true,
    },
    harness: { color: "var(--plum)", label: "harness message", flagged: true },
    context: { color: "var(--stone)", label: "system / task / other" },
  };
const STRIP_NEUTRAL = "color-mix(in srgb, var(--stone) 30%, white)";

function kindOf(step: TraceStep): Kind {
  if (step.type === "llm") return "llm";
  if (step.type === "tool")
    return step.error
      ? "tool-err"
      : step.exitCode != null && step.exitCode !== 0
        ? "tool-nz"
        : "tool";
  return step.type === "message" && step.label === "harness"
    ? "harness"
    : "context";
}

function searchText(step: TraceStep) {
  if (step.type === "llm")
    return [
      step.content,
      step.reasoning,
      ...(step.toolCalls ?? []).map((c) => c.arguments),
    ].join("\n");
  if (step.type === "tool") return [step.output, step.error].join("\n");
  return step.content ?? "";
}

const TESTS_SHOWN = 2000;
const int = (n?: number) => (n == null ? "–" : Math.round(n).toLocaleString());
const money = (c?: number) =>
  c == null ? "–" : "$" + (c < 0.01 ? c.toFixed(4) : c.toFixed(2));
function duration(s?: number) {
  if (s == null) return "–";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 90) return `${s.toFixed(1)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

/** A step's duration, when it has one. A zero span means the source had no
 *  start time for the step (its start was set to its end), not an instant call. */
function elapsed(step: { start?: number; end?: number }) {
  return step.start != null && step.end != null && step.end > step.start
    ? duration(step.end - step.start)
    : null;
}

/* ---------- text with search highlighting ---------- */
function Highlight({ text, query }: { text: string; query: string }) {
  if (!query) return <>{text}</>;
  const lower = text.toLowerCase();
  const parts: React.ReactNode[] = [];
  let at = 0;
  for (let hit = lower.indexOf(query); hit !== -1; ) {
    if (hit > at) parts.push(text.slice(at, hit));
    parts.push(
      <mark key={hit} className="rounded-sm bg-[#ffe58a] text-inherit">
        {text.slice(hit, hit + query.length)}
      </mark>,
    );
    at = hit + query.length;
    hit = lower.indexOf(query, at);
  }
  if (at < text.length) parts.push(text.slice(at));
  return <>{parts}</>;
}

function Omitted({ from, shown }: { from?: number; shown: number }) {
  if (from == null) return null;
  return (
    <p className="mt-1 text-[0.6875rem] text-[var(--muted)]">
      Showing the first {shown.toLocaleString()} of {from.toLocaleString()}{" "}
      characters.
    </p>
  );
}

/** Monospace block clamped to `lines`, with "Show all N lines". */
function Block({
  text,
  lines,
  query,
  expandAll,
  truncatedFrom,
}: {
  text: string;
  lines: number;
  query: string;
  expandAll: boolean;
  truncatedFrom?: number;
}) {
  const [open, setOpen] = useState(false);
  const count = text.split("\n").length;
  const hit = !!query && text.toLowerCase().includes(query);
  const clamped = count > lines && !open && !expandAll && !hit;
  return (
    <div>
      <pre
        className="m-0 overflow-hidden whitespace-pre-wrap break-words rounded border border-[var(--line)] bg-[var(--fog-light)] px-2.5 py-2 font-mono text-[0.75rem] leading-[1.45]"
        style={
          clamped
            ? {
                maxHeight: `calc(1.45em * ${lines} + 1rem)`,
                maskImage: "linear-gradient(to bottom, #000 75%, transparent)",
                WebkitMaskImage:
                  "linear-gradient(to bottom, #000 75%, transparent)",
              }
            : undefined
        }
      >
        <Highlight text={text} query={query} />
      </pre>
      {count > lines && !expandAll && !hit ? (
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="mt-1 rounded border border-[var(--line)] bg-white px-2 py-0.5 text-[0.6875rem] text-[var(--muted)] hover:border-[var(--muted)]"
        >
          {open ? "Show less" : `Show all ${count.toLocaleString()} lines`}
        </button>
      ) : null}
      <Omitted from={truncatedFrom} shown={text.length} />
    </div>
  );
}

function commandOf(args?: string) {
  if (args == null) return null;
  try {
    const parsed = JSON.parse(args);
    if (
      parsed &&
      typeof parsed === "object" &&
      Object.keys(parsed).length === 1 &&
      typeof parsed.command === "string"
    )
      return parsed.command as string;
    return JSON.stringify(parsed, null, 2);
  } catch {
    return args;
  }
}
function firstLine(args?: string) {
  const line = (commandOf(args) ?? "").trim().split("\n")[0];
  return line.length > 90 ? line.slice(0, 90) + "…" : line;
}

function Badge({
  tone,
  children,
  title,
}: {
  tone: "ok" | "warn" | "err" | "muted";
  children: React.ReactNode;
  title?: string;
}) {
  const tones = {
    ok: "bg-[color-mix(in_srgb,var(--digital-green)_12%,white)] text-[var(--digital-green)]",
    warn: "bg-[color-mix(in_srgb,var(--poppy)_14%,white)] text-[#9a5b00]",
    err: "bg-[color-mix(in_srgb,var(--digital-red)_10%,white)] text-[var(--digital-red)]",
    muted: "bg-[var(--fog-light)] text-[var(--muted)]",
  };
  return (
    <span
      title={title}
      className={`inline-block rounded px-1.5 text-[0.6875rem] font-semibold ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full bg-[var(--fog-light)] px-2.5 py-0.5 text-[0.6875rem] text-[var(--muted)]">
      {label} <b className="font-semibold text-[var(--ink)]">{value}</b>
    </span>
  );
}

/* ---------- steps ---------- */
const MESSAGE_TITLE: Record<MessageStep["label"], string> = {
  system: "System prompt",
  task: "Task",
  harness: "Harness message",
  user: "User",
};

function StepHead({
  title,
  meta,
  index,
}: {
  title: React.ReactNode;
  meta?: string;
  index: number;
}) {
  return (
    <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2 text-[0.75rem] text-[var(--muted)]">
      <span className="text-[0.75rem] font-semibold text-[var(--ink)]">
        {title}
      </span>
      <span>
        #{index}
        {meta ? ` · ${meta}` : ""}
      </span>
    </div>
  );
}

function Message({
  step,
  index,
  query,
  expandAll,
}: {
  step: MessageStep;
  index: number;
  query: string;
  expandAll: boolean;
}) {
  const harness = step.label === "harness";
  const title = step.participant
    ? `${MESSAGE_TITLE[step.label]} · ${step.participant}`
    : MESSAGE_TITLE[step.label];
  return (
    <>
      <StepHead
        title={title}
        index={index}
        meta={harness ? "sent by the agent harness, not a person" : undefined}
      />
      <Block
        text={step.content}
        lines={harness ? 12 : 8}
        query={query}
        expandAll={expandAll}
        truncatedFrom={step.truncated?.content}
      />
    </>
  );
}

function Llm({
  step,
  index,
  turn,
  query,
  expandAll,
  showReasoning,
  timed,
}: {
  step: LlmStep;
  index: number;
  turn: number;
  query: string;
  expandAll: boolean;
  showReasoning: boolean;
  timed: boolean;
}) {
  const u = step.usage ?? {};
  const meta = [
    u.prompt != null ? `${int(u.prompt)} in` : null,
    u.completion != null
      ? `${int(u.completion)} out` +
        (u.reasoning ? ` (${int(u.reasoning)} reasoning)` : "")
      : null,
    u.cacheRead ? `${int(u.cacheRead)} cached` : null,
    step.cost != null ? money(step.cost) : null,
    timed ? elapsed(step) : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const reasoningHit =
    !!query && !!step.reasoning?.toLowerCase().includes(query);
  const empty = !step.content && !step.reasoning && !step.toolCalls?.length;
  return (
    <>
      <StepHead title={`Model · turn ${turn}`} index={index} meta={meta} />
      {step.reasoning ? (
        <details
          className="mb-1.5"
          open={showReasoning || reasoningHit || undefined}
        >
          <summary className="cursor-pointer text-[0.6875rem] text-[var(--muted)]">
            Reasoning ({step.reasoning.length.toLocaleString()} characters)
          </summary>
          <div className="mt-1 whitespace-pre-wrap break-words text-[0.8125rem] italic text-[var(--text-secondary)]">
            <Highlight text={step.reasoning} query={query} />
          </div>
        </details>
      ) : null}
      {step.content ? (
        <div className="whitespace-pre-wrap break-words text-[0.8125rem] leading-6 text-[var(--ink)]">
          <Highlight text={step.content.trim()} query={query} />
          <Omitted from={step.truncated?.content} shown={step.content.length} />
        </div>
      ) : null}
      {(step.toolCalls ?? []).map((call, i) => (
        <div key={call.id ?? i} className="mt-2">
          <p className="mb-0.5 text-[0.6875rem] text-[var(--muted)]">
            calls{" "}
            <b className="font-semibold text-[var(--lagunita-dark)]">
              {call.name ?? "tool"}
            </b>
          </p>
          <Block
            text={commandOf(call.arguments) ?? ""}
            lines={20}
            query={query}
            expandAll={expandAll}
          />
        </div>
      ))}
      {empty ? (
        <p className="text-[0.8125rem] italic text-[var(--muted)]">
          (empty response: no text and no tool call)
        </p>
      ) : null}
    </>
  );
}

function Tool({
  step,
  index,
  query,
  expandAll,
  timed,
  showCommand,
}: {
  step: ToolStep;
  index: number;
  query: string;
  expandAll: boolean;
  timed: boolean;
  /** The command is shown on the model turn above; repeat it only when that
   *  turn made several calls, or the result matches none of its calls. */
  showCommand: boolean;
}) {
  const badge = step.error ? (
    <Badge tone="err">error</Badge>
  ) : step.exitCode == null ? (
    <Badge tone="muted">no exit code</Badge>
  ) : (
    <Badge tone={step.exitCode === 0 ? "ok" : "warn"}>
      exit {step.exitCode}
    </Badge>
  );
  const time = (timed && elapsed(step)) || undefined;
  return (
    <>
      <StepHead
        index={index}
        meta={time}
        title={
          <span className="flex min-w-0 flex-wrap items-baseline gap-1.5">
            <span>↳ {step.name ?? "tool"} output</span>
            {badge}
            {showCommand && step.arguments ? (
              <span className="font-mono font-normal text-[var(--muted)]">
                {firstLine(step.arguments)}
              </span>
            ) : null}
          </span>
        }
      />
      {step.error ? (
        <div className="mb-1.5">
          <Block
            text={step.error}
            lines={10}
            query={query}
            expandAll={expandAll}
            truncatedFrom={step.truncated?.error}
          />
        </div>
      ) : null}
      {step.output ? (
        <Block
          text={step.output}
          lines={16}
          query={query}
          expandAll={expandAll}
          truncatedFrom={step.truncated?.output}
        />
      ) : (
        <p className="text-[0.8125rem] italic text-[var(--muted)]">
          (no output)
        </p>
      )}
    </>
  );
}

/** The harness's exit status, as it reported it. Its vocabulary differs per
 *  harness, so only a recorded exception marks it as an error. */
function StatusBadge({ run }: { run: NonNullable<Trace["run"]> }) {
  return (
    <Badge
      tone={run.exception ? "err" : "muted"}
      title="Exit status reported by the agent harness"
    >
      {run.status ?? "status not recorded"}
    </Badge>
  );
}

/* ---------- result ---------- */
/** The test list opens on the tests that did not pass; a status pill narrows it to that status. */
const NOT_PASSED = "not passed";

function Result({ trace }: { trace: Trace }) {
  const [filter, setFilter] = useState("");
  const [status, setStatus] = useState(NOT_PASSED);
  const run = trace.run ?? {};
  const ev = trace.evaluation;
  const tests = ev?.tests ?? [];
  // Counted over the whole list; the page may have received only part of it.
  const counts = useMemo(() => {
    if (ev?.testCounts) return ev.testCounts;
    const c: Record<string, number> = {};
    for (const t of ev?.tests ?? []) c[t.status] = (c[t.status] ?? 0) + 1;
    return c;
  }, [ev]);
  const statuses = Object.entries(counts).sort(
    ([a, x], [b, y]) =>
      Number(a === "passed") - Number(b === "passed") || y - x,
  );
  const total = ev?.testsTotal ?? tests.length;
  const notPassedCount = total - (counts.passed ?? 0);
  const expected =
    status === NOT_PASSED ? notPassedCount : (counts[status] ?? 0);
  const inStatus = tests.filter((t) =>
    status === NOT_PASSED ? t.status !== "passed" : t.status === status,
  );
  const matching = inStatus.filter(
    (t) => !filter || t.name.toLowerCase().includes(filter),
  );
  const shown = matching.slice(0, TESTS_SHOWN);
  const h3 = "mb-1.5 mt-4 text-[0.75rem] font-semibold text-[var(--ink)]";
  const pill = (key: string, label: string, n: number) => (
    <button
      key={key}
      type="button"
      aria-pressed={status === key}
      onClick={() => setStatus(status === key ? NOT_PASSED : key)}
      className={`rounded-full px-2.5 py-0.5 text-[0.6875rem] transition-colors ${
        status === key
          ? "bg-[var(--ink)] text-white"
          : "bg-[var(--fog-light)] text-[var(--muted)] hover:bg-[var(--line)]"
      }`}
    >
      {label}{" "}
      <b
        className={`font-semibold ${status === key ? "" : "text-[var(--ink)]"}`}
      >
        {n.toLocaleString()}
      </b>
    </button>
  );
  return (
    <section className="mt-4 rounded border border-[var(--line)] bg-white p-4">
      <h2 className="mb-2 text-[0.8125rem] font-semibold text-[var(--ink)]">
        Result
      </h2>
      <div className="flex flex-wrap items-center gap-1.5 text-[0.75rem] text-[var(--muted)]">
        Run status: <StatusBadge run={run} />
      </div>
      {!ev ? (
        <p className="mt-2 text-[0.75rem] text-[var(--muted)]">
          This attempt was not graded.
        </p>
      ) : null}
      {ev?.errorCode ? (
        <>
          <h3 className={h3}>Evaluation error: {ev.errorCode}</h3>
          <Block
            text={ev.errorDetails || "(no details)"}
            lines={12}
            query=""
            expandAll={false}
          />
        </>
      ) : null}
      {ev?.judge ? (
        <>
          <h3 className={h3}>
            Judge{ev.judge.participant ? ` · ${ev.judge.participant}` : ""}
          </h3>
          <Block
            text={ev.judge.content}
            lines={12}
            query=""
            expandAll={false}
          />
        </>
      ) : null}
      {run.exception ? (
        <>
          <h3 className={h3}>Exception</h3>
          <Block
            text={run.exception + (run.traceback ? "\n\n" + run.traceback : "")}
            lines={12}
            query=""
            expandAll={false}
          />
        </>
      ) : null}
      {run.submission ? (
        <>
          <h3 className={h3}>Submission</h3>
          <Block text={run.submission} lines={12} query="" expandAll={false} />
        </>
      ) : null}
      {tests.length ? (
        <>
          <h3 className={h3}>
            Tests ({total.toLocaleString()} in this run&apos;s evaluation)
          </h3>
          <div
            role="group"
            aria-label="Show tests by status"
            className="mb-1.5 flex flex-wrap gap-1.5"
          >
            {pill(NOT_PASSED, NOT_PASSED, notPassedCount)}
            {statuses.map(([s, n]) => pill(s, s, n))}
          </div>
          {expected ? (
            <input
              type="search"
              placeholder="Filter tests…"
              aria-label="Filter tests"
              onChange={(e) => setFilter(e.target.value.trim().toLowerCase())}
              className="mb-1.5 w-full max-w-sm rounded border border-[var(--line)] px-2 py-1 text-[0.75rem]"
            />
          ) : null}
          <div className="max-h-80 overflow-auto rounded border border-[var(--line)]">
            {shown.length ? (
              shown.map((t, i) => (
                <div
                  key={i}
                  className="flex gap-2 border-b border-[var(--line)] px-2 py-0.5 font-mono text-[0.6875rem] leading-5 last:border-b-0"
                >
                  <Badge
                    tone={
                      t.status === "passed"
                        ? "ok"
                        : /fail|error/i.test(t.status)
                          ? "err"
                          : "warn"
                    }
                  >
                    {t.status}
                  </Badge>
                  <span className="break-all">{t.name}</span>
                </div>
              ))
            ) : (
              <p className="px-2 py-1 text-[0.75rem] italic text-[var(--muted)]">
                {filter && inStatus.length
                  ? "No match."
                  : status === NOT_PASSED && !expected
                    ? "Every test passed."
                    : inStatus.length < expected
                      ? "These tests are not loaded (see below)."
                      : "No tests."}
              </p>
            )}
          </div>
          {matching.length > shown.length ? (
            <p className="mt-1 text-[0.6875rem] text-[var(--muted)]">
              Showing the first {shown.length.toLocaleString()} of{" "}
              {matching.length.toLocaleString()}; filter to narrow the list.
            </p>
          ) : null}
          {inStatus.length < expected ? (
            <p className="mt-1 text-[0.6875rem] text-[var(--muted)]">
              {inStatus.length.toLocaleString()} of {expected.toLocaleString()}{" "}
              loaded: the page receives at most {tests.length.toLocaleString()}{" "}
              of this run&apos;s {total.toLocaleString()} tests, those that did
              not pass first.
            </p>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

/* ---------- viewer ---------- */
/* Inline, the viewer is part of the page: it has no scroll area of its own, and
   its toolbar sticks below the site header (which publishes its height as
   --rd-header-offset). Full screen is an overlay with its own scroll area. */
const PAGE_TOP = "var(--rd-header-offset, 0px)";
/** The site header can slide back in after a jump up, so leave room for it. */
const HEADER_ROOM = 80;

export function TraceViewer({
  trace,
  subject,
  onBack,
  paging,
}: {
  trace: Trace;
  /** The matrix row this run belongs to. */
  subject?: string | null;
  /** Scrolls back to where the run was picked from. */
  onBack?: () => void;
  /** Whole-trace counts accompany a bounded window of display steps. */
  paging?: {
    offset: number;
    total: number;
    turnOffset: number;
    modelTurns: number;
    toolResults: number;
    model?: string;
    hasReasoning: boolean;
    navigation: React.ReactNode;
    details?: React.ReactNode;
    stepDetails?: (step: number) => React.ReactNode;
    query: string;
    focus?: number;
    onFocus: (step: number) => void;
  };
}) {
  const [input, setInput] = useState("");
  const [localQuery, setQuery] = useState("");
  const query = paging?.query ?? localQuery;
  const [hitPos, setHitPos] = useState(-1);
  const [focus, setFocus] = useState<number | null>(null);
  const [showReasoning, setShowReasoning] = useState(false);
  const [expandAll, setExpandAll] = useState(false);
  const [fullScreen, setFullScreen] = useState(false);
  const overlay = useRef<HTMLDivElement>(null);
  const toolbar = useRef<HTMLDivElement>(null);
  const timeline = useRef<HTMLDivElement>(null);
  const toggle = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (paging?.focus == null) return;
    const element = timeline.current?.querySelector<HTMLElement>(
      `[data-step="${paging.focus}"]`,
    );
    if (element) {
      element.style.scrollMarginTop = `${(toolbar.current?.offsetHeight ?? 0) + HEADER_ROOM}px`;
      element.scrollIntoView({ block: "start" });
    }
  }, [paging?.focus, paging?.offset]);

  useEffect(() => {
    const t = setTimeout(() => {
      setQuery(input.trim().toLowerCase());
      setHitPos(-1);
    }, 250);
    return () => clearTimeout(t);
  }, [input]);

  const wasFullScreen = useRef(false);
  useEffect(() => {
    if (!fullScreen) {
      // Closing re-creates the toggle button inline; give it focus back.
      if (wasFullScreen.current) toggle.current?.focus();
      wasFullScreen.current = false;
      return;
    }
    wasFullScreen.current = true;
    overlay.current?.focus();
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullScreen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [fullScreen]);

  const steps = trace.steps;
  const kinds = useMemo(() => steps.map(kindOf), [steps]);
  const hits = useMemo(
    () =>
      query
        ? steps.flatMap((s, i) =>
            searchText(s).toLowerCase().includes(query) ? [i] : [],
          )
        : [],
    [steps, query],
  );
  const hitSet = useMemo(() => new Set(hits), [hits]);
  // How many calls the model turn that made each call made in total.
  const callsInTurn = useMemo(() => {
    const m = new Map<string, number>();
    for (const s of steps)
      if (s.type === "llm")
        for (const c of s.toolCalls ?? [])
          if (c.id) m.set(c.id, s.toolCalls!.length);
    return m;
  }, [steps]);
  const timed = trace.timing === "recorded" || trace.timing === "partial";

  const llm = steps.filter((s): s is LlmStep => s.type === "llm");
  const toolCount = steps.filter((s) => s.type === "tool").length;
  const model = paging?.model ?? llm.find((s) => s.model)?.model;
  const hasReasoning = paging?.hasReasoning ?? llm.some((s) => s.reasoning);
  const run = trace.run ?? {};
  const ev = trace.evaluation;

  function jump(i: number) {
    setFocus(i);
    paging?.onFocus(i);
    const el = timeline.current?.querySelector<HTMLElement>(
      `[data-step="${i}"]`,
    );
    if (!el) return;
    const bar = toolbar.current?.offsetHeight ?? 0;
    el.style.scrollMarginTop = `${bar + (fullScreen ? 8 : HEADER_ROOM)}px`;
    el.scrollIntoView({ block: "start" });
  }
  function nextHit(dir: number) {
    if (!hits.length) return;
    const pos = (hitPos + dir + hits.length) % hits.length;
    setHitPos(pos);
    jump(hits[pos]);
  }

  let turn = paging?.turnOffset ?? 0;
  const flagged = (Object.keys(KIND) as Kind[]).filter(
    (k) => KIND[k].flagged && kinds.includes(k),
  );
  const agentOf = (s: TraceStep) => s.agent ?? null;
  const button =
    "rounded border border-[var(--line)] bg-white px-2 py-0.5 text-[0.6875rem] text-[var(--muted)] hover:border-[var(--muted)]";

  const content = (
    <>
      {/* Header */}
      <section className="rounded-t border-b border-[var(--line)] bg-white p-4">
        <div className="flex flex-wrap items-center gap-2 text-[0.75rem]">
          {subject ? (
            <span className="font-semibold text-[var(--ink)]">{subject}</span>
          ) : null}
          {model ? (
            <span
              className="font-mono text-[0.6875rem] text-[var(--muted)]"
              title="Model ID the provider reported for this run"
            >
              {model}
            </span>
          ) : null}
          <StatusBadge run={run} />
          {ev?.errorCode ? (
            <Badge tone="err" title="The grader reported an error; see Result">
              evaluation: {ev.errorCode}
            </Badge>
          ) : null}
          {ev?.total && ev.notRun === ev.total ? (
            <Badge tone="warn">no tests ran</Badge>
          ) : null}
          {!ev ? <Badge tone="muted">not graded</Badge> : null}
          <button
            ref={toggle}
            type="button"
            onClick={() => setFullScreen(!fullScreen)}
            className={`ml-auto ${button}`}
          >
            {fullScreen ? "Close full screen" : "Open full screen"}
          </button>
        </div>
        {ev?.total ? (
          <div className="mt-2 flex flex-wrap items-center gap-3 text-[0.75rem] text-[var(--muted)]">
            <div
              className="h-2 w-48 overflow-hidden rounded bg-[var(--fog-light)]"
              role="img"
              aria-label={`${ev.passed ?? 0} of ${ev.total} tests passed`}
            >
              <div
                className="h-full bg-[var(--digital-green)]"
                style={{ width: `${((ev.passed ?? 0) / ev.total) * 100}%` }}
              />
            </div>
            <span>
              <b className="text-[var(--ink)]">
                {(ev.passed ?? 0).toLocaleString()} /{" "}
                {ev.total.toLocaleString()}
              </b>{" "}
              tests passed in this run&apos;s evaluation
            </span>
          </div>
        ) : null}
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Chip
            label="cost"
            value={run.cost != null ? money(run.cost) : "not recorded"}
          />
          <Chip
            label="model turns"
            value={int(paging?.modelTurns ?? llm.length)}
          />
          {run.apiCalls != null && run.apiCalls !== llm.length ? (
            <Chip label="API calls" value={int(run.apiCalls)} />
          ) : null}
          <Chip
            label="tool calls"
            value={int(paging?.toolResults ?? toolCount)}
          />
          <Chip label="prompt tokens" value={int(run.promptTokens)} />
          <Chip label="output tokens" value={int(run.completionTokens)} />
          {timed ? (
            <Chip label="duration" value={duration(run.durationS)} />
          ) : null}
        </div>
      </section>

      {/* Toolbar: search and the step strip */}
      <div
        ref={toolbar}
        className="sticky z-10 border-b border-[var(--line)] bg-white/95 px-4 pb-2 pt-3 backdrop-blur transition-[top] duration-300"
        style={{ top: fullScreen ? 0 : PAGE_TOP }}
      >
        <div className="flex flex-wrap items-center gap-2">
          {paging ? (
            paging.navigation
          ) : (
            <>
              <input
                type="search"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    const q = input.trim().toLowerCase();
                    if (q !== query) setQuery(q);
                    else nextHit(e.shiftKey ? -1 : 1);
                  }
                }}
                placeholder="Search this trace (commands, outputs, messages)…"
                aria-label="Search this trace"
                className="min-w-44 flex-1 rounded border border-[var(--line)] px-2.5 py-1 text-[0.8125rem]"
              />
              <span
                className="min-w-16 text-[0.6875rem] text-[var(--muted)]"
                aria-live="polite"
              >
                {query
                  ? hitPos >= 0
                    ? `${hitPos + 1} / ${hits.length} steps`
                    : `${hits.length} step${hits.length === 1 ? "" : "s"}`
                  : ""}
              </span>
              {[
                ["↑", -1, "Previous match"],
                ["↓", 1, "Next match"],
              ].map(([label, dir, name]) => (
                <button
                  key={name as string}
                  type="button"
                  aria-label={name as string}
                  disabled={!hits.length}
                  onClick={() => nextHit(dir as number)}
                  className="rounded border border-[var(--line)] bg-white px-2 py-0.5 text-[0.75rem] disabled:opacity-40"
                >
                  {label}
                </button>
              ))}
            </>
          )}
          {hasReasoning ? (
            <label className="flex cursor-pointer items-center gap-1 text-[0.6875rem] text-[var(--muted)]">
              <input
                type="checkbox"
                checked={showReasoning}
                onChange={(e) => setShowReasoning(e.target.checked)}
              />
              show reasoning
            </label>
          ) : null}
          <label className="flex cursor-pointer items-center gap-1 text-[0.6875rem] text-[var(--muted)]">
            <input
              type="checkbox"
              checked={expandAll}
              onChange={(e) => setExpandAll(e.target.checked)}
            />
            expand all output
          </label>
          {onBack && !fullScreen ? (
            <button type="button" onClick={onBack} className={button}>
              ↑ Back to matrix
            </button>
          ) : null}
        </div>
        <div
          className="mt-2 flex h-4 overflow-hidden rounded"
          style={{ background: STRIP_NEUTRAL }}
          role="group"
          aria-label="Steps; select one to jump to it"
        >
          {kinds.map((k, i) => (
            <button
              key={i}
              type="button"
              tabIndex={-1}
              aria-label={`Step ${i + (paging?.offset ?? 0)}: ${KIND[k].label}`}
              title={`#${i + (paging?.offset ?? 0)} ${KIND[k].label}`}
              onClick={() => jump(i + (paging?.offset ?? 0))}
              className="relative min-w-px flex-1 hover:brightness-75"
              style={{
                background: KIND[k].flagged ? KIND[k].color : STRIP_NEUTRAL,
              }}
            >
              {hitSet.has(i) ? (
                <span className="absolute inset-x-0 -top-px h-2 min-w-[3px] bg-[var(--ink)]" />
              ) : null}
              {(paging?.focus ?? focus) === i + (paging?.offset ?? 0) ? (
                <span className="absolute inset-x-0 bottom-0 h-1 min-w-[3px] bg-[var(--lagunita)]" />
              ) : null}
            </button>
          ))}
        </div>
        <div className="mt-1 flex flex-wrap gap-3 text-[0.625rem] text-[var(--muted)]">
          {flagged.map((k) => (
            <span key={k} className="flex items-center gap-1">
              <span
                className="inline-block size-2 rounded-sm"
                style={{ background: KIND[k].color }}
              />
              {KIND[k].label}
            </span>
          ))}
          <span className="flex items-center gap-1">
            <span
              className="inline-block size-2 rounded-sm"
              style={{ background: STRIP_NEUTRAL }}
            />
            {flagged.length ? "other steps" : "steps"} ·{" "}
            {(paging?.total ?? steps.length).toLocaleString()} in all
            {paging && steps.length < paging.total
              ? ` · showing #${paging.offset}–${paging.offset + steps.length - 1}`
              : ""}
          </span>
        </div>
      </div>

      {/* Timeline */}
      <div ref={timeline} className="flex flex-col gap-2 bg-white p-4">
        {steps.map((step, i) => {
          const k = kinds[i];
          const agent = agentOf(step);
          const newAgent = agent && agent !== agentOf(steps[i - 1] ?? step);
          const border =
            step.type === "tool"
              ? k === "tool-err"
                ? "border-l-[3px] border-l-[var(--digital-red)]"
                : k === "tool-nz"
                  ? "border-l-[3px] border-l-[var(--poppy)]"
                  : ""
              : `border-l-[3px]`;
          const tint =
            k === "harness"
              ? "bg-[color-mix(in_srgb,var(--plum)_6%,white)]"
              : step.type === "tool"
                ? "ml-7 bg-[var(--fog-light)]"
                : "bg-white";
          return (
            <Fragment key={i + (paging?.offset ?? 0)}>
              {newAgent || (agent && i === 0) ? (
                <h3 className="mt-2 text-[0.6875rem] uppercase tracking-wide text-[var(--muted)]">
                  {agent}
                </h3>
              ) : null}
              <div
                data-step={i + (paging?.offset ?? 0)}
                className={`rounded border border-[var(--line)] px-3 py-2.5 ${border} ${tint} ${(paging?.focus ?? focus) === i + (paging?.offset ?? 0) ? "outline outline-2 outline-[var(--lagunita)]" : ""}`}
                style={
                  step.type === "tool"
                    ? undefined
                    : { borderLeftColor: KIND[k].color }
                }
              >
                {paging?.stepDetails?.(i + (paging?.offset ?? 0))}
                {step.type === "message" ? (
                  <Message
                    step={step}
                    index={i + (paging?.offset ?? 0)}
                    query={query}
                    expandAll={expandAll}
                  />
                ) : step.type === "llm" ? (
                  <Llm
                    step={step}
                    index={i + (paging?.offset ?? 0)}
                    turn={++turn}
                    query={query}
                    expandAll={expandAll}
                    showReasoning={showReasoning}
                    timed={timed}
                  />
                ) : step.type === "tool" ? (
                  <Tool
                    step={step}
                    index={i + (paging?.offset ?? 0)}
                    query={query}
                    expandAll={expandAll}
                    timed={timed}
                    showCommand={
                      !step.callId || callsInTurn.get(step.callId) !== 1
                    }
                  />
                ) : (
                  <>
                    <StepHead
                      title={`Event · ${step.name}`}
                      index={i + (paging?.offset ?? 0)}
                    />
                    {step.content ? (
                      <Block
                        text={step.content}
                        lines={8}
                        query={query}
                        expandAll={expandAll}
                      />
                    ) : null}
                  </>
                )}
              </div>
            </Fragment>
          );
        })}
      </div>
      <div className="rounded-b bg-white px-4 pb-4">
        <Result trace={trace} />
        {paging?.details}
      </div>
    </>
  );

  if (!fullScreen) return <div className="min-w-0">{content}</div>;
  // Full screen renders into <body>: an animated ancestor would otherwise trap `fixed`.
  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex flex-col bg-[var(--fog-light)] p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Trace"
    >
      <div
        ref={overlay}
        tabIndex={-1}
        className="flex-1 overflow-auto overscroll-contain rounded bg-white outline-none"
      >
        {content}
      </div>
    </div>,
    document.body,
  );
}
