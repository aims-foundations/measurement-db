"use client";

import { useEffect, useRef, useState } from "react";
import type { RecordSource } from "@/lib/benchmark-records/client";
import type {
  BoundedTrace,
  TraceField,
  TracePage,
  TraceSearch,
} from "@/lib/benchmark-records/trace-preview";
import { TraceViewer } from "../trace-viewer";

const button =
  "rounded border border-[var(--line)] bg-white px-2 py-1 text-xs disabled:opacity-40";

function CompleteField({
  source,
  trace,
  field,
}: {
  source: RecordSource;
  trace: BoundedTrace;
  field: TraceField;
}) {
  const [open, setOpen] = useState(false);
  const [part, setPart] = useState(0);
  const [loaded, setLoaded] = useState<{
    part: number;
    text?: string;
    error?: string;
  }>();
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    source.traceField(trace, field, part, controller.signal).then(
      (text) => {
        if (!controller.signal.aborted) setLoaded({ part, text });
      },
      (error) => {
        if (!controller.signal.aborted)
          setLoaded({ part, error: String(error) });
      },
    );
    return () => controller.abort();
  }, [source, trace, field, part, open]);
  const current = loaded?.part === part ? loaded : undefined;
  const label = `${field.step == null ? "" : `Step #${field.step} · `}${field.path.join(".")}`;
  return (
    <details
      onToggle={(e) => setOpen(e.currentTarget.open)}
      className="my-2 min-w-0 text-xs"
    >
      <summary className="cursor-pointer">
        Complete text: {label} ({field.characters.toLocaleString()} characters)
      </summary>
      {open && (
        <div className="mt-2" aria-label={`Complete text: ${label}`}>
          <div className="flex flex-wrap items-center gap-2">
            <button
              className={button}
              disabled={part === 0}
              onClick={() => setPart(part - 1)}
            >
              Previous text part
            </button>
            <span>
              Part {part + 1} of {field.parts.length}
            </span>
            <button
              className={button}
              disabled={part + 1 === field.parts.length}
              onClick={() => setPart(part + 1)}
            >
              Next text part
            </button>
          </div>
          {current?.error ? (
            <p role="alert">{current.error}</p>
          ) : current?.text != null ? (
            <pre className="mt-2 max-h-[32rem] overflow-auto whitespace-pre-wrap break-words rounded bg-[var(--fog-light)] p-3">
              {current.text}
            </pre>
          ) : (
            <p role="status">Loading complete text…</p>
          )}
        </div>
      )}
    </details>
  );
}

/** Page selection, whole-trace search and cancellation are independent of step rendering. */
export function PagedTraceViewer({
  source,
  trace,
  subject,
  onBack,
}: {
  source: RecordSource;
  trace: BoundedTrace;
  subject: string;
  onBack: () => void;
}) {
  const { header } = trace;
  const [selected, setSelected] = useState({ page: 0, focus: 0 });
  const [loaded, setLoaded] = useState<{
    page: number;
    value?: TracePage;
    error?: string;
  }>();
  const [stepInput, setStepInput] = useState("0");
  const [attempt, setAttempt] = useState(0);
  const [input, setInput] = useState("");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState<TraceSearch>();
  const [hit, setHit] = useState(-1);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState([0, 0]);
  const [error, setError] = useState<string>();
  const searchRequest = useRef<AbortController | null>(null);
  useEffect(() => () => searchRequest.current?.abort(), []);
  useEffect(() => {
    if (selected.page === 0) return;
    const controller = new AbortController();
    source.tracePage(trace, selected.page, controller.signal).then(
      (value) => {
        if (!controller.signal.aborted)
          setLoaded({ page: selected.page, value });
      },
      (error) => {
        if (!controller.signal.aborted)
          setLoaded({ page: selected.page, error: String(error) });
      },
    );
    return () => controller.abort();
  }, [source, trace, selected.page, attempt]);

  function go(step: number) {
    if (!Number.isSafeInteger(step) || step < 0 || step >= header.summary.steps)
      return;
    const page = header.pages.findIndex((p) => p.start <= step && step < p.end);
    if (page < 0) return;
    setSelected({ page, focus: step });
    setStepInput(String(step));
  }
  function nextMatch(direction: number) {
    if (!search?.matches.length) return;
    const index =
      (hit + direction + search.matches.length) % search.matches.length;
    setHit(index);
    go(search.matches[index]);
  }
  function find() {
    searchRequest.current?.abort();
    const controller = new AbortController();
    searchRequest.current = controller;
    const q = input.trim().toLowerCase();
    setQuery(q);
    setSearch(undefined);
    setHit(-1);
    setError(undefined);
    if (!q) {
      setScanning(false);
      return;
    }
    setScanning(true);
    setProgress([0, header.search.length]);
    source
      .searchTrace(trace, q, controller.signal, (done, total) => {
        if (!controller.signal.aborted) setProgress([done, total]);
      })
      .then(
        (value) => {
          if (!controller.signal.aborted) {
            setSearch(value);
            setScanning(false);
          }
        },
        (error) => {
          if (!controller.signal.aborted) {
            setError(String(error));
            setScanning(false);
          }
        },
      );
  }

  const current =
    selected.page === 0
      ? trace.firstPage
      : loaded?.page === selected.page
        ? loaded.value
        : undefined;
  const descriptor = header.pages[selected.page];
  const controls = (
    <>
      <input
        type="search"
        aria-label="Search this trace"
        placeholder="Search messages and outputs…"
        maxLength={256}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            if (input.trim().toLowerCase() === query && search)
              nextMatch(e.shiftKey ? -1 : 1);
            else find();
          }
        }}
        className="min-w-40 flex-1 rounded border border-[var(--line)] px-2 py-1 text-sm"
      />
      <button className={button} onClick={find}>
        Search
      </button>
      {scanning && (
        <button
          className={button}
          onClick={() => {
            searchRequest.current?.abort();
            setScanning(false);
          }}
        >
          Cancel search
        </button>
      )}
      <span aria-live="polite" className="text-xs text-[var(--muted)]">
        {scanning
          ? `Searching ${progress[0]} / ${progress[1]} pages…`
          : search
            ? hit >= 0
              ? `${hit + 1} / ${search.total} steps`
              : `${search.total} matching steps`
            : ""}
      </span>
      <button
        className={button}
        aria-label="Previous match"
        disabled={!search?.matches.length}
        onClick={() => nextMatch(-1)}
      >
        ↑
      </button>
      <button
        className={button}
        aria-label="Next match"
        disabled={!search?.matches.length}
        onClick={() => nextMatch(1)}
      >
        ↓
      </button>
      <div
        className="flex w-full flex-wrap items-center gap-2 text-xs"
        aria-label="Trace step pages"
      >
        <button
          className={button}
          disabled={selected.page === 0}
          onClick={() => go(header.pages[selected.page - 1].start)}
        >
          Previous steps
        </button>
        <span>
          {header.summary.steps
            ? `Steps #${descriptor.start}–${descriptor.end - 1} of ${header.summary.steps}`
            : "0 steps"}
        </span>
        <button
          className={button}
          disabled={selected.page + 1 >= header.pages.length}
          onClick={() => go(header.pages[selected.page + 1].start)}
        >
          Next steps
        </button>
        <label className="flex items-center gap-1">
          Go to step
          <input
            aria-label="Go to step"
            type="number"
            min={0}
            max={Math.max(0, header.summary.steps - 1)}
            value={stepInput}
            onChange={(e) => setStepInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                go(Number(stepInput));
              }
            }}
            className="w-20 rounded border border-[var(--line)] px-2 py-1"
          />
        </label>
        <button
          className={button}
          disabled={!header.summary.steps}
          onClick={() => go(Number(stepInput))}
        >
          Go
        </button>
        <span className="text-[var(--muted)]">
          Search covers complete message and output text across all steps.
        </span>
      </div>
      {search && !search.complete && (
        <p className="w-full text-xs">
          Showing navigation for the first{" "}
          {search.matches.length.toLocaleString()} matching steps.
        </p>
      )}
    </>
  );
  const fields = header.fields;
  const details = fields.length > 0 && (
    <section
      aria-label="Complete trace text"
      className="mt-3 rounded border border-[var(--line)] p-3"
    >
      <p className="text-xs text-[var(--muted)]">
        Long fields can be read in full, one part at a time.
      </p>
      {fields.map((field) => (
        <CompleteField
          key={`${field.step ?? "header"}:${JSON.stringify(field.path)}`}
          source={source}
          trace={trace}
          field={field}
        />
      ))}
    </section>
  );
  return (
    <div
      className="min-w-0"
      data-trace-preview="bounded"
      data-response-id={header.response_id}
    >
      {error && <p role="alert">{error}</p>}
      {loaded?.page === selected.page && loaded.error ? (
        <div role="alert">
          {loaded.error}{" "}
          <button className={button} onClick={() => setAttempt(attempt + 1)}>
            Retry steps
          </button>
        </div>
      ) : null}
      {current ? (
        <TraceViewer
          trace={{ ...header.trace, steps: current.steps }}
          subject={subject}
          onBack={onBack}
          paging={{
            offset: current.start,
            total: header.summary.steps,
            turnOffset: descriptor?.turnStart ?? 0,
            modelTurns: header.summary.modelTurns,
            toolResults: header.summary.toolResults,
            model: header.summary.model,
            hasReasoning: header.summary.hasReasoning,
            navigation: controls,
            details,
            stepDetails: (step) =>
              (current?.fields ?? [])
                .filter((field) => field.step === step)
                .map((field) => (
                  <CompleteField
                    key={JSON.stringify(field.path)}
                    source={source}
                    trace={trace}
                    field={field}
                  />
                )),
            query,
            focus: selected.focus,
            onFocus: (step) => {
              setSelected({ ...selected, focus: step });
              setStepInput(String(step));
            },
          }}
        />
      ) : (
        <div className="flex flex-wrap gap-2 rounded border border-[var(--line)] p-4">
          {controls}
          <p role="status">Loading trace steps…</p>
        </div>
      )}
    </div>
  );
}
