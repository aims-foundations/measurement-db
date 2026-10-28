"use client";

import { useState } from "react";
import type { RecordSource } from "@/lib/benchmark-records/client";
import {
  prepareItem,
  type CellSelection,
  type MatrixModel,
} from "@/lib/benchmark-records/model";
import { gradeLabel } from "@/lib/benchmark-records/scale";
import { gradeColor } from "@/lib/benchmark-records/colors";
import { type Presentation } from "@/lib/benchmark-records/presentation";
import type { Observation, ResponseScale } from "@/lib/benchmark-records/types";
import { Pagination } from "./pagination";
import { useRecord } from "./use-records";
import { SourceText } from "./rich-text";
import { AnswerPanel } from "./answer-panel";
import { Attachment } from "./attachment";
import { CONTROL } from "./view-controls";
import { ItemContent } from "../item-content";
import { StructuredContent } from "../structured-content";
import { metadataFields } from "@/lib/structured-content";
import { SourceError } from "./source-error";

function Metadata({
  title,
  value,
}: {
  title: string;
  value: Record<string, unknown>;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details
      onToggle={(event) => setOpen(event.currentTarget.open)}
      className="my-3 rounded border border-[var(--line)] bg-white p-3 text-xs"
    >
      <summary className="cursor-pointer">{title}</summary>
      {open && (
        <div
          className="mt-4 max-h-80 overflow-auto break-words"
          aria-label={title}
        >
          <StructuredContent value={metadataFields(value)} />
          <SourceText text={JSON.stringify(value, null, 2)} label={title} />
        </div>
      )}
    </details>
  );
}

export function CellDetails({
  source,
  selection,
  observations,
  scale,
  presentation,
  model,
  pairs,
  onClose,
}: {
  source: RecordSource;
  selection: CellSelection;
  observations: Observation[];
  scale: ResponseScale;
  presentation: Presentation;
  model: MatrixModel;
  pairs: Map<string, Observation>;
  onClose: () => void;
}) {
  const item = useRecord(source, "items", selection.itemId);
  const [attempt, setAttempt] = useState(0);
  const value = item.status === "ready" ? item.value : null;
  const prepared = value ? prepareItem(value, scale) : undefined;
  const observation = observations[attempt];
  const opponent = observation && pairs.get(observation.response_id);
  const opponentName = model.subjects.find(
    (s) => s.subject_id === opponent?.subject_id,
  )?.display_name;
  return (
    <section
      className="mt-5 rounded-md border border-[var(--line)] bg-[var(--fog-light)] p-4 sm:p-5"
      aria-label="Cell detail"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="break-words text-sm font-medium">
            {opponent
              ? "Model comparison"
              : (selection.subject?.display_name ?? "Item")}
          </h3>
          <p className="mt-1 break-all font-mono text-[0.625rem] text-[var(--muted)]">
            Item ID: {selection.itemId}
          </p>
          {value?.testlet_id && (
            <p className="mt-2 break-all text-xs text-[var(--muted)]">
              Testlet: {value.testlet_id}
            </p>
          )}
        </div>
        <button className={CONTROL} onClick={onClose}>
          Close
        </button>
      </div>
      <div className="mt-4 grid items-start gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <section className="min-w-0 rounded-md border border-[var(--line)] bg-white p-4 sm:p-5">
          <h4 className="mb-4 border-b border-[var(--line)] pb-3 text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
            Item
          </h4>
          <div aria-label="Item content">
            {item.status === "loading" ? (
              <p role="status">Loading item…</p>
            ) : item.status === "error" ? (
              <SourceError
                source={source}
                table="items"
                message={item.message}
                paged={item.paged}
                onRetry={item.retry}
              />
            ) : value && prepared ? (
              <ItemContent
                key={selection.itemId}
                benchmark={source.manifest.benchmark.benchmark_id}
                text={value.content}
                assets={prepared.attachments}
                renderAsset={(link) => (
                  <Attachment source={source} link={link} />
                )}
              />
            ) : (
              <p>The selected item is unavailable.</p>
            )}
          </div>
          {prepared?.criterion.reference_answer != null && (
            <div className="mt-4">
              <h4 className="mb-2 text-xs font-medium text-[var(--muted)]">
                Reference answer
              </h4>
              <div
                aria-label="Reference answer"
                className="max-h-[32rem] overflow-auto break-words"
              >
                <StructuredContent
                  value={prepared.criterion.reference_answer}
                />
              </div>
              <SourceText
                text={prepared.criterion.reference_answer}
                label="Reference answer"
              />
            </div>
          )}
        </section>
        <section className="min-w-0">
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--muted)]">
            Recorded result
          </h4>
          {observation ? (
            <>
              <p className="flex items-center gap-2 text-sm">
                <span
                  className="h-3 w-3 shrink-0 rounded-sm"
                  style={{
                    background: gradeColor(
                      observation.response,
                      prepared?.scale ?? scale,
                    ),
                  }}
                />
                <span className="min-w-0 break-words">
                  {prepared
                    ? gradeLabel(observation.response, prepared.scale)
                    : (observation.response ?? "Ungraded")}
                </span>
              </p>
              <p className="mt-2 text-xs text-[var(--muted)]">
                Trial {observation.trial} · Observation {attempt + 1} of{" "}
                {observations.length}
                {model.partialItemIds?.includes(selection.itemId)
                  ? " on this page"
                  : ""}
              </p>
            </>
          ) : (
            <p className="text-sm text-[var(--muted)]">
              {model.partialItemIds?.includes(selection.itemId)
                ? "No observation on this page."
                : "No recorded observation."}
            </p>
          )}
          {prepared?.criterion.rule && (
            <section className="mt-4" aria-label="Grading rule">
              <h5 className="mb-1 text-xs font-medium text-[var(--muted)]">
                Grading rule
              </h5>
              <div
                className="max-h-[32rem] overflow-auto break-words"
                aria-label="Formatted grading rule"
              >
                <StructuredContent value={prepared.criterion.rule} />
              </div>
              <SourceText text={prepared.criterion.rule} label="Grading rule" />
            </section>
          )}
          {value && <Metadata title="Item metadata" value={value} />}
          {selection.subject && (
            <Metadata title="Model metadata" value={selection.subject} />
          )}
          {observation && (
            <Metadata title="Observation metadata" value={observation} />
          )}
        </section>
      </div>
      {observations.length > 1 && (
        <div className="mt-5">
          <Pagination
            label="observations"
            page={attempt}
            total={observations.length}
            size={1}
            onChange={setAttempt}
          />
        </div>
      )}
      {observation && (
        <div
          className={`mt-6 grid gap-5 border-t border-[var(--line)] pt-6 ${opponent ? "lg:grid-cols-2" : ""}`}
        >
          <AnswerPanel
            key={observation.response_id}
            source={source}
            observation={observation}
            name={selection.subject?.display_name ?? "Selected model"}
            showName={Boolean(opponent)}
            presentation={presentation}
          />
          {opponent && (
            <AnswerPanel
              key={opponent.response_id}
              source={source}
              observation={opponent}
              name={opponentName ?? opponent.subject_id}
              showName
              presentation={presentation}
            />
          )}
        </div>
      )}
    </section>
  );
}
