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
import {
  displayText,
  type Presentation,
} from "@/lib/benchmark-records/presentation";
import type { Observation, ResponseScale } from "@/lib/benchmark-records/types";
import { Pagination } from "./pagination";
import { useRecord } from "./use-records";
import { RichText, SourceText } from "./rich-text";
import { AnswerPanel } from "./answer-panel";
import { Attachment } from "./attachment";
import { CONTROL } from "./view-controls";
import { TaskContent } from "../task-content";

function Metadata({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="my-3 text-xs text-[var(--muted)]">
      <summary className="cursor-pointer">{title}</summary>
      <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded bg-black/5 p-3">
        {JSON.stringify(value, null, 2)}
      </pre>
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
  if (item.status === "loading")
    return (
      <p role="status" className="py-6 text-sm">
        Loading task…
      </p>
    );
  if (item.status === "error")
    return (
      <p role="alert" className="py-6 text-sm">
        {item.message}
      </p>
    );
  if (!item.value) return <p role="alert">The selected task is unavailable.</p>;
  const prepared = prepareItem(item.value, scale),
    observation = observations[attempt];
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
            {selection.subject?.display_name ?? "Task"}
            {opponentName && (
              <span className="text-[var(--muted)]"> vs {opponentName}</span>
            )}
          </h3>
          <p className="mt-1 break-all font-mono text-[0.625rem] text-[var(--muted)]">
            {selection.itemId}
          </p>
        </div>
        <button className={CONTROL} onClick={onClose}>
          Close
        </button>
      </div>
      <div className="mt-4 grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <section className="min-w-0">
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--muted)]">
            Task
          </h4>
          <div
            className="max-h-[28rem] overflow-auto rounded bg-white p-4"
            aria-label="Task content"
          >
            <TaskContent
              format={presentation.taskFormat}
              text={
                displayText(item.value.content, presentation.promptPath) ??
                "No text component released."
              }
            />
          </div>
          {item.value.content && (
            <SourceText text={item.value.content} label="Task" />
          )}
          {prepared.attachments.map((link, i) => (
            <Attachment
              key={`${link.asset_id}-${i}`}
              source={source}
              link={link}
            />
          ))}
          {prepared.criterion.reference_answer != null && (
            <div className="mt-4">
              <h4 className="mb-2 text-xs font-medium text-[var(--muted)]">
                Reference answer
              </h4>
              <div className="text-[var(--palo-alto)]">
                <RichText text={prepared.criterion.reference_answer} />
              </div>
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
                      prepared.scale,
                    ),
                  }}
                />
                {gradeLabel(observation.response, prepared.scale)}
              </p>
              <p className="mt-2 text-xs text-[var(--muted)]">
                Trial {observation.trial} · Attempt {attempt + 1} of{" "}
                {observations.length}
              </p>
            </>
          ) : (
            <p className="text-sm text-[var(--muted)]">
              No recorded observation.
            </p>
          )}
          {prepared.criterion.rule && (
            <div className="mt-4">
              <h5 className="mb-1 text-xs font-medium text-[var(--muted)]">
                Grading rule
              </h5>
              <RichText text={prepared.criterion.rule} />
            </div>
          )}
          <Metadata title="Task metadata" value={item.value} />
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
            label="attempts"
            page={attempt}
            total={observations.length}
            size={1}
            onChange={setAttempt}
          />
        </div>
      )}
      {observation && (
        <div className={`mt-5 grid gap-5 ${opponent ? "lg:grid-cols-2" : ""}`}>
          <AnswerPanel
            key={observation.response_id}
            source={source}
            observation={observation}
            name={selection.subject?.display_name ?? "Model answer"}
            presentation={presentation}
          />
          {opponent && (
            <AnswerPanel
              key={opponent.response_id}
              source={source}
              observation={opponent}
              name={opponentName!}
              presentation={presentation}
            />
          )}
        </div>
      )}
    </section>
  );
}
