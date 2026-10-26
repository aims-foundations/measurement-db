"use client";

import { useState } from "react";
import type { RecordSource } from "@/lib/benchmark-records/client";
import type { CellSelection } from "@/lib/benchmark-records/model";
import { pageOf, prepareItem } from "@/lib/benchmark-records/model";
import { gradeLabel } from "@/lib/benchmark-records/scale";
import type {
  AssetLink,
  Observation,
  ResponseScale,
} from "@/lib/benchmark-records/types";
import { Pagination } from "./pagination";
import { useAttachment, useRecord } from "./use-records";

const ATTEMPTS_PER_PAGE = 10;
const TEXT_STYLE =
  "max-h-[28rem] overflow-auto whitespace-pre-wrap break-words rounded bg-black/[0.03] p-4 text-xs leading-relaxed";

function Metadata({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="my-3 text-sm">
      <summary className="cursor-pointer">{title}</summary>
      <pre className={`${TEXT_STYLE} mt-2`}>
        {JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}

function Answer({
  source,
  responseId,
}: {
  source: RecordSource;
  responseId: string;
}) {
  const [opened, setOpened] = useState(false);
  const answer = useRecord(source, "traces", opened ? responseId : null);
  if (!opened)
    return (
      <button
        className="rounded border border-[var(--line)] px-3 py-2 text-xs hover:bg-black/5 disabled:cursor-not-allowed disabled:opacity-40 mt-3"
        onClick={() => setOpened(true)}
      >
        Load answer
      </button>
    );
  if (answer.status === "loading")
    return (
      <p role="status" className="mt-3 text-sm">
        Loading answer…
      </p>
    );
  if (answer.status === "error")
    return (
      <p role="alert" className="mt-3 text-sm">
        {answer.message}
      </p>
    );
  return (
    <pre className={`${TEXT_STYLE} mt-3`} aria-label="Model answer">
      {answer.value?.trace ?? "No answer text released."}
    </pre>
  );
}

function Attachment({
  source,
  link,
}: {
  source: RecordSource;
  link: AssetLink;
}) {
  const asset = useAttachment(source, link);
  if (asset.status === "loading")
    return <p role="status">Loading {link.path}…</p>;
  if (asset.status === "error") return <p role="alert">{asset.message}</p>;
  return (
    <div className="my-4">
      <a
        className="text-sm underline"
        href={asset.value.url}
        download={link.path}
      >
        Download {link.path} ({link.role})
      </a>
      {asset.value.imageUrl && (
        // The loader supplies exact bytes and a local Blob URL; Next image optimization is unnecessary.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={asset.value.imageUrl}
          alt={link.path}
          className="mt-3 max-h-[28rem] max-w-full object-contain"
        />
      )}
    </div>
  );
}

export function CellDetails({
  source,
  selection,
  observations,
  scale,
  onClose,
}: {
  source: RecordSource;
  selection: CellSelection;
  observations: Observation[];
  scale: ResponseScale;
  onClose: () => void;
}) {
  const item = useRecord(source, "items", selection.itemId);
  const [page, setPage] = useState(0);
  if (item.status === "loading") return <p role="status">Loading task…</p>;
  if (item.status === "error") return <p role="alert">{item.message}</p>;
  if (!item.value) return <p role="alert">The selected task is unavailable.</p>;
  const prepared = prepareItem(item.value, scale);
  return (
    <section
      className="mt-6 rounded-lg border border-[var(--line)] bg-white p-4 sm:p-6"
      aria-label="Cell detail"
    >
      <div className="flex items-start justify-between gap-4">
        <h3 className="min-w-0 break-words text-lg font-medium">
          {selection.subject?.display_name ?? "Task"}
          <span className="mt-1 block break-all font-mono text-xs text-[var(--muted)]">
            {selection.itemId}
          </span>
        </h3>
        <button
          className="rounded border border-[var(--line)] px-3 py-2 text-xs hover:bg-black/5 disabled:cursor-not-allowed disabled:opacity-40"
          onClick={onClose}
        >
          Close
        </button>
      </div>
      <h4 className="mb-2 mt-5 text-sm font-medium">Task</h4>
      <pre className={TEXT_STYLE} aria-label="Task content">
        {item.value.content ?? "No text component released."}
      </pre>
      {prepared.attachments.map((link, i) => (
        <Attachment key={`${link.asset_id}-${i}`} source={source} link={link} />
      ))}
      <Metadata title="Grading criterion" value={prepared.criterion} />
      <Metadata title="Task metadata" value={item.value} />
      {selection.subject && (
        <Metadata title="Model metadata" value={selection.subject} />
      )}
      <h4 className="mb-3 mt-6 font-medium">
        {observations.length.toLocaleString()} recorded{" "}
        {observations.length === 1 ? "attempt" : "attempts"}
      </h4>
      {observations.length > ATTEMPTS_PER_PAGE && (
        <Pagination
          label="attempts"
          page={page}
          total={observations.length}
          size={ATTEMPTS_PER_PAGE}
          onChange={setPage}
        />
      )}
      {pageOf(observations, page, ATTEMPTS_PER_PAGE).map((observation) => (
        <article
          key={observation.response_id}
          className="mt-4 border-t border-[var(--line)] pt-4"
        >
          <h5 className="text-sm font-medium">
            {gradeLabel(observation.response, prepared.scale)} · trial{" "}
            {observation.trial}
          </h5>
          <p className="mt-2 break-words text-xs leading-relaxed text-[var(--muted)]">
            Condition: {observation.test_condition ?? "not recorded"}
            <br />
            Interactors: {observation.interactors ?? "not recorded"}
          </p>
          <Metadata title="Observation metadata" value={observation} />
          <Answer source={source} responseId={observation.response_id} />
        </article>
      ))}
    </section>
  );
}
