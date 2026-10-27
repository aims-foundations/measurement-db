"use client";

import { useState } from "react";
import type { PagedRecord } from "@/lib/benchmark-records/record-preview";
import { fieldLabel } from "@/lib/structured-content";
import { usePreviewPart } from "./use-records";
import { Pagination } from "./pagination";
import { StructuredContent } from "../structured-content";
import { CodeBlock } from "../code-block";

function RecordField({
  record,
  field,
}: {
  record: PagedRecord;
  field: string;
}) {
  const [page, setPage] = useState(0);
  const part = usePreviewPart(record, field, page);
  const total = record.fields[field].length;
  return (
    <section className="mt-4 min-w-0" aria-label={fieldLabel(field)}>
      <h5 className="mb-2 text-sm font-medium">{fieldLabel(field)}</h5>
      <Pagination
        label={`${fieldLabel(field)} parts`}
        page={page}
        total={total}
        size={1}
        onChange={setPage}
      />
      <label className="my-2 inline-flex items-center gap-2 text-xs">
        Go to part
        <input
          type="number"
          min={1}
          max={total}
          value={page + 1}
          className="w-20 rounded border border-[var(--line)] p-2"
          onChange={(event) => {
            const value = Number(event.target.value);
            if (Number.isInteger(value) && value >= 1 && value <= total)
              setPage(value - 1);
          }}
        />
      </label>
      {part.status === "loading" ? (
        <p role="status">Loading part…</p>
      ) : part.status === "error" ? (
        <p role="alert">{part.message}</p>
      ) : (
        <CodeBlock text={part.value} language="Original text" />
      )}
    </section>
  );
}

/** Large fields remain complete, but only the selected text part enters the DOM. */
export function PagedRecordView({ record }: { record: PagedRecord }) {
  return (
    <div className="min-w-0 text-sm" aria-label="Record in parts">
      <p className="text-[var(--muted)]">
        This is a long record. Read its full text in parts below.
      </p>
      {Object.keys(record.fields).map((field) => (
        <RecordField key={field} record={record} field={field} />
      ))}
      <details className="mt-4">
        <summary className="cursor-pointer">Other record fields</summary>
        <div className="mt-3">
          <StructuredContent value={record.record} />
        </div>
      </details>
    </div>
  );
}
