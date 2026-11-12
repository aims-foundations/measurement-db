"use client";

import { useState } from "react";
import type { RecordSource } from "@/lib/benchmark-records/client";
import { useRecord } from "./use-records";
import { SourceError } from "./source-error";
import { CodeBlock } from "../code-block";

function OriginalRecord({ source, id }: { source: RecordSource; id: string }) {
  const record = useRecord(source, "traces", id);
  if (record.status === "loading")
    return <p role="status">Loading original trace…</p>;
  if (record.status === "error")
    return (
      <SourceError
        source={source}
        table="traces"
        message={record.message}
        paged={record.paged}
        onRetry={record.retry}
      />
    );
  return (
    <div aria-label="Original trace">
      <CodeBlock
        text={record.value?.trace ?? "No trace released."}
        language="Original text"
      />
    </div>
  );
}

/** Fetch the full source only when requested; display caps never alter the source. */
export function OriginalTrace({
  source,
  id,
}: {
  source: RecordSource;
  id: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details
      className="mt-3 text-xs text-[var(--muted)]"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer">Original trace</summary>
      {open && <OriginalRecord source={source} id={id} />}
    </details>
  );
}
