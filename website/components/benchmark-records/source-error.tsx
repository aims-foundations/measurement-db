import type { PagedRecord } from "@/lib/benchmark-records/record-preview";
import { PagedRecordView } from "./paged-record";
import type { RecordSource } from "@/lib/benchmark-records/client";
import type { TableName } from "@/lib/benchmark-records/types";

export function SourceError({
  source,
  table,
  message,
  paged,
  onRetry,
}: {
  source: RecordSource;
  table: TableName;
  message: string;
  paged?: PagedRecord;
  onRetry?: () => void;
}) {
  if (paged) return <PagedRecordView record={paged} />;
  const parquet = source.manifest.tables[table]?.parquet;
  const download = parquet?.download;
  const size =
    parquet &&
    (parquet.bytes >= 1024 ** 3
      ? `${(parquet.bytes / 1024 ** 3).toFixed(1)} GiB`
      : `${(parquet.bytes / 1024 ** 2).toFixed(1)} MiB`);
  return (
    <div role="alert" className="text-sm">
      <p>{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="my-2 rounded border border-[var(--line)] px-3 py-2"
        >
          Retry loading
        </button>
      )}
      {download && (
        <a
          className="mt-2 inline-block text-[var(--lagunita)] underline"
          href={download}
          target="_blank"
          rel="noreferrer"
        >
          Download original {table} table ({size})
        </a>
      )}
    </div>
  );
}
