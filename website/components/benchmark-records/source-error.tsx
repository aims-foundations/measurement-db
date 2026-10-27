import type { RecordSource } from "@/lib/benchmark-records/client";
import type { TableName } from "@/lib/benchmark-records/types";

export function SourceError({
  source,
  table,
  message,
}: {
  source: RecordSource;
  table: TableName;
  message: string;
}) {
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
