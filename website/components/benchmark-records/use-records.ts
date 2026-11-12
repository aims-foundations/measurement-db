"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { withBase } from "@/lib/base-path";
import { RecordSource } from "@/lib/benchmark-records/client";
import {
  PagedRecordError,
  readPreviewPart,
  type PagedRecord,
} from "@/lib/benchmark-records/record-preview";
import { prepareMatrix, type MatrixModel } from "@/lib/benchmark-records/model";
import type {
  AssetLink,
  GalleryEntry,
  TableName,
  TableRows,
  Subject,
} from "@/lib/benchmark-records/types";

export type LoadState<T> =
  | { status: "loading" }
  | { status: "error"; message: string; paged?: PagedRecord }
  | { status: "ready"; value: T };

const failure = (error: unknown): LoadState<never> => ({
  status: "error",
  message: error instanceof Error ? error.message : String(error),
  ...(error instanceof PagedRecordError ? { paged: error.record } : {}),
});

/** Item-ID lookup shares the same cancellation rules as other record requests. */
export function useItemWindow(entry: GalleryEntry) {
  const source = useMemo(
    () => new RecordSource(withBase(entry.base), entry.manifest),
    [entry],
  );
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => pending.current?.abort(), [source]);
  const cancelFind = () => pending.current?.abort();
  const findItem = (id: string) => {
    cancelFind();
    const controller = new AbortController();
    pending.current = controller;
    return source.windowForItem(id, controller.signal);
  };
  return { findItem, cancelFind };
}

/** Request lifecycle is separate from both record preparation and JSX. */
export function useGallery(
  entry: GalleryEntry,
  subjectIds?: string[],
  windowPage?: number,
) {
  const source = useMemo(
    () => new RecordSource(withBase(entry.base), entry.manifest),
    [entry],
  );
  const [result, setResult] = useState<{
    source: RecordSource;
    selection: string;
    state: LoadState<MatrixModel>;
  }>();
  const selection = JSON.stringify({ subjectIds, windowPage });
  useEffect(() => {
    const controller = new AbortController();
    const selected = JSON.parse(selection);
    const request =
      selected.windowPage === undefined
        ? source.load(controller.signal, selected.subjectIds)
        : source.window(selected.windowPage, controller.signal);
    request.then(prepareMatrix).then(
      (value) => {
        if (!controller.signal.aborted)
          setResult({ source, selection, state: { status: "ready", value } });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ source, selection, state: failure(error) });
      },
    );
    return () => controller.abort();
  }, [source, selection]);
  return {
    source,
    state:
      result?.source === source && result.selection === selection
        ? result.state
        : ({ status: "loading" } as LoadState<MatrixModel>),
  };
}

/** The full subject registry remains searchable before response pages are fetched. */
export function useSubjects(entry: GalleryEntry): LoadState<Subject[]> {
  const [result, setResult] = useState<{
    entry: GalleryEntry;
    state: LoadState<Subject[]>;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    const source = new RecordSource(withBase(entry.base), entry.manifest);
    source.subjects(controller.signal).then(
      (value) => {
        if (!controller.signal.aborted)
          setResult({ entry, state: { status: "ready", value } });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ entry, state: failure(error) });
      },
    );
    return () => controller.abort();
  }, [entry]);
  return result?.entry === entry ? result.state : { status: "loading" };
}

export function useRecord<T extends TableName>(
  source: RecordSource,
  table: T,
  id: string | null,
): LoadState<TableRows[T] | null> & { retry: () => void } {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<{
    source: RecordSource;
    table: T;
    id: string;
    state: LoadState<TableRows[T] | null>;
  }>();
  useEffect(() => {
    if (id === null) return;
    const controller = new AbortController();
    source.get(table, id, controller.signal).then(
      (value) => {
        if (!controller.signal.aborted)
          setResult({ source, table, id, state: { status: "ready", value } });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ source, table, id, state: failure(error) });
      },
    );
    return () => controller.abort();
  }, [source, table, id, attempt]);
  const state: LoadState<TableRows[T] | null> =
    result?.source === source && result.table === table && result.id === id
      ? result.state
      : { status: "loading" };
  return {
    ...state,
    retry: () => {
      setResult(undefined);
      setAttempt((value) => value + 1);
    },
  };
}

export function useAttachment(
  source: RecordSource,
  link: AssetLink,
): LoadState<{ url: string; imageUrl: string | null }> {
  const stableLink = useMemo(
    () => ({
      asset_id: link.asset_id,
      path: link.path,
      media_type: link.media_type,
      role: link.role,
    }),
    [link.asset_id, link.path, link.media_type, link.role],
  );
  const [result, setResult] = useState<{
    source: RecordSource;
    link: AssetLink;
    state: LoadState<{ url: string; imageUrl: string | null }>;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    let imageUrl: string | null = null;
    source.asset(stableLink, controller.signal).then(
      ({ url, blob }) => {
        if (controller.signal.aborted) return;
        imageUrl = blob ? URL.createObjectURL(blob) : null;
        setResult({
          source,
          link: stableLink,
          state: {
            status: "ready",
            value: {
              url: imageUrl ?? url,
              imageUrl: stableLink.media_type.startsWith("image/")
                ? imageUrl
                : null,
            },
          },
        });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ source, link: stableLink, state: failure(error) });
      },
    );
    return () => {
      controller.abort();
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [source, stableLink]);
  return result?.source === source && result.link === stableLink
    ? result.state
    : { status: "loading" };
}

/** Text parts use the same cancellation and stale-selection protection as records. */
export function usePreviewPart(
  record: PagedRecord,
  field: string,
  page: number,
): LoadState<string> {
  const [result, setResult] = useState<{
    record: PagedRecord;
    field: string;
    page: number;
    state: LoadState<string>;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    readPreviewPart(
      record.base,
      record.preview,
      record.fields[field][page],
      controller.signal,
    ).then(
      (value) => {
        if (!controller.signal.aborted)
          setResult({ record, field, page, state: { status: "ready", value } });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ record, field, page, state: failure(error) });
      },
    );
    return () => controller.abort();
  }, [record, field, page]);
  return result?.record === record &&
    result.field === field &&
    result.page === page
    ? result.state
    : { status: "loading" };
}
