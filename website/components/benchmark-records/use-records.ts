"use client";

import { useEffect, useMemo, useState } from "react";
import { withBase } from "@/lib/base-path";
import { RecordSource } from "@/lib/benchmark-records/client";
import { prepareMatrix, type MatrixModel } from "@/lib/benchmark-records/model";
import type {
  AssetLink,
  GalleryEntry,
  TableName,
  TableRows,
} from "@/lib/benchmark-records/types";

export type LoadState<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; value: T };

const failure = (error: unknown): LoadState<never> => ({
  status: "error",
  message: error instanceof Error ? error.message : String(error),
});

/** Request lifecycle is separate from both record preparation and JSX. */
export function useGallery(entry: GalleryEntry) {
  const source = useMemo(
    () => new RecordSource(withBase(entry.base), entry.manifest),
    [entry],
  );
  const [result, setResult] = useState<{
    source: RecordSource;
    state: LoadState<MatrixModel>;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    source
      .load(controller.signal)
      .then(prepareMatrix)
      .then(
        (value) => {
          if (!controller.signal.aborted)
            setResult({ source, state: { status: "ready", value } });
        },
        (error) => {
          if (!controller.signal.aborted)
            setResult({ source, state: failure(error) });
        },
      );
    return () => controller.abort();
  }, [source]);
  return {
    source,
    state:
      result?.source === source
        ? result.state
        : ({ status: "loading" } as LoadState<MatrixModel>),
  };
}

export function useRecord<T extends TableName>(
  source: RecordSource,
  table: T,
  id: string | null,
): LoadState<TableRows[T] | null> {
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
  }, [source, table, id]);
  return result?.source === source && result.table === table && result.id === id
    ? result.state
    : { status: "loading" };
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
      ({ url, image }) => {
        if (controller.signal.aborted) return;
        imageUrl = image ? URL.createObjectURL(image) : null;
        setResult({
          source,
          link: stableLink,
          state: { status: "ready", value: { url, imageUrl } },
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
