"use client";
import { useState } from "react";
import { Pagination } from "./benchmark-records/pagination";
import { CodeBlock } from "./code-block";
import type { ComponentType } from "react";
import {
  fieldLabel,
  isRecord,
  structuredValue,
} from "@/lib/structured-content";
import { RichText } from "./benchmark-records/rich-text";

type TextRenderer = ComponentType<{ text: string }>;
const HEADING = "mb-2 text-xs font-medium text-[var(--muted)]";

/** Shared by items, answers, and metadata; preserves unfamiliar fields and their order. */
export function StructuredContent({
  value: source,
  Text = RichText,
  depth = 0,
}: {
  value: unknown;
  Text?: TextRenderer;
  depth?: number;
}) {
  const value = structuredValue(source);
  if (typeof value === "string") return <Text text={value} />;
  if (value == null || typeof value !== "object")
    return (
      <span className="text-sm">
        {value == null ? "Not provided" : String(value)}
      </span>
    );
  if (depth >= 6) return <NestedFields value={value} />;
  if (Array.isArray(value))
    return <ContentArray value={value} Text={Text} depth={depth} />;

  const { role, content, ...other } = value as Record<string, unknown>;
  if (typeof role === "string" && Object.hasOwn(value, "content"))
    return (
      <section className="min-w-0 rounded border border-[var(--line)] p-3">
        <h5 className={HEADING}>{fieldLabel(role)}</h5>
        <StructuredContent value={content} Text={Text} depth={depth + 1} />
        {Object.keys(other).length > 0 && (
          <div className="mt-3">
            <StructuredContent value={other} Text={Text} depth={depth + 1} />
          </div>
        )}
      </section>
    );
  if (!Object.keys(value).length)
    return <span className="text-sm text-[var(--muted)]">None</span>;
  return (
    <dl className="space-y-4">
      {Object.entries(value).map(([key, entry]) => (
        <div key={key} className="min-w-0 break-words">
          <dt className={HEADING}>{fieldLabel(key)}</dt>
          <dd
            className={
              isRecord(structuredValue(entry))
                ? "border-l border-[var(--line)] pl-3"
                : ""
            }
          >
            <StructuredContent value={entry} Text={Text} depth={depth + 1} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

function NestedFields({ value }: { value: object }) {
  const [open, setOpen] = useState(false);
  return (
    <details
      className="text-xs"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer text-[var(--muted)]">
        Nested fields
      </summary>
      {open && (
        <CodeBlock text={JSON.stringify(value, null, 2)} language="json" />
      )}
    </details>
  );
}

/** Long conversations remain readable messages, with a fixed amount of DOM work. */
function ContentArray({
  value,
  Text,
  depth,
}: {
  value: unknown[];
  Text: TextRenderer;
  depth: number;
}) {
  const [selected, setSelected] = useState(0);
  const size = 10;
  const page = Math.min(
    selected,
    Math.max(0, Math.ceil(value.length / size) - 1),
  );
  if (!value.length)
    return <span className="text-sm text-[var(--muted)]">None</span>;
  return (
    <div className="space-y-3">
      {value.length > size && (
        <Pagination
          label="entries"
          page={page}
          total={value.length}
          size={size}
          onChange={setSelected}
        />
      )}
      <ol start={page * size + 1} className="list-decimal space-y-3 pl-5">
        {value.slice(page * size, (page + 1) * size).map((entry, i) => (
          <li key={page * size + i} className="min-w-0 pl-1">
            <StructuredContent value={entry} Text={Text} depth={depth + 1} />
          </li>
        ))}
      </ol>
    </div>
  );
}
