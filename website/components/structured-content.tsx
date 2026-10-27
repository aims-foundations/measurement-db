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
  if (depth >= 6 || (Array.isArray(value) && value.length > 100))
    return (
      <details className="text-xs">
        <summary className="cursor-pointer text-[var(--muted)]">
          {Array.isArray(value)
            ? `${value.length.toLocaleString()} entries`
            : "Nested fields"}
        </summary>
        <div className="mt-2 max-h-64 overflow-auto">
          <RichText text={JSON.stringify(value, null, 2)} />
        </div>
      </details>
    );
  if (Array.isArray(value))
    return value.length ? (
      <ol className="list-decimal space-y-3 pl-5">
        {value.map((entry, i) => (
          <li key={i} className="min-w-0 pl-1">
            <StructuredContent value={entry} Text={Text} depth={depth + 1} />
          </li>
        ))}
      </ol>
    ) : (
      <span className="text-sm text-[var(--muted)]">None</span>
    );

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
