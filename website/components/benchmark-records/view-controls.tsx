import { useState } from "react";
import type { Dimension, Filters } from "@/lib/benchmark-records/context";
import type { Order } from "@/lib/benchmark-records/view";

export const CONTROL =
  "max-w-full rounded border border-[var(--line)] bg-white px-2 py-1.5 text-xs text-[var(--ink)] disabled:opacity-40";
export type ViewSettings = {
  filters: Filters;
  query: string;
  order: Order;
  display: "heatmap" | "table";
  actual: boolean;
};

function Filter({
  dimension,
  value,
  onChange,
}: {
  dimension: Dimension;
  value: string;
  onChange: (v: string) => void;
}) {
  const options = (
    <>
      <option value="">All</option>
      {dimension.values.map((value) => (
        <option key={JSON.stringify(value)} value={JSON.stringify(value)}>
          {value ?? "Not recorded"}
        </option>
      ))}
    </>
  );
  return (
    <label className="flex min-w-0 items-center gap-2 text-xs text-[var(--muted)]">
      <span className="shrink-0 capitalize">{dimension.label}</span>
      <select
        aria-label={`Filter ${dimension.label}`}
        className={`${CONTROL} min-w-0`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options}
      </select>
    </label>
  );
}

export function ViewControls({
  dimensions,
  settings,
  canOrder,
  onChange,
  paged = false,
}: {
  dimensions: Dimension[];
  settings: ViewSettings;
  canOrder: boolean;
  onChange: (value: ViewSettings) => void;
  paged?: boolean;
}) {
  const [advanced, setAdvanced] = useState(false);
  const update = (patch: Partial<ViewSettings>) =>
    onChange({ ...settings, ...patch });
  const varied = dimensions.filter((d) => d.values.length > 1);
  const common = varied.filter((d) => d.values.length <= 30);
  const extra = varied.filter((d) => d.values.length > 30);
  return (
    <div className="mb-4 space-y-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
        <label className="flex min-w-0 items-center gap-2 text-xs text-[var(--muted)]">
          Find a model
          <input
            type="search"
            placeholder="Name or subject ID"
            className={`${CONTROL} min-w-0`}
            value={settings.query}
            onChange={(e) => update({ query: e.target.value })}
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
          {paged ? "Order on this page" : "Order"}
          <select
            className={CONTROL}
            value={settings.order}
            onChange={(e) => update({ order: e.target.value as Order })}
          >
            <option value="recorded">Recorded order</option>
            {canOrder && <option value="grade">Mean grade</option>}
            <option value="name">Name / ID</option>
          </select>
        </label>
        <div className="flex gap-1" aria-label="View">
          {(["heatmap", "table"] as const).map((display) => (
            <button
              key={display}
              className={`${CONTROL} ${settings.display === display ? "!bg-[var(--fog-light)] font-medium" : ""}`}
              aria-pressed={settings.display === display}
              onClick={() => update({ display })}
            >
              {display === "heatmap" ? "Heatmap" : "Table"}
            </button>
          ))}
        </div>
        {settings.display === "heatmap" && (
          <button
            className={CONTROL}
            aria-pressed={settings.actual}
            onClick={() => update({ actual: !settings.actual })}
          >
            {settings.actual ? "Fit to width" : "Actual size"}
          </button>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {common.map((d) => (
          <Filter
            key={d.key}
            dimension={d}
            value={settings.filters[d.key] ?? ""}
            onChange={(value) =>
              update({ filters: { ...settings.filters, [d.key]: value } })
            }
          />
        ))}
        {extra.length > 0 && (
          <button
            className={`${CONTROL} !border-transparent underline`}
            aria-expanded={advanced}
            onClick={() => setAdvanced(!advanced)}
          >
            More conditions
          </button>
        )}
        {Object.values(settings.filters).some(Boolean) && (
          <button
            className={`${CONTROL} !border-transparent underline`}
            onClick={() => update({ filters: {} })}
          >
            Clear filters
          </button>
        )}
      </div>
      {advanced && (
        <div className="flex flex-wrap gap-3">
          {extra.map((d) => (
            <Filter
              key={d.key}
              dimension={d}
              value={settings.filters[d.key] ?? ""}
              onChange={(value) =>
                update({ filters: { ...settings.filters, [d.key]: value } })
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
