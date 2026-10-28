import type { MatrixView } from "@/lib/benchmark-records/view";
import type { CellSelection } from "@/lib/benchmark-records/model";
import type { ResponseScale, Subject } from "@/lib/benchmark-records/types";
import { HeatmapCanvas } from "./heatmap-canvas";
import { InlineMath } from "../inline-math";
import { ROW_HEIGHT } from "./paint-heatmap";
import { TESTLET_HEADER_HEIGHT } from "./testlet-header";
import { columnGroups } from "@/lib/benchmark-records/testlets";

export function Heatmap({
  view,
  subjects,
  scale,
  actual,
  theta,
  onSelect,
}: {
  view: MatrixView;
  subjects: Subject[];
  scale: ResponseScale;
  actual: boolean;
  theta?: Record<string, number>;
  onSelect: (selection: CellSelection) => void;
}) {
  const groups = columnGroups(view).filter((group) => group.testletId !== null);
  const grouped = groups.length > 0;
  const top = grouped ? TESTLET_HEADER_HEIGHT : 0;
  const columns = theta
    ? "grid-cols-[80px_minmax(100px,1fr)_32px_40px] sm:grid-cols-[minmax(100px,170px)_minmax(160px,1fr)_48px_56px]"
    : "grid-cols-[80px_minmax(100px,1fr)_40px] sm:grid-cols-[minmax(100px,170px)_minmax(160px,1fr)_56px]";
  if (!subjects.length || !view.itemIds.length)
    return (
      <p className="py-8 text-sm text-[var(--muted)]">
        No observations match these filters.
      </p>
    );
  return (
    <div className="overflow-auto rounded-md border border-[var(--line)] bg-white p-3">
      {grouped && (
        <details className="mb-3 text-xs text-[var(--muted)] sm:hidden">
          <summary className="cursor-pointer">Testlets on this page</summary>
          <ol className="mt-2 list-inside list-decimal space-y-2 break-all">
            {groups.map((group) => (
              <li key={group.testletId}>{group.testletId}</li>
            ))}
          </ol>
        </details>
      )}
      <div className={`grid min-w-[260px] gap-x-2 sm:gap-x-3 ${columns}`}>
        <div className="pb-2 text-xs text-[var(--muted)]">Model</div>
        <div className="pb-2 text-xs text-[var(--muted)]">
          {view.itemIds.length.toLocaleString()} items
        </div>
        {theta && (
          <div
            className="pb-2 text-right text-xs text-[var(--muted)]"
            title="Published fitted ability"
          >
            <InlineMath tex="\theta" />
          </div>
        )}
        <div
          className="pb-2 text-right text-xs text-[var(--muted)]"
          title="Mean of graded observations in this selection"
        >
          Mean
        </div>
        <div style={{ paddingTop: top }}>
          {subjects.map((s) => (
            <div
              key={s.subject_id}
              title={`${s.display_name} · ${s.subject_id}`}
              className="flex items-center truncate border-b border-[var(--line)] text-xs"
              style={{ height: ROW_HEIGHT }}
            >
              <span className="truncate">{s.display_name}</span>
            </div>
          ))}
        </div>
        <HeatmapCanvas
          view={view}
          subjects={subjects}
          scale={scale}
          actual={actual}
          onSelect={onSelect}
        />
        {theta && (
          <div style={{ paddingTop: top }}>
            {subjects.map((s) => (
              <div
                key={s.subject_id}
                className="flex items-center justify-end border-b border-[var(--line)] font-mono text-xs text-[var(--muted)]"
                style={{ height: ROW_HEIGHT }}
              >
                {theta[s.subject_id]?.toFixed(2) ?? "—"}
              </div>
            ))}
          </div>
        )}
        <div style={{ paddingTop: top }}>
          {subjects.map((s) => (
            <div
              key={s.subject_id}
              className="flex items-center justify-end border-b border-[var(--line)] font-mono text-xs text-[var(--muted)]"
              style={{ height: ROW_HEIGHT }}
            >
              {view.rowMeans.get(s.subject_id)?.toFixed(3) ?? "—"}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
