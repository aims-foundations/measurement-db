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
  const grouped = columnGroups(view).some((group) => group.testletId !== null);
  const top = grouped ? TESTLET_HEADER_HEIGHT : 0;
  if (!subjects.length || !view.itemIds.length)
    return (
      <p className="py-8 text-sm text-[var(--muted)]">
        No observations match these filters.
      </p>
    );
  return (
    <div className="overflow-auto rounded-md border border-[var(--line)] bg-white p-3">
      <div
        className="grid min-w-[440px] gap-x-3"
        style={{
          gridTemplateColumns: `minmax(100px,170px) minmax(160px,1fr) ${theta ? "48px " : ""}56px`,
        }}
      >
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
