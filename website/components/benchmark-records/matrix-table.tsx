import type { CellSelection, MatrixModel } from "@/lib/benchmark-records/model";
import { gradeColor } from "@/lib/benchmark-records/colors";
import type { ResponseScale, Subject } from "@/lib/benchmark-records/types";
import { columnGroups } from "@/lib/benchmark-records/testlets";
import { TestletHeader, TESTLET_HEADER_HEIGHT } from "./testlet-header";

/** Render the selected axes; observations have already been indexed by ID. */
export function MatrixTable({
  model,
  subjects,
  items,
  itemOffset,
  onSelect,
  scale,
}: {
  model: MatrixModel;
  scale: ResponseScale;
  subjects: Subject[];
  items: string[];
  itemOffset: number;
  onSelect: (selection: CellSelection) => void;
}) {
  const groups = columnGroups(model, items);
  const grouped = groups.some((group) => group.testletId !== null);
  const starts = new Set(groups.slice(1).map((group) => group.start));
  const ends = new Set(groups.map((group) => group.start + group.length - 1));
  const edges = (i: number) =>
    grouped && starts.has(i) ? "border-l-8 border-l-[var(--paper)]" : "";
  const outline = (i: number) => ({
    boxShadow: grouped
      ? [
          (i === 0 || starts.has(i)) && "inset 1px 0 #9c9a94",
          ends.has(i) && "inset -1px 0 #9c9a94",
        ]
          .filter(Boolean)
          .join(", ")
      : undefined,
  });
  return (
    <div
      className="max-h-[38rem] overflow-auto rounded-lg border border-[var(--line)]"
      tabIndex={0}
      aria-label="Model by item matrix"
    >
      <table className="w-full border-separate border-spacing-0 text-xs">
        <thead>
          {grouped && (
            <tr>
              <th
                rowSpan={2}
                scope="col"
                className="sticky top-0 z-20 min-w-48 border-b border-[var(--line)] bg-[var(--paper)] p-3 text-left"
              >
                Model / item
              </th>
              {groups.map((group) => (
                <th
                  key={group.start}
                  colSpan={group.length}
                  scope="colgroup"
                  className={`sticky top-0 z-20 bg-[var(--paper)] p-0 text-left font-normal ${edges(group.start)}`}
                >
                  <TestletHeader group={group} />
                </th>
              ))}
            </tr>
          )}
          <tr>
            {!grouped && (
              <th
                scope="col"
                className="sticky top-0 z-20 min-w-48 border-b border-[var(--line)] bg-[var(--paper)] p-3 text-left"
              >
                Model / item
              </th>
            )}
            {items.map((id, i) => (
              <th
                key={id}
                scope="col"
                className={`sticky z-10 min-w-24 border-b border-[var(--line)] bg-[var(--paper)] p-2 ${edges(i)}`}
                style={{
                  top: grouped ? TESTLET_HEADER_HEIGHT : 0,
                  ...outline(i),
                }}
              >
                <button
                  className="underline decoration-[var(--line)] underline-offset-4"
                  title={id}
                  onClick={() => onSelect({ subject: null, itemId: id })}
                >
                  Item {itemOffset + i + 1}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {subjects.map((subject) => (
            <tr key={subject.subject_id}>
              <th
                scope="row"
                className="sticky left-0 z-10 min-w-48 max-w-64 border-b border-[var(--line)] bg-white p-3 text-left"
              >
                {subject.display_name}
                <span className="mt-1 block break-all font-mono text-[0.6rem] font-normal text-[var(--muted)]">
                  {subject.subject_id}
                </span>
              </th>
              {items.map((id, i) => {
                const attempts =
                  model.cells.get(subject.subject_id)?.get(id) ?? [];
                const label =
                  attempts.length > 1
                    ? `${attempts.length} observations`
                    : String(attempts[0]?.response ?? "?");
                const ungraded = attempts.every(
                  (attempt) => attempt.response === null,
                );
                return (
                  <td
                    key={id}
                    className={`border-b border-[var(--line)] p-1 text-center ${edges(i)}`}
                    style={outline(i)}
                  >
                    {attempts.length ? (
                      <button
                        className={`w-full whitespace-nowrap rounded px-3 py-2 hover:ring-1 hover:ring-[var(--palo-alto)] ${ungraded ? "bg-black/5 text-[var(--muted)]" : "bg-[var(--palo-alto)]/10"}`}
                        aria-label={`${subject.display_name}, item ${itemOffset + i + 1}: ${label}`}
                        onClick={() => onSelect({ subject, itemId: id })}
                      >
                        {attempts.length === 1 && (
                          <span
                            aria-hidden="true"
                            className="mr-2 inline-block h-2.5 w-2.5 rounded-sm border border-black/15"
                            style={{
                              background: gradeColor(
                                attempts[0].response,
                                scale,
                              ),
                            }}
                          />
                        )}
                        {label}
                      </button>
                    ) : (
                      <span
                        title={
                          model.partialItemIds?.includes(id)
                            ? "No observation on this page"
                            : "No recorded observation"
                        }
                      >
                        —
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
