import {
  testletLabel,
  type ColumnGroup,
} from "@/lib/benchmark-records/testlets";

export const TESTLET_HEADER_HEIGHT = 44;

/** A group label only: every matrix column still represents an individual item. */
export function TestletHeader({ group }: { group: ColumnGroup }) {
  const label = testletLabel(group);
  const detail =
    group.testletId === null
      ? "No recorded testlet"
      : `${group.length.toLocaleString()} of ${group.totalItems.toLocaleString()} items${group.continuedAfter ? " · continues →" : ""}`;
  return (
    <div
      data-testlet-id={group.testletId ?? undefined}
      title={`${label} · ${detail}`}
      className="flex min-w-0 flex-col justify-center overflow-hidden border-x border-t border-black/25 bg-[var(--fog-light)] px-2 text-center"
      style={{ height: TESTLET_HEADER_HEIGHT }}
    >
      <span className="truncate text-xs font-medium">{label}</span>
      <span className="truncate text-[0.6875rem] font-normal text-[var(--muted)]">
        {detail}
      </span>
    </div>
  );
}
