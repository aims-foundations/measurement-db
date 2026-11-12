import {
  bounds,
  gradeColor,
  EMPTY,
  UNGRADED,
  MIXED,
} from "@/lib/benchmark-records/colors";
import { describeScale, gradeLabel } from "@/lib/benchmark-records/scale";
import type { ResponseScale } from "@/lib/benchmark-records/types";

export function Legend({ scale }: { scale: ResponseScale }) {
  const entries =
    scale.kind === "discrete"
      ? scale.values.map((value) => ({
          color: gradeColor(value, scale),
          label: gradeLabel(value, scale),
        }))
      : scale.kind === "mixed"
        ? [{ color: MIXED, label: "Graded · item-specific scale" }]
        : bounds(scale)
            .filter((v): v is number => v !== null)
            .map((value) => ({
              color: gradeColor(value, scale),
              label: String(value),
            }));
  return (
    <div
      className="mt-4 space-y-2 text-xs text-[var(--muted)]"
      aria-label="Grade legend"
    >
      <div className="flex flex-wrap gap-x-5 gap-y-2">
        {[
          ...entries,
          { color: UNGRADED, label: "Grade unavailable" },
          { color: EMPTY, label: "No recorded observation" },
        ].map(({ color, label }) => (
          <span
            key={label}
            className="inline-flex min-w-0 items-center gap-1.5"
          >
            <span
              className="h-3 w-3 shrink-0 rounded-sm border border-black/15"
              style={{ background: color }}
            />
            <span className="min-w-0 break-words">{label}</span>
          </span>
        ))}
      </div>
      <p>
        {scale.kind === "discrete"
          ? scale.direction && `${scale.direction.replaceAll("_", " ")}. `
          : `${describeScale(scale)}. `}
        Stacked colors show separate attempts.
      </p>
    </div>
  );
}
