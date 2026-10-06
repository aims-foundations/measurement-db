import type { Benchmark } from "@/content/measurement-db";

// Two-state "Saturation status" tag: red = saturated (top models ≥ 0.9),
// blue = not saturated. When saturation can't be asserted (null), we render
// no tag at all rather than an "unknown" one.
const STYLES = {
  yes: "border-[var(--digital-red)]/40 bg-[var(--digital-red)]/5 text-[var(--digital-red)]",
  no: "border-[var(--digital-blue)]/40 bg-[var(--digital-blue)]/5 text-[var(--digital-blue)]",
} as const;

const LABELS = { yes: "Yes", no: "No" } as const;

export function SaturationTag({
  saturation,
  variant = "card",
}: {
  saturation: Benchmark["saturation"];
  variant?: "card" | "detail";
}) {
  if (saturation === null) return null;
  const key = saturation ? "yes" : "no";
  const shape =
    variant === "detail"
      ? "rounded-full px-3 py-1 text-xs"
      : "rounded px-2 py-0.5 text-[0.6875rem]";
  return (
    <div
      className={`inline-flex items-baseline border font-medium ${shape} ${STYLES[key]}`}
    >
      Saturation status: {LABELS[key]}
    </div>
  );
}
