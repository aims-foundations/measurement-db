import { MotionReveal, MotionRevealItem } from "@/components/motion-reveal";
import type { ShelfStat } from "@/content/measurement-db";

// The headline stat row: three hairline-ruled columns, each a small mono index
// over an oversized light numeral. Used by the "data layer" band (cardinal) and
// by the judged shelf in section 04 (sage), so both read as the same object at
// two brightnesses rather than as two similar-looking components.
type Tone = "dark" | "light";

// Only the ink weights change between tones. On the cardinal band the rule and
// labels are white at reduced alpha; on a light band the same relationships are
// expressed in black, a touch lighter so the numerals stay dominant.
const TONE: Record<Tone, { rule: string; index: string; label: string }> = {
  dark: {
    rule: "border-white/40",
    index: "opacity-60",
    label: "opacity-85",
  },
  light: {
    rule: "border-black/25",
    index: "opacity-50",
    label: "opacity-70",
  },
};

export function StatRow({
  stats,
  tone = "dark",
  bare = false,
  className = "",
}: {
  stats: readonly ShelfStat[];
  /** Which band this sits on — picks the ink weights, not the layout. */
  tone?: Tone;
  /** Drop the column rule and the mono index, so the numerals lead. */
  bare?: boolean;
  className?: string;
}) {
  const ink = TONE[tone];

  return (
    <MotionReveal
      className={`grid gap-x-10 gap-y-12 sm:grid-cols-3 ${className}`.trim()}
      stagger={0.1}
    >
      {stats.map((stat, i) => (
        <MotionRevealItem key={stat.label}>
          <div className={bare ? "" : `border-t ${ink.rule} pt-6`}>
            {!bare && (
              <div className={`rd-mono ${ink.index}`}>
                {String(i + 1).padStart(2, "0")}
              </div>
            )}
            <p
              className={`${bare ? "" : "mt-5 "}font-light leading-none tracking-tight`}
              style={{ fontSize: "clamp(2.6rem, 5.4vw, 4.4rem)" }}
            >
              {stat.value}
            </p>
            <p
              className={`mt-4 text-sm font-light leading-relaxed ${ink.label}`}
            >
              {stat.label}
            </p>
          </div>
        </MotionRevealItem>
      ))}
    </MotionReveal>
  );
}
