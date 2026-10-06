/*
 * AIMS logomark in the Stanford HAI style: letterforms drawn as three
 * parallel inline strokes, next to a two-line unit name. Inherits
 * currentColor so it works on light and dark bands.
 */

function AimsMark({ height }: { height: number }) {
  const width = (height / 70) * 177;
  return (
    <svg
      viewBox="0 0 177 70"
      width={width}
      height={height}
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      aria-hidden="true"
    >
      {/* A — nested chevrons + crossbar */}
      <path d="M 0 70 L 20 1 L 40 70" />
      <path d="M 4.7 70 L 20 17.2 L 35.3 70" />
      <path d="M 9.4 70 L 20 33.4 L 30.6 70" />
      <path d="M 14.2 53.5 H 25.8" />
      <path d="M 12.9 58 H 27.1" />

      {/* I — three verticals */}
      <path d="M 53 1 V 70" />
      <path d="M 57.5 1 V 70" />
      <path d="M 62 1 V 70" />

      {/* M — triple stems + nested center V */}
      <path d="M 75 1 V 70" />
      <path d="M 79.5 1 V 70" />
      <path d="M 84 1 V 70" />
      <path d="M 110 1 V 70" />
      <path d="M 114.5 1 V 70" />
      <path d="M 119 1 V 70" />
      <path d="M 75 1 L 97 48 L 119 1" />
      <path d="M 79.5 1 L 97 41 L 114.5 1" />
      <path d="M 84 1 L 97 34 L 110 1" />

      {/* S — two loops of concentric arcs */}
      <g transform="translate(132 0)">
        <path d="M 38.5 17 A 16 16 0 1 0 22.5 33 A 13.5 13.5 0 1 1 9 46.5" />
        <path d="M 34 17 A 11.5 11.5 0 1 0 22.5 28.5 A 18 18 0 1 1 4.5 46.5" />
        <path d="M 29.5 17 A 7 7 0 1 0 22.5 24 A 22.5 22.5 0 1 1 0 46.5" />
      </g>
    </svg>
  );
}

export function RdLogo({
  height = 34,
  compactBelowSm = false,
}: {
  height?: number;
  compactBelowSm?: boolean;
}) {
  const textSize = height * 0.34;
  return (
    <span className="inline-flex items-center" style={{ gap: height * 0.32 }}>
      <AimsMark height={height} />
      <span
        className={`${compactBelowSm ? "hidden sm:flex" : "flex"} flex-col justify-center`}
        style={{ fontSize: textSize, lineHeight: 1.3 }}
      >
        <span className="font-semibold">Stanford University</span>
        <span className="font-normal">AI Measurement Science</span>
      </span>
      <span className="sr-only">AIMS — AI Measurement Science</span>
    </span>
  );
}
