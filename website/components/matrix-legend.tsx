import type { BenchmarkDetail } from "@/content/benchmark-details";

export function MatrixLegend({ detail }: { detail: BenchmarkDetail }) {
  return (
    <figcaption className="mt-4 space-y-2 text-xs font-light opacity-70">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        {(detail.stats.graded ??
          (detail.stats.meanResponse === null ? 0 : 1)) ===
        0 ? null : detail.categories && detail.categories.length ? (
          detail.categories.map((c) => (
            <span key={c.value} className="inline-flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ backgroundColor: c.color }}
              />
              {c.label}
            </span>
          ))
        ) : detail.isBinary ? (
          // Binary matrices are recoloured at display time (see
          // MatrixViewer's COLOR_SWAP filter): the "1" cells show
          // blue, the "0" cells show red. The swatches match.
          <>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-sm bg-[#1f77b4]" />
              {detail.binaryLabels ? `${detail.binaryLabels.one} (1)` : "1"}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-sm bg-[#d62728]" />
              {detail.binaryLabels ? `${detail.binaryLabels.zero} (0)` : "0"}
            </span>
          </>
        ) : detail.responseType === "mixed" ? (
          <span>Each task uses its declared grade scale.</span>
        ) : (
          <span className="inline-flex items-center gap-2">
            <span>low</span>
            <span
              aria-hidden="true"
              className="inline-block h-3 w-24 rounded-sm"
              style={{
                backgroundImage:
                  "linear-gradient(to right, #1f77b4, #f2f2f2, #d62728)",
              }}
            />
            <span>high</span>
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm bg-[#9e9e9e]" />
          Grade unavailable
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm border border-[var(--line)] bg-white" />
          No recorded observation
        </span>
      </div>
      <p>
        <span className="rd-mono !text-[0.6rem] opacity-60">Scale:</span>{" "}
        <span className="font-mono">{detail.scaleLabel}</span>
      </p>
    </figcaption>
  );
}
