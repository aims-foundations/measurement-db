import type { BenchmarkIrt } from "@/content/benchmark-irt";

/* ============================================================
   IrtPanel — the Rasch fit summary that sits above the response
   matrix: what was fitted, and how well it predicts a held-out
   cell.

   The AUC pair is the point. Train AUC says how much of the
   matrix a two-parameter model can absorb at all; test AUC says
   how much of that survives on cells it never saw. A wide gap
   means the fit is memorising subjects and items rather than
   ordering them.
   ============================================================ */

function fmt(v: number | null): string {
  return v == null ? "n/a" : v.toFixed(3);
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-right">
      <div className="text-[0.625rem] uppercase tracking-wide text-[var(--muted)]">
        {label}
      </div>
      <div className="text-sm font-medium tabular-nums text-[var(--ink)]">
        {value}
      </div>
    </div>
  );
}

export function IrtPanel({ irt }: { irt: BenchmarkIrt }) {
  const n = (irt.nTrain + irt.nTest).toLocaleString("en-US");
  return (
    <div className="mb-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-3 rounded-md border border-[var(--line)] bg-[var(--fog-light)] px-3 py-2.5">
      <div className="min-w-0">
        <div className="text-xs font-medium text-[var(--ink)]">
          Rasch analysis{" "}
          <span className="font-normal text-[var(--muted)]">
            {/* The fitted link, spelled the way the parameters are named in the
                gutters (θ) and in subjects.csv. */}
            p = σ(θ − z{irt.nConditions > 1 ? " + c" : ""})
          </span>
        </div>
        <p className="mt-0.5 text-[0.6875rem] leading-snug text-[var(--muted)]">
          {n} responses, 80/20 split over cells · {irt.nSubjects} subjects ·{" "}
          {irt.nItems.toLocaleString("en-US")} items
          {irt.nConditions > 1 ? ` · ${irt.nConditions} conditions` : ""}
          {irt.converged ? "" : " · did not converge"}
        </p>
        {/* A perfectly-separated subject has no finite ability, so its θ is a
            saturation artefact rather than a measurement. Saying how many there
            are is the difference between an odd number and a misleading one. */}
        {irt.separatedSubjects ? (
          <p className="mt-0.5 text-[0.6875rem] leading-snug text-[var(--muted)]">
            {irt.separatedSubjects} of {irt.nSubjects} subjects answered every
            item alike, so {irt.separatedSubjects === 1 ? "its" : "their"} θ is
            unbounded and sits pegged to the column edge.
          </p>
        ) : null}
      </div>
      <div className="flex shrink-0 items-start gap-5">
        <Stat label="AUC train" value={fmt(irt.aucTrain)} />
        <Stat label="AUC test" value={fmt(irt.aucTest)} />
      </div>
    </div>
  );
}
