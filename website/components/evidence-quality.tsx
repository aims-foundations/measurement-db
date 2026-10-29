import type {
  BandQuality,
  QualityEntry,
  QuestionQuality,
} from "@/content/annotations";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  evidenceRubric,
  evidenceRubricText,
  levelName,
} from "@/content/annotations";

/* ============================================================
   EvidenceQuality — the rubric, then one pulldown per band,
   opening to one pulldown per question, opening to every coded
   score.

   The rubric forbids combining the four dimensions: "Record each
   score separately. Do not add them up or assign a band." So a
   mean is only ever taken WITHIN a dimension, and the four are
   always shown side by side rather than rolled into one figure.

   Native <details> throughout, so none of it needs client JS.
   ============================================================ */

const DIMS = ["v", "s", "p", "c"] as const;

/** Group a question's ratings by the code they describe, so each code opens to
 *  the annotators who rated it. Codes are matched case-insensitively, since
 *  coders typed both "items/tasks" and "Items/tasks"; the first spelling seen
 *  is kept. Order follows how many annotators rated the code. */
function groupByCode(
  entries: readonly QualityEntry[],
): [string, QualityEntry[]][] {
  const groups = new Map<string, { label: string; rows: QualityEntry[] }>();
  for (const entry of entries) {
    const key = entry.code.trim().toLowerCase();
    const group = groups.get(key);
    if (group) group.rows.push(entry);
    else groups.set(key, { label: entry.code.trim(), rows: [entry] });
  }
  return [...groups.values()]
    .sort((a, b) => b.rows.length - a.rows.length)
    .map((g) => [g.label, g.rows] as [string, QualityEntry[]]);
}

/** One annotator's row: who, the four numbers, and what they cited for them.
 *  Numbers only — the level names live in the rubric pulldown, and repeating
 *  them inline buried the scores. */
function ScoreRow({ entry }: { entry: QualityEntry }) {
  return (
    <li className="grid gap-x-4 gap-y-2 border-t border-black/10 px-3 py-2.5 sm:grid-cols-[6rem_auto_minmax(0,1fr)]">
      <span className="rd-mono !text-[0.58rem] uppercase tracking-wider opacity-55">
        {entry.by}
      </span>
      <span className="flex h-fit flex-wrap gap-x-3 gap-y-1">
        {DIMS.map((key) => (
          <span
            key={key}
            className="rd-mono !text-[0.66rem] tabular-nums"
            title={`${evidenceRubric.find((d) => d.key === key)?.name}: ${entry[key]} ${levelName(key, entry[key])}`}
          >
            <span className="opacity-50">{key.toUpperCase()}</span>
            {entry[key]}
          </span>
        ))}
      </span>
      <span className="min-w-0">
        {entry.rationale && entry.rationale.length > 0 ? (
          <span className="flex flex-col gap-2.5">
            {entry.rationale.map((item, i) => (
              <span key={i} className="flex flex-col gap-1">
                {item.quotes.map((quote, qi) => (
                  <span
                    key={`${qi}-${quote}`}
                    className="border-l-2 border-black/20 pl-2.5 text-sm font-light leading-snug opacity-80"
                  >
                    &ldquo;{quote}&rdquo;
                  </span>
                ))}
                {item.note ? (
                  <span className="text-sm font-light leading-snug opacity-60">
                    {item.note}
                  </span>
                ) : null}
                {item.sources.length > 0 ? (
                  <span className="flex flex-wrap gap-x-3 gap-y-1">
                    {item.sources.map((source, si) => (
                      <a
                        key={`${si}-${source}`}
                        href={source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="rd-mono !text-[0.58rem] max-w-full truncate underline decoration-black/25 underline-offset-2 opacity-60 hover:opacity-100"
                      >
                        {source.replace(/^https?:\/\/(www\.)?/, "")}
                      </a>
                    ))}
                  </span>
                ) : null}
              </span>
            ))}
          </span>
        ) : (
          /* NOT a gap in the annotation: every rated code was written up in
             the form. These are rows where this pipeline failed to attach the
             note to its code, because the annotator's note header names the
             code differently from the option text. Flagged loudly so the
             matcher gets fixed rather than the data being blamed. */
          <span className="rd-mono !text-[0.6rem] uppercase tracking-wider text-[var(--rd-cardinal)] opacity-70">
            Code transfer error
          </span>
        )}
      </span>
    </li>
  );
}

/** One code, opening to each annotator's scores for it. */
function CodeBlock({
  code,
  entries,
}: {
  code: string;
  entries: readonly QualityEntry[];
}) {
  return (
    <details className="group border-t border-black/10">
      <summary className="flex cursor-pointer list-none items-start gap-2.5 py-2.5 [&::-webkit-details-marker]:hidden">
        <svg
          width="10"
          height="10"
          viewBox="0 0 12 12"
          fill="none"
          aria-hidden
          className="mt-[0.4rem] shrink-0 opacity-40 transition-transform duration-200 group-open:rotate-180"
        >
          <path
            d="M2 4.5 6 8.5l4-4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="min-w-0 flex-1 text-sm font-light">{code}</span>
        <span className="rd-mono !text-[0.58rem] shrink-0 opacity-45">
          {entries.length}
        </span>
      </summary>
      {/* The code itself sits on the card; each annotator's reading is tinted,
          so one person's scores and evidence read as a single block. */}
      <ul className="mb-2 ml-[1.3rem] flex flex-col bg-black/[0.045]">
        {entries.map((entry) => (
          <ScoreRow key={`${entry.by}-${code}`} entry={entry} />
        ))}
      </ul>
    </details>
  );
}

function QuestionBlock({ question }: { question: QuestionQuality }) {
  const n = question.entries.length;

  return (
    <details className="group border-t border-black/10">
      <summary className="flex cursor-pointer list-none items-start gap-3 py-3.5 [&::-webkit-details-marker]:hidden">
        <svg
          width="11"
          height="11"
          viewBox="0 0 12 12"
          fill="none"
          aria-hidden
          className="mt-[0.45rem] shrink-0 opacity-45 transition-transform duration-200 group-open:rotate-180"
        >
          <path
            d="M2 4.5 6 8.5l4-4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium leading-snug">
            {question.question}
          </span>
          <span className="mt-1.5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="rd-mono !text-[0.6rem] opacity-45">
              {n > 0 ? `${n} rated` : "no rated codes"}
            </span>
          </span>
        </span>
      </summary>

      <div className="pb-5 pl-[1.4rem]">
        {question.entries.length > 0 ? (
          <div className="flex flex-col">
            {groupByCode(question.entries).map(([code, entries]) => (
              <CodeBlock key={code} code={code} entries={entries} />
            ))}
          </div>
        ) : (
          <p className="text-sm font-light italic opacity-50">
            No codes rated. The rubric exempts N/A, &ldquo;none
            documented&rdquo; and &ldquo;not reported&rdquo;.
          </p>
        )}

      </div>
    </details>
  );
}

export function RubricPulldown() {
  return (
    <details className="group border border-black/15 bg-white/55">
      <summary className="flex cursor-pointer list-none items-center gap-3 p-4 [&::-webkit-details-marker]:hidden sm:p-5">
        <svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          aria-hidden
          className="shrink-0 opacity-45 transition-transform duration-200 group-open:rotate-180"
        >
          <path
            d="M2 4.5 6 8.5l4-4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="rd-h3">Evidence Quality Rubric (v1)</span>
      </summary>
      {/* The rubric verbatim. Tables are wide, so the whole document scrolls
          horizontally on small screens rather than being reflowed. */}
      <div className="rubric-md overflow-x-auto px-4 pb-5 sm:px-5">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {evidenceRubricText}
        </ReactMarkdown>
      </div>
    </details>
  );
}

export function EvidenceQuality({ bands }: { bands: readonly BandQuality[] }) {
  if (bands.length === 0) return null;

  // Computed from the entries rather than from the band means, so the overall
  // figure is not an average of rounded averages.
  const all = bands.flatMap((band) =>
    band.questions.flatMap((question) => question.entries),
  );
  const overall = all.length
    ? (Object.fromEntries(
        DIMS.map((key) => [
          key,
          all.reduce((sum, entry) => sum + entry[key], 0) / all.length,
        ]),
      ) as { v: number; s: number; p: number; c: number })
    : null;

  return (
    <div className="mt-6 flex flex-col gap-4">
      {overall ? (
        <div className="border border-black/15 bg-white/55 p-4 sm:p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
            <span className="rd-h3">Summary:</span>
            <span className="rd-mono !text-[0.6rem] opacity-45">
              {all.length} rated
            </span>
          </div>
          <p className="mt-4 border-t border-black/10 pt-3.5 text-sm font-light leading-relaxed opacity-45">
            <span className="italic">Summary:</span> [fill out later]
          </p>
        </div>
      ) : null}

      <div className="border border-black/15 bg-white/55">
        {bands.map((band) => (
          <details key={band.id} className="group border-b border-black/10 last:border-b-0">
            <summary className="flex cursor-pointer list-none items-start gap-3 p-4 [&::-webkit-details-marker]:hidden sm:p-5">
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                aria-hidden
                className="mt-[0.3rem] shrink-0 opacity-45 transition-transform duration-200 group-open:rotate-180"
              >
                <path
                  d="M2 4.5 6 8.5l4-4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <span className="min-w-0 flex-1">
                <span className="rd-h3 block">{band.label}</span>
                <span className="mt-1.5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
                  <span className="rd-mono !text-[0.6rem] opacity-45">
                    {band.rated} rated
                  </span>
                </span>
              </span>
            </summary>
            <div className="px-4 pb-4 sm:px-5 sm:pb-5">
              {band.questions.map((question) => (
                <QuestionBlock key={question.leaf} question={question} />
              ))}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}
