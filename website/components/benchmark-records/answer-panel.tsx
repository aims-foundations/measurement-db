import type { RecordSource } from "@/lib/benchmark-records/client";
import type { Observation } from "@/lib/benchmark-records/types";
import {
  displayText,
  type Presentation,
} from "@/lib/benchmark-records/presentation";
import { useRecord } from "./use-records";
import { RichText, SourceText } from "./rich-text";

export function AnswerPanel({
  source,
  observation,
  name,
  showName = false,
  presentation,
}: {
  source: RecordSource;
  observation: Observation;
  name: string;
  showName?: boolean;
  presentation: Presentation;
}) {
  const answer = useRecord(source, "traces", observation.response_id);
  return (
    <section
      className="min-w-0 rounded-md border border-[var(--line)] bg-white p-4 sm:p-5"
      aria-label={`Answer from ${name}`}
    >
      <header className="mb-4 border-b border-[var(--line)] pb-3">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
          Model Answer
        </h4>
        {showName && (
          <p className="mt-2 break-words text-sm font-medium">{name}</p>
        )}
      </header>
      {answer.status === "loading" ? (
        <p role="status" className="text-sm">
          Loading answer…
        </p>
      ) : answer.status === "error" ? (
        <p role="alert" className="text-sm">
          {answer.message}
        </p>
      ) : (
        <>
          <div
            className="max-h-[32rem] overflow-auto"
            aria-label="Model answer"
          >
            <RichText
              text={
                displayText(
                  answer.value?.trace ?? null,
                  presentation.answerPath,
                ) ?? "No answer text released."
              }
            />
          </div>
          {answer.value?.trace && (
            <SourceText text={answer.value.trace} label="Answer" />
          )}
        </>
      )}
    </section>
  );
}
