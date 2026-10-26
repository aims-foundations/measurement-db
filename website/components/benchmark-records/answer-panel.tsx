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
  presentation,
}: {
  source: RecordSource;
  observation: Observation;
  name: string;
  presentation: Presentation;
}) {
  const answer = useRecord(source, "traces", observation.response_id);
  return (
    <section className="min-w-0" aria-label={`Answer from ${name}`}>
      <h5 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--muted)]">
        {name}
      </h5>
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
            className="max-h-[32rem] overflow-auto rounded bg-white p-4"
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
