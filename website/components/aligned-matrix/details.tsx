"use client";

import { groupAt, type ScoreMatrix } from "@/lib/aligned-matrix/model";
import type { SelectedCell } from "@/lib/aligned-matrix/types";
import type { TestSource } from "@/lib/aligned-matrix/client";
import type { RecordSource } from "@/lib/benchmark-records/client";
import { presentationFor } from "@/lib/benchmark-records/presentation";
import type { Observation, ResponseScale } from "@/lib/benchmark-records/types";
import { CellDetails } from "../benchmark-records/cell-details";
import { CONTROL } from "../benchmark-records/view-controls";
import { useSelection } from "./use-selection";

const NO_PAIRS = new Map<string, Observation>();

export function TestDetails({
  matrix,
  tests,
  source,
  selection,
  onSelect,
  onClose,
}: {
  matrix: ScoreMatrix;
  tests: TestSource;
  source: RecordSource;
  selection: SelectedCell;
  onSelect: (cell: SelectedCell) => void;
  onClose: () => void;
}) {
  const { row, column } = selection;
  const group = groupAt(matrix.manifest.groups, column);
  const setup = group.rowSetups[row];
  const subject = matrix.manifest.subjects[row];
  const { state, retry } = useSelection(matrix, tests, source, row, column);
  if (state.status === "loading")
    return (
      <p role="status" className="p-5 text-sm">
        Loading the selected test and its source observation…
      </p>
    );
  if (state.status === "error")
    return (
      <div role="alert" className="p-5 text-sm">
        {state.message}{" "}
        <button className={CONTROL} onClick={retry}>
          Retry details
        </button>
      </div>
    );
  const { test, itemId, model } = state.value;
  const available = test.items.flatMap((id, index) => (id ? [index] : []));
  return (
    <section
      aria-label="Selected test"
      className="mt-6 border-t border-[var(--line)] pt-5"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-wide text-[var(--muted)]">
            Test {column + 1} · {group.id}
          </p>
          <h3 className="mt-2 break-all font-mono text-sm">{test.name}</h3>
          {group.setups[setup] && (
            <p className="mt-2 text-xs text-[var(--muted)]">
              Selected setup: grader {group.setups[setup].grader} · prompt{" "}
              <code>{group.setups[setup].promptHash.slice(0, 8)}</code>
            </p>
          )}
        </div>
        <button className={CONTROL} onClick={onClose}>
          Close test
        </button>
      </div>
      <details className="mt-4 rounded border border-[var(--line)] bg-white p-3">
        <summary className="cursor-pointer text-sm">
          Expand variants ({available.length} recorded items)
        </summary>
        <p className="my-3 text-xs text-[var(--muted)]">
          These items share the full upstream test identifier. Their prompts,
          environments, or graders differ; every result retains its original
          setup. Prompt hashes identify exact message text.
        </p>
        <div className="overflow-x-auto">
          <table
            className="w-full text-left text-xs"
            aria-label="Recorded setup variants"
          >
            <thead>
              <tr>
                {["Grader", "Prompt", "Environment", "Source item", ""].map(
                  (label, i) => (
                    <th
                      className="border-b border-[var(--line)] p-2 font-medium"
                      key={i}
                    >
                      {label}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {available.map((index) => {
                const variant = group.setups[index];
                const modelRow = group.rowSetups.findIndex(
                  (value, r) =>
                    value === index && matrix.value(r, column) !== 0,
                );
                return (
                  <tr
                    key={test.items[index]}
                    className={index === setup ? "bg-[var(--fog-light)]" : ""}
                  >
                    <td className="p-2">{variant.grader}</td>
                    <td className="p-2 font-mono" title={variant.promptHash}>
                      {variant.promptHash.slice(0, 8)}
                    </td>
                    <td className="max-w-72 break-all p-2">
                      {variant.image}
                      <br />
                      <span className="text-[var(--muted)]">{variant.cwd}</span>
                    </td>
                    <td className="p-2 font-mono">{test.items[index]}</td>
                    <td className="p-2">
                      <button
                        className={CONTROL}
                        disabled={modelRow < 0}
                        onClick={() => onSelect({ row: modelRow, column })}
                      >
                        Inspect variant {index + 1}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
      {model && itemId ? (
        <CellDetails
          key={`${subject.subject_id}/${itemId}`}
          source={source}
          selection={{ subject, itemId }}
          observations={model.observations}
          scale={
            JSON.parse(
              source.manifest.benchmark.response_scale,
            ) as ResponseScale
          }
          presentation={presentationFor(source.manifest.benchmark.benchmark_id)}
          model={model}
          pairs={NO_PAIRS}
          onClose={onClose}
        />
      ) : (
        <p className="mt-4 text-sm">
          No recorded outcome for {subject.display_name} on this test. Expand
          variants to inspect an available observation.
        </p>
      )}
    </section>
  );
}
