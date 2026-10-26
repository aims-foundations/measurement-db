import { contextValues, parseFields } from "./context";
import type { Presentation } from "./presentation";
import type { MatrixModel } from "./model";
import type { Observation } from "./types";

/** Pair only unique reciprocal observations with an explicit shared context. */
export function prepareComparisons(
  model: MatrixModel,
  presentation: Presentation,
) {
  const aliases = new Map<string, string[]>();
  for (const subject of model.subjects) {
    const names = new Set([subject.subject_id, subject.display_name]);
    if (presentation.pairing) {
      const alias = parseFields(subject.subject_features_extra)[
        presentation.pairing.subjectFeature
      ];
      if (alias) names.add(alias);
    }
    for (const name of names)
      aliases.set(name, [...(aliases.get(name) ?? []), subject.subject_id]);
  }
  const identity = (record: Observation) => {
    const partners = parseFields(record.interactors);
    const ids = aliases.get(partners.opponent);
    if (ids?.length !== 1 || ids[0] === record.subject_id) return null;
    delete partners.opponent;
    const condition = contextValues(record.test_condition);
    const keys =
      presentation.pairing?.contextFields ?? Object.keys(condition).sort();
    if (keys.some((key) => !Object.hasOwn(condition, key))) return null;
    const context = JSON.stringify([
      keys.map((key) => [key, condition[key]]),
      Object.entries(partners).sort(),
    ]);
    return { opponent: ids[0], context };
  };
  const records = new Map<string, Observation[]>();
  // Trial numbers are local repetition counters, not shared comparison IDs.
  // Uniqueness must hold across trials; explicit source IDs belong in context.
  const key = (record: Observation, a: string, b: string, context: string) =>
    JSON.stringify([record.item_id, a, b, context]);
  for (const record of model.observations) {
    const match = identity(record);
    if (!match) continue;
    const id = key(record, record.subject_id, match.opponent, match.context);
    records.set(id, [...(records.get(id) ?? []), record]);
  }
  const pairs = new Map<string, Observation>();
  for (const record of model.observations) {
    const match = identity(record);
    if (!match) continue;
    const own = records.get(
      key(record, record.subject_id, match.opponent, match.context),
    );
    const other = records.get(
      key(record, match.opponent, record.subject_id, match.context),
    );
    if (
      own?.length === 1 &&
      other?.length === 1 &&
      other[0].trial === record.trial
    )
      pairs.set(record.response_id, other[0]);
  }
  return pairs;
}
