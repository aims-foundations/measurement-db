import type {
  ConcreteScale,
  GradingCriterion,
  Item,
  ResponseScale,
} from "./types";

export function itemScale(scale: ResponseScale, item: Item): ConcreteScale {
  if (scale.kind !== "mixed") return scale;
  const criterion = JSON.parse(item.grading_criterion) as GradingCriterion;
  if (!criterion.response_scale)
    throw new Error(`Item ${item.item_id} has no declared grade scale.`);
  return criterion.response_scale;
}

export function gradeLabel(value: number | null, scale: ConcreteScale): string {
  if (value === null) return "Grade unavailable";
  // Validation permits roundoff for fractional categories. Labels use the declared
  // category, while the number displayed remains the exact recorded value.
  const category =
    scale.kind === "discrete"
      ? scale.values.find(
          (category) =>
            Math.abs(category - value) <=
            Number.EPSILON *
              8 *
              Math.max(1, Math.abs(category), Math.abs(value)),
        )
      : value;
  const meaning =
    scale.kind === "discrete" ? scale.meanings?.[String(category)] : undefined;
  return meaning ? `${meaning} (${value})` : String(value);
}

export function describeScale(scale: ResponseScale): string {
  if (scale.kind === "mixed")
    return "Each task uses its own declared grade scale.";
  const domain =
    scale.kind === "discrete"
      ? scale.values.map((value) => gradeLabel(value, scale)).join(" · ")
      : `Grades from ${scale.min ?? "−∞"} to ${scale.max ?? "+∞"}`;
  const direction = {
    higher_is_better: "Higher is better",
    lower_is_better: "Lower is better",
    unordered: "Unordered categories",
  };
  return scale.direction ? `${domain} · ${direction[scale.direction]}` : domain;
}
