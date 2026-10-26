"""Interpret gallery fields and grades using their declared table metadata."""

from __future__ import annotations

import colorsys
import json
import re

import pandas as pd

from scripts.build_measurement_tables.response_scales import (
    canonical_response_scale,
    item_response_scale,
    validate_grade,
    validate_scale_type,
)


def parse_fields(value) -> dict[str, str]:
    """Read registry/observation fields without changing their original value."""
    if not isinstance(value, str) or not value:
        return {}
    if value.lstrip().startswith("{"):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return {k: v if isinstance(v, str) else json.dumps(
                v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for k, v in payload.items()}
    parts = [part.partition("=") for part in value.split(";")]
    if not all(equal and re.fullmatch(r"[\w.-]+", key.strip()) for key, equal, _ in parts):
        return {"condition": value}
    return {key.strip(): field.strip() for key, _, field in parts}


def attach_features(frame: pd.DataFrame, subjects: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Keep raw fields intact; join explicitly owned features for display only."""
    subject_fields = {}
    for row in subjects.to_dict("records"):
        fields = {key: row[key] for key in ("harness", "reasoning_effort", "harness_version")
                  if isinstance(row.get(key), str)}
        fields.update(parse_fields(row.get("subject_features_extra")))
        subject_fields[row["subject_id"]] = fields
    item_fields = {row["item_id"]: parse_fields(row.get("item_features"))
                   for row in items.to_dict("records")}
    sources = {
        "condition": frame["test_condition"].map(parse_fields),
        "subject": frame["subject_id"].map(subject_fields),
        "item": frame["item_id"].map(item_fields),
        "interactor": frame.get("interactors", pd.Series(None, index=frame.index)).map(parse_fields),
    }
    names, kinds, used = {}, {}, {"trial"}
    for owner, values in sources.items():
        if values.isna().any():
            raise ValueError(f"Unknown {owner} identifier in responses")
        fields = sorted({key for selection in values for key in selection})
        names[owner] = {}
        for key in fields:
            name = key if key not in used else f"{owner}.{key}"
            while name in used:
                name = f"{owner}.{name}"
            used.add(name)
            names[owner][key] = name
            kinds[name] = owner if owner in {"subject", "item"} else "condition"
        frame[f"_{owner}_features"] = values
    frame["_features"] = [
        {names[owner][key]: value for owner, selection in zip(sources, selections)
         for key, value in selection.items()}
        for selections in zip(*sources.values())
    ]
    # Slice identity contains only the two original observation fields. Registry
    # features never multiply condition slices or consume a trial budget.
    frame["_slice"] = [json.dumps([key[2], key[4] if len(key) == 5 else None],
                                   ensure_ascii=False, separators=(",", ":"))
                       for key in frame["_key"]]
    frame.attrs["dimension_kinds"] = kinds
    frame.attrs["field_names"] = names
    return frame


class ResponseScale:
    """A declared grade domain, including labels and item-specific domains."""

    def __init__(self, metadata: dict, observations: pd.DataFrame, items: pd.DataFrame):
        self.response_type = metadata["response_type"]
        self.domain = json.loads(canonical_response_scale(metadata["response_scale"]))
        validate_scale_type(self.response_type, self.domain)
        self.binary = self.response_type in {"binary", "error_presence"}
        self.mixed = self.domain["kind"] == "mixed"
        self.average = not self.mixed and self.domain.get("direction") != "unordered"
        self.item_domains = ({row.item_id: item_response_scale(self.domain, row.grading_criterion)
                              for row in items.itertuples()} if self.mixed else {})
        observed = observations["response"].dropna()
        self.bounds = self.domain_bounds(self.domain, observed)
        self.item_bounds = {}
        if self.mixed:
            for iid, group in observations.groupby("item_id"):
                self.item_bounds[iid] = self.domain_bounds(self.item_domains[iid], group.response.dropna())
                for value in group.response.dropna().unique():
                    validate_grade(value, self.item_domains[iid])
        else:
            for value in observed.unique():
                validate_grade(value, self.domain)

    @staticmethod
    def domain_bounds(domain: dict, values) -> tuple[float, float]:
        if domain["kind"] == "discrete":
            return min(domain["values"]), max(domain["values"])
        low = domain.get("min")
        high = domain.get("max")
        return (float(low if low is not None else values.min() if len(values) else 0),
                float(high if high is not None else values.max() if len(values) else 1))

    def label(self, value, item_id=None) -> str:
        if pd.isna(value):
            return "Grade unavailable"
        domain = self.item_domains.get(item_id, self.domain)
        declared = min(domain["values"], key=lambda candidate: abs(candidate - value)) if "values" in domain else value
        key = str(int(declared)) if float(declared).is_integer() else repr(float(declared))
        meaning = domain.get("meanings", {}).get(key)
        numeric = str(int(value)) if float(value).is_integer() else repr(float(value))
        return f"{meaning} ({numeric})" if meaning else numeric

    def color(self, value, item_id=None) -> str:
        if pd.isna(value):
            return "#9e9e9e"
        domain = self.item_domains.get(item_id, self.domain)
        if domain.get("direction") == "unordered":
            index = min(range(len(domain["values"])), key=lambda i: abs(domain["values"][i] - value))
            rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.65, 0.75)
            return "#%02x%02x%02x" % tuple(round(v * 255) for v in rgb)
        low, high = self.item_bounds.get(item_id, self.bounds)
        t = (value - low) / (high - low) if high > low else 0.5
        start, end, t = ((31, 119, 180), (242, 242, 242), t * 2) if t < 0.5 else (
            (242, 242, 242), (214, 39, 40), (t - 0.5) * 2)
        return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(start, end))

    def levels(self, observed) -> list[dict]:
        # Validation permits float64 roundoff for fractional categories. Preserve
        # those native grades rather than snapping them to a declared value.
        values = sorted(set(self.domain.get("values", [])) | set(observed.dropna().unique()))
        return [{"value": float(v), "label": self.label(v), "color": self.color(v)} for v in values]

    def binary_labels(self) -> dict | None:
        if not self.binary:
            return None
        meanings = self.domain.get("meanings", {})
        return {"zero": meanings.get("0", "0"), "one": meanings.get("1", "1")}

    def description(self) -> str:
        if self.binary:
            return f"1 = {self.binary_labels()['one']} · 0 = {self.binary_labels()['zero']}"
        if self.mixed:
            return "Item-specific grading scales"
        if self.domain["kind"] == "discrete":
            return "Grades: " + ", ".join(self.label(v) for v in self.domain["values"])
        low, high = self.domain["min"], self.domain["max"]
        if low is None and high is None:
            return "Unbounded numeric score"
        if low is None:
            return f"Score ≤ {high:g}"
        if high is None:
            return f"Score ≥ {low:g}"
        return f"Score in [{low:g}, {high:g}]"
