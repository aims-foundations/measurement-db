"""Load and validate benchmark ``metadata.yaml`` definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .define_benchmark_vocabulary import validate_benchmark_release_date
from .response_scales import resolve_categorical, validate_scale_type


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "benchmark_metadata_schema.yaml"


class BenchmarkMetadataError(ValueError):
    """A metadata file is unreadable or violates the shared schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            already_seen = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if already_seen:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_mapping(text: str, *, path: Path) -> dict[str, object]:
    try:
        payload = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise BenchmarkMetadataError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkMetadataError(f"{path}: expected a YAML mapping")
    return payload


def _load_schema() -> dict[str, object]:
    try:
        schema_text = _SCHEMA_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"could not read metadata schema {_SCHEMA_PATH}: {exc}"
        ) from exc
    try:
        schema = _load_yaml_mapping(schema_text, path=_SCHEMA_PATH)
    except BenchmarkMetadataError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise RuntimeError(
            f"invalid metadata schema {_SCHEMA_PATH}: {exc.message}"
        ) from exc
    return schema


_VALIDATOR = Draft202012Validator(_load_schema())


def _format_yaml_path(parts: Sequence[object]) -> str:
    return ".".join(str(part) for part in parts) or "<root>"


def _expanded_error_messages(error: ValidationError) -> list[tuple[str, str]]:
    """Turn aggregate JSON Schema errors into one path-aware message per key."""

    absolute_path = tuple(error.absolute_path)
    if error.validator == "not" and absolute_path in {("archive_layout",), ("expectations",)}:
        return [(_format_yaml_path(absolute_path),
                 "unsupported in contract 2; keep parsing rules in the builder and expectations in tests")]
    if error.validator == "required":
        match = re.fullmatch(r"'([^']+)' is a required property", error.message)
        if match:
            missing_key = match.group(1)
            return [
                (
                    _format_yaml_path((*absolute_path, missing_key)),
                    "required key is missing",
                )
            ]

    if error.validator == "additionalProperties" and isinstance(
        error.instance, Mapping
    ):
        allowed_keys = set(error.schema.get("properties", {}))
        unknown_keys = sorted(
            (key for key in error.instance if key not in allowed_keys),
            key=str,
        )
        if unknown_keys:
            messages: list[tuple[str, str]] = []
            for unknown_key in unknown_keys:
                yaml_path = _format_yaml_path((*absolute_path, unknown_key))
                message = "unknown key"
                if yaml_path == "benchmark.slug":
                    message += "; slug is derived from the benchmark folder"
                if yaml_path in {"sources.downloads", "sources.inputs"}:
                    message += "; declare upstream references under sources.upstream"
                messages.append((yaml_path, message))
            return messages

    if error.validator == "type" and error.validator_value == "object":
        return [(_format_yaml_path(absolute_path), "must be a mapping")]

    return [(_format_yaml_path(absolute_path), error.message)]


def _source_reference_problems(metadata: Mapping[str, object]) -> list[str]:
    """Return dangling ``validation`` source references after schema checks."""
    validation = metadata.get("validation")
    if not isinstance(validation, Mapping):
        return []
    source_claims = validation.get("source_claims")
    if not isinstance(source_claims, Mapping):
        return []

    problems: list[str] = []
    for category, claims in source_claims.items():
        if not isinstance(claims, Mapping):
            continue
        for claim_id, claim in claims.items():
            if not isinstance(claim, Mapping):
                continue
            reference = claim.get("source_ref")
            if not isinstance(reference, str):
                continue
            permitted_reference = reference in {
                "benchmark.paper_url",
                "benchmark.data_source_url",
                "sources.upstream",
            } or reference.startswith(
                ("sources.downloads.", "sources.inputs.", "sources.references.")
            )
            if not permitted_reference:
                problems.append(
                    "validation.source_claims."
                    f"{category}.{claim_id}.source_ref: must point to a benchmark "
                    "paper/data URL, sources.upstream, or a legacy source entry"
                )
                continue
            target: object = metadata
            for part in reference.split("."):
                if not isinstance(target, Mapping) or part not in target:
                    problems.append(
                        "validation.source_claims."
                        f"{category}.{claim_id}.source_ref: unresolved metadata "
                        f"reference {reference!r}"
                    )
                    break
                target = target[part]
            else:
                if target is None or target == "":
                    problems.append(
                        "validation.source_claims."
                        f"{category}.{claim_id}.source_ref: metadata reference "
                        f"{reference!r} has no value"
                    )
    return problems


def validate_benchmark_metadata(
    metadata: object,
    *,
    path: str | Path = "metadata.yaml",
) -> None:
    """Validate one parsed benchmark definition against the shared schema."""

    errors = sorted(
        _VALIDATOR.iter_errors(metadata),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    problems = [
        f"{yaml_path}: {message}"
        for error in errors
        for yaml_path, message in _expanded_error_messages(error)
    ]
    if not errors and isinstance(metadata, Mapping):
        try:
            validate_benchmark_release_date(metadata["benchmark"]["release_date"])
        except ValueError as exc:
            problems.append(f"benchmark.release_date: {exc}")
        benchmark = metadata["benchmark"]
        try:
            validate_scale_type(benchmark["response_type"], benchmark["response_scale"])
        except ValueError as exc:
            problems.append(f"benchmark.response_scale: {exc}")
        try:
            resolve_categorical(benchmark["response_type"], benchmark.get("categorical"))
        except ValueError as exc:
            problems.append(f"benchmark.categorical: {exc}")
        problems.extend(_source_reference_problems(metadata))
        sources = metadata.get("sources", {})
        if isinstance(sources, Mapping) and any(key in sources for key in ("downloads", "inputs")):
            try:
                declared_source_artifacts(sources)
            except ValueError as exc:
                problems.append(str(exc))
    if not problems:
        return
    raise BenchmarkMetadataError(
        f"{Path(path)}: invalid benchmark metadata — " + "; ".join(problems)
    )


def load_benchmark_metadata(path: str | Path) -> dict[str, object]:
    """Read and validate one benchmark ``metadata.yaml`` file."""

    metadata_path = Path(path)
    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkMetadataError(
            f"could not read benchmark metadata {metadata_path}: {exc}"
        ) from exc

    metadata = _load_yaml_mapping(metadata_text, path=metadata_path)
    validate_benchmark_metadata(metadata, path=metadata_path)
    return metadata


def declared_source_artifacts(sources: Mapping, *, benchmark_dir: str | Path | None = None) -> list[dict]:
    """Return validated input descriptors; references alone are not inputs.

    Call after metadata-schema validation. Contract 2 obtains the file inventory
    from the pinned Hub tree; legacy contracts declare their inputs inline.
    """
    if "upstream" in sources:
        from .source_snapshots import snapshot_artifacts, snapshot_location
        if benchmark_dir is None:
            raise ValueError("benchmark_dir is required to resolve the source snapshot")
        return snapshot_artifacts(snapshot_location(benchmark_dir))
    artifacts = []
    files = set()
    for group in ("downloads", "inputs"):
        for name, descriptor in sources.get(group, {}).items():
            path = descriptor["file"]
            if path == "_provenance.json":
                raise ValueError("raw/_provenance.json is a retired build artifact, not an upstream input")
            if path in files:
                raise ValueError(f"sources.{group}.{name}: duplicate input file {path!r}")
            files.add(path)
            artifacts.append(descriptor)
    if not artifacts:
        raise ValueError("metadata.yaml must declare inputs under sources.downloads or sources.inputs")
    return artifacts


__all__ = [
    "BenchmarkMetadataError",
    "load_benchmark_metadata",
    "validate_benchmark_metadata",
    "declared_source_artifacts",
]


def main():
    """Validate metadata without downloading data; CI requires the current contract."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--require-current", action="store_true")
    args = parser.parse_args()
    for path in args.paths:
        metadata = load_benchmark_metadata(path)
        if args.require_current and metadata["build"]["contract_version"] != 2:
            parser.error(f"{path}: new and public benchmarks require contract_version: 2")
        print(f"Validated {path}")


if __name__ == "__main__":
    main()
