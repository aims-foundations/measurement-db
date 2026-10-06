"""Read publication decisions before CI installs builders or contacts sources.

Only PyYAML is required; importing the table-building package here would require
the entire build stack before we can exclude withheld benchmarks.
"""

from pathlib import Path

import yaml


class ReleaseLoader(yaml.SafeLoader):
    """Reject duplicate keys, including contradictory release declarations."""


def unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


ReleaseLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def validate_public_release(benchmark: dict) -> tuple[str, str]:
    release = benchmark.get("release")
    if release not in ("public", "withheld"):
        raise ValueError("benchmark.release must explicitly be public or withheld in this repository")
    reason = benchmark.get("release_reason", "")
    if not isinstance(reason, str) or (release == "withheld" and not reason.strip()):
        raise ValueError("benchmark.release_reason must explain why a benchmark is withheld")
    return release, reason.strip()


def load_release(path: Path) -> tuple[str, str]:
    try:
        metadata = yaml.load(path.read_text(encoding="utf-8"), Loader=ReleaseLoader)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("benchmark"), dict):
            raise ValueError("expected a benchmark mapping")
        return validate_public_release(metadata["benchmark"])
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
