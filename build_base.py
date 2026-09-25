"""Shared lifecycle and validation for benchmark builders.

For the curation workflow and authoring instructions, follow the maintained
guide linked from ``README.md`` and start from ``benchmarks/_template/``.
Canonical metadata and table contracts live in
``benchmark_metadata_schema.yaml`` and ``parquet_schemas.yaml``.

Modern builders provide ``metadata.yaml`` and implement ``build_tables()``
or the existing ``build_subject_item_response_rows()`` hook. A short ``download()``
hook selects named upstream sources from metadata using ``fetch_sources()``;
``self.source_files`` identifies the verified inputs. Archive restoration remains
available explicitly, and for older definitions without upstream selections. A
table builder returns DataFrames with local keys; the shared layer commits
final subjects, items, and responses through
:meth:`BenchmarkBuild.add_subject`, ``add_item``, and ``add_response``.
:meth:`BenchmarkBuild.main` validates source provenance from metadata.yaml and
writes the Parquet output. No second provenance manifest is generated.
"""

from abc import ABC
import copy
import argparse
import hashlib
import json
from numbers import Integral
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

# Quiet the notice spam from the HuggingFace libraries.
os.environ.setdefault("DATASETS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.build_measurement_tables as _tables  # noqa: E402
from scripts.build_measurement_tables import (  # noqa: E402
    define_benchmark_vocabulary as _benchmark_vocabulary,
    hash_measurement_ids as _measurement_ids,
    load_source_files as _source_files,
    normalize_evaluation_settings as _evaluation_settings,
    validate_benchmark_metadata as _benchmark_metadata,
    source_snapshots as _source_snapshots,
)

# Intentional author-facing verifier specifications. Everything else imported
# from the measurement-table package remains an implementation dependency.
ExactMatcher = _tables.ExactMatcher
Judge = _tables.Judge
_REFERENCE_FROM_ITEM = object()


class BuildContractError(RuntimeError):
    """Raised when a ``BenchmarkBuild`` subclass violates its author contract."""


class BenchmarkBuild(ABC):
    """Shared pipeline for benchmark-specific builders.

    Subclasses implement :meth:`build_tables` or :meth:`build_subject_item_response_rows` and override
    :meth:`download` only when the metadata-driven HTTP downloader is
    insufficient.
    """

    # Inline configuration remains available; metadata.yaml is preferred.
    INFO: _benchmark_vocabulary.BenchmarkInfo = {}  # type: ignore[assignment]
    slug: str = ""   # short benchmark id, e.g. "jailbreakbench"
    name: str = ""   # display name, e.g. "JailbreakBench"
    BUILD_CONTRACT_VERSION: int = 1

    def __init__(self, benchmark_file: str):
        """`benchmark_file` is the child's `__file__`; the dataset directory and
        all output paths are derived from it. The subclass definition is
        validated before any directory is created or data is downloaded."""
        self.dir = Path(benchmark_file).resolve().parent
        self.raw_dir = self.dir / "raw"
        self.tables_dir = self.dir / "formatted_tables"
        self.responses_path = self.tables_dir / "responses.parquet"
        self.traces_path = self.tables_dir / "traces.parquet"
        self.assets_path = self.tables_dir / "assets.parquet"
        self.source_manifest: dict[str, object] = {}
        self.source_files: tuple[str, ...] = ()
        self._source_artifacts: list[dict] = []
        self._source_archive: dict | None = None
        self.archive_layout: dict[str, object] = {}
        self.expectations: dict[str, object] = {}
        self.grading: dict[str, object] = {}
        self.build_parameters: dict[str, dict[str, str]] = {}
        # ``main()`` resets and activates this per-run state immediately
        # before calling the benchmark-specific
        # ``build_subject_item_response_rows()`` hook.
        self._active_benchmark_id: str | None = None
        self._response_rows: list[dict[str, object]] = []
        self._response_keys: set[
            tuple[str, str, int, str | None, str | None]
        ] = set()
        self._response_ids: set[str] = set()
        self._asset_rows: dict[str, dict[str, object]] = {}
        self._asset_id_by_source_path: dict[Path, str] = {}
        self._item_asset_ids: dict[str, set[str]] = {}
        self._item_response_scales: dict[str, dict] = {}

        metadata_path = self.dir / "metadata.yaml"
        if metadata_path.exists():
            if not metadata_path.is_file():
                raise BuildContractError(
                    f"{metadata_path} exists but is not a file"
                )

            conflicting_attributes = sorted(
                attribute
                for attribute in (
                    "INFO",
                    "slug",
                    "name",
                    "BUILD_CONTRACT_VERSION",
                    "source_manifest",
                    "archive_layout",
                    "expectations",
                    "grading",
                    "build_parameters",
                )
                if attribute in type(self).__dict__
            )
            if conflicting_attributes:
                raise BuildContractError(
                    f"{type(self).__name__} has {metadata_path.name} and also "
                    f"declares {conflicting_attributes}; keep the definition in "
                    f"{metadata_path.name} only"
                )

            try:
                metadata = _benchmark_metadata.load_benchmark_metadata(metadata_path)
            except _benchmark_metadata.BenchmarkMetadataError as exc:
                raise BuildContractError(str(exc)) from exc

            benchmark_metadata = metadata["benchmark"]
            build_metadata = metadata["build"]
            self.build_parameters = copy.deepcopy(build_metadata.get("parameters", {}))

            for section, attribute in (
                ("sources", "source_manifest"),
                ("archive_layout", "archive_layout"),
                ("expectations", "expectations"),
                ("grading", "grading"),
            ):
                if section not in metadata:
                    continue
                setattr(self, attribute, copy.deepcopy(metadata[section]))

            self.slug = self.dir.name
            self.name = copy.deepcopy(benchmark_metadata["name"])
            self.INFO = copy.deepcopy({
                key: value
                for key, value in benchmark_metadata.items()
                if key != "name"
            })
            self.BUILD_CONTRACT_VERSION = copy.deepcopy(
                build_metadata["contract_version"]
            )

        self._validate_definition()
        self.raw_dir.mkdir(exist_ok=True)

    def _validate_definition(self) -> None:
        """Validate the benchmark definition before performing any side effects."""

        problems: list[str] = []
        class_name = type(self).__name__
        slug = self.slug
        name = self.name
        contract_version = self.BUILD_CONTRACT_VERSION
        info = self.INFO
        context = slug or class_name

        if (type(self).build_tables is BenchmarkBuild.build_tables
                and type(self).build_subject_item_response_rows is BenchmarkBuild.build_subject_item_response_rows):
            problems.append("implement build_tables() or build_subject_item_response_rows()")

        # Benchmark identity
        if not isinstance(slug, str) or not slug.strip():
            problems.append("`slug` must be a non-empty string")
        elif not re.fullmatch(r"[A-Za-z0-9_]+", slug):
            problems.append("`slug` must contain only letters, digits, and `_`")
        elif slug != slug.lower():
            problems.append(
                "`slug` must contain only lowercase letters, digits, and `_`"
            )
        if not isinstance(name, str) or not name.strip():
            problems.append("`name` must be a non-empty display string")
        if isinstance(slug, str) and slug and self.dir.name != slug:
            problems.append(
                f"slug {slug!r} must equal benchmark folder {self.dir.name!r}"
            )

        # Build contract version
        if (
            isinstance(contract_version, bool)
            or not isinstance(contract_version, int)
            or contract_version not in (1, 2)
        ):
            problems.append("`BUILD_CONTRACT_VERSION` must be 1 or 2")

        # Benchmark metadata
        try:
            _benchmark_vocabulary.validate_info(info, context=context)
        except ValueError as exc:
            problems.append(str(exc))
        if (
            isinstance(info, dict)
            and isinstance(info.get("benchmark_features"), dict)
        ):
            try:
                _evaluation_settings.canonicalize_features(
                    info["benchmark_features"]
                )
            except ValueError as exc:
                problems.append(
                    f"{context}: invalid INFO "
                    f"'benchmark_features' — {exc}"
                )
        if isinstance(info, dict) and info.get("multi_single_turn") is None:
            problems.append(
                "contract version 1 requires INFO['multi_single_turn']"
            )

        if problems:
            raise BuildContractError(
                f"{class_name} violates the BenchmarkBuild author contract — "
                + "; ".join(problems)
            )

    def _download(
        self,
        url: str,
        dest: Path,
        min_size: int = 100,
        timeout: int = 60,
        announce_cache: bool = False,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> Path:
        """Download and optionally verify one cached source artifact.

        Historical callers retain the ``size > min_size`` cache rule; as before,
        that threshold applies only to existing caches. A pinned caller can
        instead provide an exact byte count and/or SHA-256, which applies to both
        cached and newly fetched files. Invalid caches are replaced only after a
        pinned temporary file verifies.
        """
        has_pinned_integrity = (
            expected_size is not None or expected_sha256 is not None
        )
        if dest.exists():
            try:
                if has_pinned_integrity:
                    _source_files.verify_file(
                        dest,
                        expected_size=expected_size,
                        expected_sha256=expected_sha256,
                    )
                else:
                    _source_files.verify_file(dest, minimum_size=min_size + 1)
            except _source_files.SourceDataError:
                pass
            else:
                if announce_cache:
                    print(f"[{self.slug}] cached {dest}")
                return dest

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "measurement-db", **(request_headers or {})},
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=dest.parent,
                    prefix=f".{dest.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    shutil.copyfileobj(response, temporary)
            if has_pinned_integrity:
                try:
                    _source_files.verify_file(
                        temporary_path,
                        expected_size=expected_size,
                        expected_sha256=expected_sha256,
                    )
                except _source_files.SourceDataError as exc:
                    raise _source_files.SourceDataError(
                        f"[{self.slug}] downloaded {url} for {dest} failed "
                        f"integrity validation: {exc}"
                    ) from None
            temporary_path.replace(dest)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return dest

    def add_subject(
        self,
        raw_label: str,
        *,
        features: dict | None = None,
        access_date: str | None = None,
    ) -> str:
        """Register one final subject row and return its derived ID.

        Call this only from ``build_subject_item_response_rows()``. Subject
        features and the access date are identity-bearing, so they must already
        be final at this commit boundary.
        """

        if self._active_benchmark_id is None:
            raise BuildContractError(
                f"{self.slug}: add_subject() may only be called from "
                "build_subject_item_response_rows() while main() is running"
            )
        return _tables.resolve_subject(
            raw_label,
            features=self.subject_settings(raw_label, features or {}),
            access_date=access_date,
        )

    def add_item(
        self,
        *,
        raw_item_id: str,
        content: str | None,
        attachments: list[dict[str, object]] | None = None,
        grading_criterion: dict | str,
        verifier: Judge | ExactMatcher,
        features: dict | None = None,
        verifier_features: dict | None = None,
    ) -> str:
        """Register one final item row and return its derived ID.

        Call this only from ``build_subject_item_response_rows()``. Item criterion and
        verifier fields (including the rubric or rule) and attachment links
        are identity-bearing and must be supplied here rather than through response
        ``settings``. The effective response scale is resolved from ``INFO`` and
        the criterion and included in identity without duplicating inherited
        scales in item records. Each attachment mapping contains either ``source_path``
        (a file beneath ``raw/``) or ``data`` (bytes already read from a source table),
        ``path`` (its stable logical POSIX path),
        ``media_type``, and ``role``. Exact bytes are content-addressed and
        written once to ``assets.parquet``.
        """

        benchmark_id = self._active_benchmark_id
        if benchmark_id is None:
            raise BuildContractError(
                f"{self.slug}: add_item() may only be called from "
                "build_subject_item_response_rows() while main() is running"
            )
        if content is not None and (
            not isinstance(content, str) or not content.strip()
        ):
            raise BuildContractError(
                f"{self.slug}: add_item() content must be a non-empty string or None"
            )
        if attachments is not None and not isinstance(attachments, list):
            raise BuildContractError(
                f"{self.slug}: add_item() attachments must be a list or None"
            )
        if content is None and not attachments:
            raise BuildContractError(
                f"{self.slug}: add_item() requires non-empty text content or "
                "at least one attachment"
            )
        try:
            grading_criterion = _tables.canonical_grading_criterion(grading_criterion)
            response_scale = _tables.item_response_scale(self.INFO["response_scale"], grading_criterion)
        except (TypeError, ValueError) as exc:
            raise BuildContractError(f"{self.slug}: add_item() {exc}") from exc
        manifest_entries: list[dict[str, object]] = []
        seen_logical_paths: set[str] = set()
        allowed_attachment_fields = {
            "path",
            "media_type",
            "role",
        }
        raw_root = self.raw_dir.resolve()
        for ordinal, attachment in enumerate(attachments or [], start=1):
            context = f"{self.slug}: add_item() attachment {ordinal}"
            if not isinstance(attachment, dict):
                raise BuildContractError(f"{context} must be a mapping")
            if set(attachment) not in (
                allowed_attachment_fields | {"source_path"},
                allowed_attachment_fields | {"data"},
            ):
                raise BuildContractError(
                    f"{context} must contain {sorted(allowed_attachment_fields)} "
                    "and exactly one of source_path or data"
                )

            resolved_source = None
            if "data" in attachment:
                payload = attachment["data"]
                if not isinstance(payload, bytes):
                    raise BuildContractError(f"{context}.data must be bytes")
            else:
                supplied_source = attachment["source_path"]
                if not isinstance(supplied_source, (str, Path)):
                    raise BuildContractError(f"{context}.source_path must be a string or Path")
                source_path = Path(supplied_source)
                if not source_path.is_absolute():
                    source_path = self.raw_dir / source_path
                try:
                    resolved_source = source_path.resolve(strict=True)
                except (FileNotFoundError, OSError) as exc:
                    raise BuildContractError(
                        f"{context}.source_path is not a readable file: {source_path}"
                    ) from exc
                if not resolved_source.is_relative_to(raw_root):
                    raise BuildContractError(
                        f"{context}.source_path must stay beneath {self.raw_dir}"
                    )
                if not resolved_source.is_file():
                    raise BuildContractError(f"{context}.source_path is not a file: {source_path}")

            logical_path = attachment["path"]
            if not isinstance(logical_path, str) or not logical_path:
                raise BuildContractError(
                    f"{context}.path must be a non-empty relative POSIX path"
                )
            parsed_logical_path = PurePosixPath(logical_path)
            if (
                parsed_logical_path.is_absolute()
                or logical_path != parsed_logical_path.as_posix()
                or any(part in ("", ".", "..") for part in parsed_logical_path.parts)
                or re.match(r"^[A-Za-z]:", logical_path)
                or "\\" in logical_path
                or any(ord(character) < 32 for character in logical_path)
            ):
                raise BuildContractError(
                    f"{context}.path must be a normalized relative POSIX path "
                    "without traversal"
                )
            if logical_path in seen_logical_paths:
                raise BuildContractError(
                    f"{context}.path {logical_path!r} is repeated for this item"
                )

            media_type = attachment["media_type"]
            if not isinstance(media_type, str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*",
                media_type,
            ):
                raise BuildContractError(
                    f"{context}.media_type must be a lowercase MIME type"
                )
            role = attachment["role"]
            if not isinstance(role, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*", role
            ):
                raise BuildContractError(
                    f"{context}.role must match [a-z][a-z0-9_]*"
                )

            asset_id = self._asset_id_by_source_path.get(resolved_source)
            if asset_id is None:
                if resolved_source is not None:
                    payload = resolved_source.read_bytes()
                asset_id = _measurement_ids.asset_id_from_bytes(payload)
                existing = self._asset_rows.get(asset_id)
                if existing is not None and existing["data"] != payload:
                    raise BuildContractError(
                        f"{self.slug}: SHA-256 collision for asset {asset_id}"
                    )
                self._asset_rows.setdefault(
                    asset_id,
                    {
                        "asset_id": asset_id,
                        "benchmark_id": benchmark_id,
                        "byte_size": len(payload),
                        "data": payload,
                    },
                )
                if resolved_source is not None:
                    self._asset_id_by_source_path[resolved_source] = asset_id

            seen_logical_paths.add(logical_path)
            manifest_entries.append(
                {
                    "asset_id": asset_id,
                    "path": logical_path,
                    "media_type": media_type,
                    "role": role,
                    "ordinal": ordinal,
                }
            )

        asset_manifest = _measurement_ids.canonical_asset_manifest(manifest_entries)
        item_id = _tables.register_item(
            benchmark_id,
            raw_item_id,
            content,
            asset_manifest=asset_manifest,
            grading_criterion=grading_criterion,
            verifier=verifier,
            response_scale=self.INFO["response_scale"],
            features=features,
            verifier_features=verifier_features,
        )
        linked_asset_ids = {
            str(entry["asset_id"]) for entry in manifest_entries
        }
        prior_asset_ids = self._item_asset_ids.setdefault(item_id, linked_asset_ids)
        if prior_asset_ids != linked_asset_ids:
            raise BuildContractError(
                f"{self.slug}: item_id collision {item_id!r} across asset manifests"
            )
        self._item_response_scales[item_id] = response_scale
        return item_id

    def add_response(
        self,
        *,
        subject_id: str,
        item_id: str,
        response: float | None,
        reference_answer: str | None = _REFERENCE_FROM_ITEM,
        trace: str | None,
        trial: int = 1,
        test_condition: str | None = None,
        interactors: str | None = None,
        **extra_columns: object,
    ) -> str:
        """Validate and commit one final response row, then return its ID.

        The row must already be in its final response-table form: subject/item
        identities, trial, condition, and interactors may
        not be changed later. Reference answers are read from the item criterion; raw trace text is written only to
        traces.parquet, linked by the returned response_id.

        The optional reference_answer argument is a compatibility input for
        version-1 ID reproduction, not a response column. New builders omit
        it. A supplied non-null value must agree with the finalized criterion;
        response creation cannot change item grading data. Explicit None preserves IDs of legacy
        builds that left the copy null.

        A released attempt with no usable grade may retain ``response=None``.
        Its identity and trace are preserved; zero remains an observed grade.
        Use an absent row for an attempt that was never released.
        """

        benchmark_id = self._active_benchmark_id
        if benchmark_id is None:
            raise BuildContractError(
                f"{self.slug}: add_response() may only be called from "
                "build_subject_item_response_rows() while main() is running"
            )
        declared = self.INFO.get("granularity") or "item"
        if declared != "item":
            raise BuildContractError(
                f"{self.slug}: add_response() cannot add observations when "
                f"INFO['granularity'] is {declared!r}; non-item builds must "
                "register the bank/subjects without calling add_response()"
            )

        response_columns = _tables.parquet_columns(
            "responses", include_derived=True
        )
        if extra_columns:
            raise BuildContractError(
                f"{self.slug}: add_response() only accepts the fixed response "
                f"schema; unsupported column(s) {sorted(extra_columns)}. "
                "Keep source evidence in pinned raw inputs and document the "
                "transformation in the builder; subject/item metadata belongs "
                "in add_subject()/add_item()."
            )
        if not isinstance(subject_id, str):
            raise BuildContractError(
                f"{self.slug}: add_response() subject_id must be a string"
            )
        if not isinstance(item_id, str):
            raise BuildContractError(
                f"{self.slug}: add_response() item_id must be a string"
            )
        if (
            isinstance(trial, bool)
            or not isinstance(trial, Integral)
            or trial < 1
        ):
            raise BuildContractError(
                f"{self.slug}: add_response() trial must be a positive "
                f"1-based integer, got {trial!r}"
            )
        if test_condition is not None and not isinstance(test_condition, str):
            raise BuildContractError(
                f"{self.slug}: add_response() test_condition must be a string "
                "or None"
            )
        if interactors is not None and not isinstance(interactors, str):
            raise BuildContractError(
                f"{self.slug}: add_response() interactors must be a string or None"
            )
        try:
            _tables.validate_grade(response)
        except ValueError as exc:
            raise BuildContractError(
                f"{self.slug}: add_response() response must be a finite number "
                f"or None for an ungraded attempt, got {response!r}"
            ) from exc
        if (
            reference_answer is not _REFERENCE_FROM_ITEM
            and reference_answer is not None
            and not isinstance(reference_answer, str)
        ):
            raise BuildContractError(
                f"{self.slug}: add_response() reference_answer must be a "
                "string or None"
            )
        if trace is not None and not isinstance(trace, str):
            raise BuildContractError(
                f"{self.slug}: add_response() trace must be a string or None"
            )

        try:
            _tables.get_subject_registration(subject_id)
        except KeyError:
            raise BuildContractError(
                f"{self.slug}: add_response() references subject_id "
                f"{subject_id!r} that was not registered in this build"
            ) from None
        try:
            item_registration = _tables.get_item_registration(item_id)
        except KeyError:
            raise BuildContractError(
                f"{self.slug}: add_response() references item_id {item_id!r} "
                "that was not registered in this build"
            ) from None
        if item_registration.get("benchmark_id") != benchmark_id:
            raise BuildContractError(
                f"{self.slug}: add_response() references item_id {item_id!r} "
                f"owned by benchmark {item_registration.get('benchmark_id')!r}"
            )
        try:
            _tables.validate_grade(response, self._item_response_scales[item_id])
        except ValueError as exc:
            raise BuildContractError(f"{self.slug}: item {item_id}: {exc}") from exc

        item_reference = json.loads(item_registration["grading_criterion"])["reference_answer"]
        if reference_answer is _REFERENCE_FROM_ITEM:
            reference_answer = item_reference
        elif reference_answer is not None and reference_answer != item_reference:
            raise BuildContractError(
                f"{self.slug}: reference_answer disagrees with the registered "
                "item; store the authoritative answer in add_item()"
            )

        normalized_trial = int(trial)
        response_key = (
            subject_id,
            item_id,
            normalized_trial,
            test_condition,
            interactors,
        )
        if response_key in self._response_keys:
            raise BuildContractError(
                f"{self.slug}: duplicate add_response() primary key "
                "(subject_id, item_id, trial, test_condition, interactors)="
                f"{response_key!r}; assign the final unique trial before "
                "adding the row"
            )

        # Export the fixed schema order. The version-1 identity
        # payload is reconstructed separately so retired columns do not alter IDs.
        stored_row: dict[str, object] = dict.fromkeys(response_columns)
        stored_row.update(
            {
                "subject_id": subject_id,
                "item_id": item_id,
                "benchmark_id": benchmark_id,
                "trial": normalized_trial,
                "test_condition": test_condition,
                "interactors": interactors,
                "response": response,
            }
        )
        hash_row = {
            column: value
            for column, value in stored_row.items()
            if column != "response_id"
        }
        response_id = _measurement_ids.response_id_from_row(
            _measurement_ids.response_identity_v1(hash_row, reference_answer)
        )
        if response_id in self._response_ids:
            raise BuildContractError(
                f"{self.slug}: response_id collision {response_id!r} while "
                "adding a distinct response row"
            )

        stored_row["response_id"] = response_id
        stored_row["trace"] = trace

        self._response_keys.add(response_key)
        self._response_ids.add(response_id)
        self._response_rows.append(stored_row)
        return response_id

    # --- subclass hooks --------------------------------------------------
    def fetch_sources(self, *names: str) -> list[str]:
        """Fetch named metadata sources, or restore an explicitly selected archive."""
        if self._source_archive is not None:
            return BenchmarkBuild.download(self)
        artifacts = _source_files.upstream_artifacts(self.source_manifest["upstream"], names, raw_dir=self.raw_dir)
        root = self.raw_dir.resolve()
        for artifact in artifacts:
            target = self.raw_dir / artifact["file"]
            if not target.resolve().is_relative_to(root):
                raise BuildContractError(f"Upstream destination escapes raw/: {target}")
            if target.exists():
                _source_snapshots.verify_snapshot_file(target, artifact)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Verify before installing a file; never replace an existing raw input.
            with tempfile.TemporaryDirectory(prefix=".download-", dir=target.parent) as staging:
                temporary = Path(staging) / "input"
                if "hf_repo" in artifact:
                    from huggingface_hub import hf_hub_download
                    # Keep byte ranges and checksums on the original representation;
                    # compressed CDN responses can break streamed/resumed downloads.
                    cached = Path(hf_hub_download(artifact["hf_repo"], artifact["hf_path"],
                                  repo_type="dataset", revision=artifact["hf_revision"],
                                  headers={"Accept-Encoding": "identity"}))
                    _source_snapshots.verify_snapshot_file(cached, artifact)
                    shutil.copyfile(cached, temporary)
                else:
                    # GCS transcodes gzip for some browser user agents even
                    # when Accept-Encoding is set. Keep the stored bytes.
                    encoding_options = ({"request_headers": {"Accept-Encoding": "gzip", "User-Agent": "measurement-db"}}
                                        if artifact.get("content_encoding") == "gzip" else {})
                    self._download(artifact["url"], temporary, timeout=600,
                                   expected_size=artifact["size"],
                                   expected_sha256=artifact["digest"] if artifact["hash_kind"] == "sha256" else None,
                                   **encoding_options)
                _source_snapshots.verify_snapshot_file(temporary, artifact)
                temporary.replace(target)
        self._source_artifacts = artifacts
        self.source_files = tuple(artifact["file"] for artifact in artifacts)
        return [artifact["url"] for artifact in artifacts]

    def download(self) -> list[str] | tuple[str, ...]:
        """Restore an archive, or fetch legacy static HTTP artifacts.

        New builders select upstream sources with a short override calling
        ``fetch_sources()``. The following legacy behavior remains available for
        contract-1 builders and definitions without upstream file selections.
        Custom implementations must still cache their inputs beneath
        ``self.raw_dir`` and return every declared source locator, including on
        cache hits. Declare custom-acquired files under ``sources.inputs`` in
        metadata.yaml, with their sizes and SHA-256 hashes before building.
        """
        if self._source_archive is not None:
            _source_snapshots.restore_snapshot(self._source_archive, self.raw_dir, self._source_artifacts)
            return [artifact["url"] for artifact in self._source_artifacts]
        if "upstream" in self.source_manifest:
            raise BuildContractError(f"{self.slug}: implement download() with fetch_sources() to select named upstream sources")
        downloads = self.source_manifest.get("downloads")
        if not isinstance(downloads, dict) or not downloads:
            raise BuildContractError(
                f"{self.slug}: define at least one static HTTP artifact under "
                "metadata sources.downloads, or override download() for "
                "custom acquisition"
            )

        source_urls: list[str] = []
        allowed_fields = {"url", "file", "size", "sha256", "timeout"}
        for source_name, descriptor in downloads.items():
            context = f"{self.slug}: sources.downloads.{source_name}"
            if not isinstance(source_name, str) or not re.fullmatch(
                r"\S+", source_name
            ):
                raise BuildContractError(
                    f"{self.slug}: sources.downloads keys must be non-empty "
                    "strings"
                )
            if not isinstance(descriptor, dict):
                raise BuildContractError(f"{context} must be a mapping")
            unknown_fields = sorted(set(descriptor) - allowed_fields)
            if unknown_fields:
                raise BuildContractError(
                    f"{context} has unsupported fields {unknown_fields}"
                )

            url = descriptor.get("url")
            relative_file = descriptor.get("file")
            expected_size = descriptor.get("size")
            expected_sha256 = descriptor.get("sha256")
            timeout = descriptor.get("timeout", 60)
            if not isinstance(url, str) or not re.fullmatch(r"https?://\S+", url):
                raise BuildContractError(
                    f"{context}.url must be a non-empty HTTP(S) URL"
                )
            if not isinstance(relative_file, str) or not re.fullmatch(
                r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*",
                relative_file,
            ):
                raise BuildContractError(
                    f"{context}.file must be a non-empty raw-relative path"
                )
            relative_path = Path(relative_file)
            if (
                relative_path.is_absolute()
                or relative_path == Path(".")
                or ".." in relative_path.parts
            ):
                raise BuildContractError(
                    f"{context}.file must stay within the benchmark's raw directory"
                )
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                raise BuildContractError(
                    f"{context}.size must be a non-negative integer"
                )
            if not isinstance(expected_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_sha256
            ):
                raise BuildContractError(
                    f"{context}.sha256 must be 64 lowercase hexadecimal characters"
                )
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, int)
                or timeout <= 0
            ):
                raise BuildContractError(
                    f"{context}.timeout must be a positive integer"
                )

            self._download(
                url,
                self.raw_dir / relative_path,
                timeout=timeout,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            source_urls.append(url)
        return source_urls

    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Return subjects, items, responses, and optional traces for registration.

        Keys are local to these input tables, not canonical measurement IDs.
        Column names match the add methods, with subject_key/item_key replacing
        subject_id/item_id and response_key linking optional traces. If trial is
        omitted, attempts are numbered in input order after canonical IDs resolve.
        See README.md for required columns and the REAL builder for an example.
        """
        raise NotImplementedError

    def build_subject_item_response_rows(self) -> None:
        """Register build_tables() output using the existing identity/validation path.

        Row-based builders may continue overriding this hook directly. main()
        still owns benchmark registration, output validation, and atomic writing.
        """
        supplied = self.build_tables()
        required = {
            "subjects": {"subject_key", "raw_label"},
            "items": {"item_key", "raw_item_id", "content", "grading_criterion", "verifier"},
            "responses": {"response_key", "subject_key", "item_key", "response"},
            "traces": {"response_key", "trace"},
        }
        optional = {
            "subjects": {"features", "access_date"},
            "items": {"attachments", "features", "verifier_features"},
            # reference_answer is the existing add_response compatibility input;
            # it is never an output column. Explicit None preserves legacy IDs.
            "responses": {"trial", "test_condition", "interactors", "reference_answer"},
            "traces": set(),
        }
        primary_keys = {"subjects": "subject_key", "items": "item_key",
                        "responses": "response_key", "traces": "response_key"}
        if not isinstance(supplied, dict) or not {"subjects", "items", "responses"} <= supplied.keys():
            raise BuildContractError("build_tables() must return subjects, items, and responses DataFrames")
        if supplied.keys() - required.keys():
            raise BuildContractError(f"build_tables(): unknown tables {sorted(supplied.keys() - required.keys())}")

        tables = {}
        for name, frame in supplied.items():
            if not isinstance(frame, pd.DataFrame):
                raise BuildContractError(f"build_tables().{name} must be a DataFrame")
            if not frame.columns.is_unique:
                raise BuildContractError(f"build_tables().{name} has duplicate columns")
            missing = required[name] - set(frame.columns)
            extra = set(frame.columns) - required[name] - optional[name]
            if missing or extra:
                raise BuildContractError(f"build_tables().{name}: missing columns {sorted(missing)}, unknown columns {sorted(extra)}")
            for key in set(frame.columns) & {"subject_key", "item_key", "response_key"}:
                valid = frame[key].map(lambda value: isinstance(value, (str, Integral)) and not isinstance(value, bool))
                if not valid.all():
                    raise BuildContractError(f"build_tables().{name}.{key} requires non-null string or integer keys")
            if frame[primary_keys[name]].duplicated().any():
                raise BuildContractError(f"build_tables().{name} has duplicate {primary_keys[name]}")
            tables[name] = frame.astype(object).where(frame.notna(), None)

        subjects, items, responses = (tables[name] for name in ("subjects", "items", "responses"))
        for key, parent in (("subject_key", subjects), ("item_key", items)):
            if not responses[key].isin(parent[key]).all():
                raise BuildContractError(f"build_tables().responses references an unknown {key}")
        if "traces" in tables and not tables["traces"].response_key.isin(responses.response_key).all():
            raise BuildContractError("build_tables().traces references an unknown response_key")

        # Reuse the existing ID, grading, attachment, and collision checks.
        subject_ids = subjects[["subject_key"]].copy()
        subject_ids["subject_id"] = [
            self.add_subject(**{key: value for key, value in row.items() if key != "subject_key"})
            for row in subjects.to_dict("records")
        ]
        item_ids = items[["item_key"]].copy()
        item_ids["item_id"] = [
            self.add_item(**{key: value for key, value in row.items() if key != "item_key"})
            for row in items.to_dict("records")
        ]
        responses = responses.merge(subject_ids, on="subject_key", how="left", sort=False, validate="many_to_one")
        responses = responses.merge(item_ids, on="item_key", how="left", sort=False, validate="many_to_one")
        if "traces" in tables:
            responses = responses.merge(tables["traces"], on="response_key", how="left", sort=False, validate="one_to_one")
        else:
            responses["trace"] = None
        for column in ("test_condition", "interactors"):
            if column not in responses:
                responses[column] = None
        if "trial" not in responses:
            observation = ["subject_id", "item_id", "test_condition", "interactors"]
            responses["trial"] = responses.groupby(observation, sort=False, dropna=False).cumcount() + 1
        responses = responses.drop(columns=["subject_key", "item_key", "response_key"])
        responses = responses.astype(object).where(responses.notna(), None)
        for row in responses.to_dict("records"):
            self.add_response(**row)

    def _validate_and_write_tables(self) -> pd.DataFrame:
        """Validate the completed build and write its output tables."""
        df = pd.DataFrame(self._response_rows)
        declared = self.INFO.get("granularity") or "item"
        if df.empty and declared == "item":
            raise BuildContractError(
                f"{self.slug}: build_subject_item_response_rows() added no "
                "observations for "
                "granularity='item'. Fix the parser/source, or explicitly "
                "set INFO['granularity'] to 'aggregate'/'not_released' "
                "when that is what upstream released."
            )
        counts = _tables.registration_counts()
        if counts["benchmarks"]:
            raise BuildContractError(
                f"{self.slug}: benchmark registration is owned by main(); "
                "build_subject_item_response_rows() must not call "
                "get_benchmark_id(). The completed build must register exactly "
                f"benchmark_id {self.slug!r}."
            )
        if counts["items_with_stimulus"] != counts["items"]:
            missing_stimulus = counts["items"] - counts["items_with_stimulus"]
            raise BuildContractError(
                f"{self.slug}: contract version 1 registered {missing_stimulus} "
                "item(s) without non-empty text or an asset manifest"
            )
        if len(self._item_asset_ids) != counts["items"]:
            raise BuildContractError(
                f"{self.slug}: every item must be registered through "
                "BenchmarkBuild.add_item(); lower-level register_item() calls "
                "bypass attachment and ownership validation"
            )

        registered_items = pd.DataFrame(
            [
                _tables.get_item_registration(item_id)
                for item_id in self._item_asset_ids
            ],
            columns=_tables.parquet_columns("items"),
        )
        assets = pd.DataFrame(
            [self._asset_rows[key] for key in sorted(self._asset_rows)],
            columns=_tables.parquet_columns("assets"),
        )
        try:
            registered_asset_ids = _tables.validate_asset_relations(
                registered_items,
                assets,
                benchmark_id=self.slug,
                context=self.slug,
                response_scale=self.INFO["response_scale"],
            )
        except RuntimeError as exc:
            raise BuildContractError(str(exc)) from None
        committed_asset_ids = set().union(
            *self._item_asset_ids.values()
        ) if self._item_asset_ids else set()
        if registered_asset_ids != committed_asset_ids:
            raise BuildContractError(
                f"{self.slug}: registered item manifests do not match the "
                "attachments committed through add_item()"
            )

        if df.empty:
            missing_registrations = [
                name for name in ("items", "subjects") if counts[name] == 0
            ]
            if missing_registrations:
                raise BuildContractError(
                    f"{self.slug}: no-response build registered no "
                    f"{' or '.join(missing_registrations)}; register the real "
                    "item bank and at least one provider-reported AI subject."
                )
            traces = pd.DataFrame()
            resp = None
        else:
            traces = df.loc[
                df["trace"].notna(), _tables.parquet_columns("traces")
            ].copy()
            resp = df.drop(columns="trace")
            # Revalidate the assembled serialization boundary before anything
            # is written. DataFrame construction determines final column order
            # and dtypes; no table permits benchmark-specific columns.
            _tables.validate_table(
                "responses",
                resp,
                include_derived=True,
                context=self.slug,
            )
            _tables.validate_table("traces", traces, context=self.slug)
            _tables.validate_trace_relations(resp, traces, context=self.slug)

        n_subjects = int(counts["subjects"])
        n_items = int(counts["items"])
        n_responses = int(len(df))
        denominator = n_items * n_subjects
        # A matrix cell is observed once it has a grade, regardless of the
        # number of trials, conditions, or interactors recorded for that pair.
        n_observed_pairs = (
            len(
                df.loc[df["response"].notna(), ["subject_id", "item_id"]]
                .drop_duplicates()
            )
            if not df.empty else 0
        )
        info = self.INFO
        _tables.get_benchmark_id(
            self.slug,
            name=self.name,
            version=info.get("version"),
            license=info["license"],
            source_url=info["data_source_url"],
            description=info["description"],
            one_line_description=info.get("one_line_description"),
            modality=info["modality"],
            domain=info["domain"],
            multi_single_turn=info.get("multi_single_turn"),
            response_type=info["response_type"],
            response_scale=info["response_scale"],
            categorical=info.get("categorical"),
            paper_url=info["paper_url"],
            release_date=info["release_date"],
            granularity=info.get("granularity"),
            release=info.get("release"),
            benchmark_features=info.get("benchmark_features"),
            n_response_values=(
                int(df["response"].nunique(dropna=True)) if not df.empty else 0
            ),
            n_subjects=n_subjects,
            n_items=n_items,
            n_responses=n_responses,
            max_trial=int(df["trial"].max()) if not df.empty else 0,
            coverage=(
                n_observed_pairs / denominator if denominator else 0.0
            ),
            has_reference_answer=bool(counts["items_with_reference_answer"]),
        )

        # Validate every pending registry table before the first staged write.
        # In particular, an unmapped first-run subject must not leave behind a
        # lone fresh responses.parquet when subject validation fails.
        _tables.validate_registrations(
            self.slug, expected_benchmark_id=self.slug
        )
        # Serialize every table successfully before replacing any published
        # output. This prevents a failed large-asset write from leaving new item
        # manifests pointing at an absent or incomplete sidecar.
        with tempfile.TemporaryDirectory(
            prefix=".measurement-build-",
            dir=self.dir,
        ) as staging_directory:
            staging_dir = Path(staging_directory)
            if resp is not None:
                _tables.write_parquet(
                    resp, staging_dir / "responses.parquet",
                    response_scale=self.INFO["response_scale"], items=registered_items,
                )
            if not traces.empty:
                _tables.write_parquet(traces, staging_dir / "traces.parquet")
            if not assets.empty:
                # One file payload per row group permits readers to retrieve a
                # selected asset without decoding unrelated byte blobs.
                _tables.write_parquet(assets,
                    staging_dir / "assets.parquet",
                    row_group_size=1,
                )
            _tables.save(staging_dir, expected_benchmark_id=self.slug, additional_tables={
                name: table for name, table in (("responses", resp), ("traces", traces), ("assets", assets))
                if table is not None and not table.empty
            })

            self.tables_dir.mkdir(parents=True, exist_ok=True)
            staged_assets = staging_dir / "assets.parquet"
            has_staged_assets = staged_assets.exists()
            if has_staged_assets:
                staged_assets.replace(self.assets_path)

            output_names = (
                "items.parquet",
                "subjects.parquet",
                "benchmarks.parquet",
                "responses.parquet",
                "traces.parquet",
            )
            for output_name in output_names:
                staged_path = staging_dir / output_name
                destination = self.tables_dir / output_name
                if staged_path.exists():
                    staged_path.replace(destination)
                else:
                    destination.unlink(missing_ok=True)
            if not has_staged_assets:
                # Install attachment-free items before removing an older
                # sidecar so a failed replacement cannot leave old manifests
                # dangling.
                self.assets_path.unlink(missing_ok=True)
        # The accepted metadata manifest now covers every reported input.
        # Retire the old generated sidecar only after all outputs are installed.
        (self.raw_dir / "_provenance.json").unlink(missing_ok=True)

        if resp is None:
            print(
                f"[{self.slug}] no per-item responses (granularity={declared}); "
                "wrote registry tables"
                + (" and assets.parquet." if not assets.empty else " only.")
            )
            return df

        return df

    def main_from_args(self, argv=None):
        """CLI entry point for builders supporting a separate local result set."""
        parser = argparse.ArgumentParser(description=type(self).__doc__)
        parser.add_argument("--source", type=Path, help="Local directory in the benchmark's upstream input format")
        parser.add_argument("--output", type=Path, help="Separate output directory (default: SOURCE's sibling formatted_tables/)")
        parser.add_argument("--archive", action="store_true", help="Restore the pinned MeasurementDB HF archive instead of fetching upstream")
        args = parser.parse_args(argv)
        if args.archive and args.source is not None:
            parser.error("--archive and --source select different input sources; choose one")
        if args.source is not None:
            return self._build_local_source(args.source, args.output)
        if args.output is not None:
            parser.error("--output requires --source; omit both for the published-release build")
        self._archive_requested = args.archive
        try:
            return self.main()
        finally:
            self._archive_requested = False

    def _build_local_source(self, source, output):
        """Use the same parser and shared table writer for an explicit local input."""
        source = source.resolve(strict=True)
        output = (output or source.parent / "formatted_tables").resolve()
        if not source.is_dir():
            raise ValueError("--source must be a directory")
        if output == self.dir or output.is_relative_to(self.tables_dir.resolve()) or output.is_relative_to(self.raw_dir.resolve()) or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Local results need a separate output directory outside the source and published tables")
        # The older shared writer retires this legacy file after writing. Never
        # let that cleanup edit a caller-supplied, immutable input directory.
        if (source / "_provenance.json").exists():
            raise ValueError("The selected source contains a legacy sidecar the shared writer would modify")

        def fingerprint():
            hashes = {}
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    hashes[str(path.relative_to(source))] = digest.hexdigest()
            return hashes

        before = fingerprint()
        if not before:
            raise ValueError("The selected source directory is empty")
        original = self.dir, self.raw_dir, self.tables_dir, self.responses_path, self.traces_path, self.assets_path, self.source_files
        if output.exists() and any(output.glob("*.parquet")):
            raise ValueError("Choose a new output directory; existing tables are not overwritten")
        output.mkdir(parents=True, exist_ok=True)
        self.dir, self.raw_dir, self.tables_dir = output, source, output
        self.responses_path, self.traces_path, self.assets_path = (
            output / f"{name}.parquet" for name in ("responses", "traces", "assets"))
        self.source_files = tuple(before)
        self._response_rows = []
        self._response_keys = set()
        self._response_ids = set()
        self._asset_rows = {}
        self._asset_id_by_source_path = {}
        self._item_asset_ids = {}
        self._item_response_scales = {}
        self._active_benchmark_id = self.slug
        self._local_source = True
        try:
            result = self.build_subject_item_response_rows()
            if result is not None:
                raise ValueError("The builder must register rows, not return another data format")
            if fingerprint() != before:
                raise ValueError("Input files changed while building")
            result = self._validate_and_write_tables()
            (output / "source.json").write_text(json.dumps(
                {"source": str(source), "sha256": before}, indent=2) + "\n")
            return result
        finally:
            self._local_source = False
            self._active_benchmark_id = None
            self.dir, self.raw_dir, self.tables_dir, self.responses_path, self.traces_path, self.assets_path, self.source_files = original

    def subject_settings(self, label, defaults):
        """Optional recorded run settings override historical release defaults."""
        path = self.raw_dir / "subject_settings.json"
        if not path.is_file():
            return defaults
        settings = json.loads(path.read_text())
        if label not in settings:
            raise ValueError(f"Missing recorded settings for subject {label!r}")
        return {**defaults, **settings[label]}

    def main(self) -> pd.DataFrame:
        metadata_path = self.dir / "metadata.yaml"
        try:
            metadata = _benchmark_metadata.load_benchmark_metadata(metadata_path)
            manifest = copy.deepcopy(metadata.get("sources", {}))
            named_upstream = any("name" in source for source in manifest.get("upstream", []))
            archive_requested = getattr(self, "_archive_requested", False) or any(
                key in os.environ for key in ("MEASUREMENT_DB_SOURCE_REPO",
                "MEASUREMENT_DB_SOURCE_REVISION", "MEASUREMENT_DB_SOURCE_MANIFEST"))
            self._source_archive = None
            if metadata["build"]["contract_version"] == 2 and (archive_requested or not named_upstream):
                self._source_archive = _source_snapshots.snapshot_location(self.dir)
                artifacts = _source_snapshots.snapshot_artifacts(self._source_archive)
            elif named_upstream:
                artifacts = []  # The download hook selects and resolves these inputs.
            else:
                artifacts = _benchmark_metadata.declared_source_artifacts(manifest)
        except (TypeError, ValueError) as exc:
            raise BuildContractError(f"{self.slug}: {exc}") from exc
        # Always load provenance from the authoritative file, including when a
        # legacy caller constructed an inline INFO definition programmatically.
        self.source_manifest = copy.deepcopy(manifest)
        self._source_artifacts = artifacts
        self.source_files = tuple(artifact["file"] for artifact in artifacts)
        download_sources = self.download()
        artifacts = self._source_artifacts
        if not isinstance(download_sources, (list, tuple)):
            raise BuildContractError(
                f"{self.slug}: download() must return a list or tuple of "
                "upstream source URLs or paths"
            )
        if not download_sources or any(
            not isinstance(source, str) or not source.strip()
            for source in download_sources
        ):
            raise BuildContractError(
                f"{self.slug}: download() must return at least one non-empty "
                "source URL or path"
            )
        declared = {artifact["url"] for artifact in artifacts}
        reported = set(download_sources)
        if declared != reported:
            raise BuildContractError(
                f"{self.slug}: download() source locators differ from metadata.yaml; "
                f"undeclared={sorted(reported - declared)!r}, "
                f"unreported={sorted(declared - reported)!r}. "
                "Declare every input before building, including cache hits."
            )
        self._verify_source_manifest(metadata_path, manifest, artifacts)

        # The slug is already the benchmark ID used in item/response identity.
        # Register the complete benchmark row only after all build rows exist.
        self._active_benchmark_id = self.slug
        self._response_rows = []
        self._response_keys = set()
        self._response_ids = set()
        self._asset_rows = {}
        self._asset_id_by_source_path = {}
        self._item_asset_ids = {}
        self._item_response_scales = {}
        try:
            result = self.build_subject_item_response_rows()
            if result is not None:
                raise BuildContractError(
                    f"{self.slug}: build_subject_item_response_rows() must "
                    "return None and commit rows through add_subject(), "
                    "add_item(), and add_response()"
                )
            self._verify_source_manifest(metadata_path, manifest, artifacts)
            return self._validate_and_write_tables()
        finally:
            self._active_benchmark_id = None

    def _verify_source_manifest(self, path: Path, manifest: dict, artifacts: list[dict]) -> None:
        """Check the declared manifest and cached bytes before accepting output."""
        try:
            current = _benchmark_metadata.load_benchmark_metadata(path)
            if current.get("sources", {}) != manifest or self.source_manifest != manifest:
                raise ValueError("sources changed during the build; update metadata.yaml and restart")
            root = self.raw_dir.resolve()
            for artifact in artifacts:
                cached = self.raw_dir / artifact["file"]
                if not cached.resolve().is_relative_to(root) or not cached.is_file():
                    raise ValueError(f"declared raw input is missing or outside raw/: {cached}")
                if "hash_kind" in artifact:
                    _source_snapshots.verify_snapshot_file(cached, artifact)
                else:
                    _source_files.verify_file(
                        cached, expected_size=artifact["size"],
                        expected_sha256=artifact["sha256"],
                    )
        except (OSError, TypeError, ValueError, _source_files.SourceDataError) as exc:
            raise BuildContractError(f"{self.slug}: source manifest validation failed: {exc}") from exc
