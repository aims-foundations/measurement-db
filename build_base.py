"""Shared lifecycle and validation for benchmark builders.

For the curation workflow and authoring instructions, see
``docs/curate_benchmark_data.md`` and start from ``benchmarks/_template/``.
Canonical metadata and table contracts live in
``benchmark_metadata_schema.yaml`` and ``parquet_schemas.yaml``.

Modern builders provide ``metadata.yaml`` and implement
``build_subject_item_response_rows()``. Static HTTP artifacts belong under
``sources.downloads``; override ``download()`` only for custom acquisition. A
builder commits final subjects, items, and responses through
:meth:`BenchmarkBuild.add_subject`, ``add_item``, and ``add_response``.
:meth:`BenchmarkBuild.main` handles validation, provenance, and parquet output.
"""

from abc import ABC, abstractmethod
import copy
import json
from numbers import Integral, Real
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
)

# Intentional author-facing verifier specifications. Everything else imported
# from the measurement-table package remains an implementation dependency.
ExactMatcher = _tables.ExactMatcher
Judge = _tables.Judge


class BuildContractError(RuntimeError):
    """Raised when a ``BenchmarkBuild`` subclass violates its author contract."""


class BenchmarkBuild(ABC):
    """Shared pipeline for benchmark-specific builders.

    Subclasses implement :meth:`build_subject_item_response_rows` and override
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
        self.responses_path = self.dir / "responses.parquet"
        self.traces_path = self.dir / "traces.parquet"
        self.assets_path = self.dir / "assets.parquet"
        self.source_manifest: dict[str, object] = {}
        self.archive_layout: dict[str, object] = {}
        self.expectations: dict[str, object] = {}
        # ``main()`` resets and activates this per-run state immediately
        # before calling the benchmark-specific
        # ``build_subject_item_response_rows()`` hook.
        self._active_benchmark_id: str | None = None
        self._response_rows: list[dict[str, object]] = []
        self._response_keys: set[
            tuple[str, str, int, str | None, str | None]
        ] = set()
        self._response_ids: set[str] = set()
        self._response_extra_columns: tuple[str, ...] | None = None
        self._asset_rows: dict[str, dict[str, object]] = {}
        self._asset_id_by_source_path: dict[Path, str] = {}
        self._item_asset_ids: dict[str, set[str]] = {}

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

            for section, attribute in (
                ("sources", "source_manifest"),
                ("archive_layout", "archive_layout"),
                ("expectations", "expectations"),
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
            or contract_version != 1
        ):
            problems.append("`BUILD_CONTRACT_VERSION` must be 1")

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
            headers={"User-Agent": "Mozilla/5.0"},
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
            features=features,
            access_date=access_date,
        )

    def add_item(
        self,
        *,
        raw_item_id: str,
        content: str | None,
        attachments: list[dict[str, object]] | None = None,
        reference_answer: str | None = None,
        verifier: Judge | ExactMatcher | None = None,
        features: dict | None = None,
        verifier_features: dict | None = None,
    ) -> str:
        """Register one final item row and return its derived ID.

        Call this only from ``build_subject_item_response_rows()``. Item and
        verifier features and attachment links are identity-bearing and
        therefore must be supplied here rather than deferred through response
        ``settings``. Each attachment mapping contains ``source_path`` (a file
        beneath ``raw/``), ``path`` (its stable logical POSIX path),
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
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise BuildContractError(
                f"{self.slug}: add_item() reference_answer must be a string "
                "or None"
            )
        manifest_entries: list[dict[str, object]] = []
        seen_logical_paths: set[str] = set()
        allowed_attachment_fields = {
            "source_path",
            "path",
            "media_type",
            "role",
        }
        raw_root = self.raw_dir.resolve()
        for ordinal, attachment in enumerate(attachments or [], start=1):
            context = f"{self.slug}: add_item() attachment {ordinal}"
            if not isinstance(attachment, dict):
                raise BuildContractError(f"{context} must be a mapping")
            if set(attachment) != allowed_attachment_fields:
                raise BuildContractError(
                    f"{context} must contain exactly "
                    f"{sorted(allowed_attachment_fields)}"
                )

            supplied_source = attachment["source_path"]
            if not isinstance(supplied_source, (str, Path)):
                raise BuildContractError(
                    f"{context}.source_path must be a string or Path"
                )
            source_path = Path(supplied_source)
            if not source_path.is_absolute():
                source_path = self.raw_dir / source_path
            try:
                resolved_source = source_path.resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                raise BuildContractError(
                    f"{context}.source_path is not a readable file: {source_path}"
                ) from exc
            try:
                resolved_source.relative_to(raw_root)
            except ValueError:
                raise BuildContractError(
                    f"{context}.source_path must stay beneath {self.raw_dir}"
                ) from None
            if not resolved_source.is_file():
                raise BuildContractError(
                    f"{context}.source_path is not a file: {source_path}"
                )

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
            reference_answer=reference_answer,
            verifier=verifier,
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
        return item_id

    def add_response(
        self,
        *,
        subject_id: str,
        item_id: str,
        response: float,
        reference_answer: str | None,
        trace: str | None,
        trial: int = 1,
        test_condition: str | None = None,
        interactors: str | None = None,
        **extra_columns: object,
    ) -> str:
        """Validate and commit one final response row, then return its ID.

        The row must already be in its final response-table form: subject/item
        identities, trial, condition, interactors, and extension columns may
        not be changed later. Raw ``trace`` text is retained for
        ``traces.parquet`` but represented as null in the response row that
        enters the hash.
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
        reserved = set(response_columns) | {"settings", "access_date"}
        conflicting = sorted(reserved.intersection(extra_columns))
        if conflicting:
            raise BuildContractError(
                f"{self.slug}: add_response() extension column(s) {conflicting} "
                "are reserved; pass canonical response fields explicitly, and "
                "put final subject/item metadata in add_subject()/add_item()"
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
        if (
            isinstance(response, bool)
            or not isinstance(response, Real)
            or pd.isna(response)
        ):
            raise BuildContractError(
                f"{self.slug}: add_response() response must be a non-null "
                f"number, got {response!r}"
            )
        if reference_answer is not None and not isinstance(reference_answer, str):
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

        extension_names = tuple(extra_columns)
        if self._response_extra_columns is None:
            self._response_extra_columns = extension_names
        elif extension_names != self._response_extra_columns:
            raise BuildContractError(
                f"{self.slug}: every add_response() call must use the same "
                "extension columns in the same order; expected "
                f"{list(self._response_extra_columns)}, got "
                f"{list(extension_names)}"
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

        # Start from schema order because that order is part of response
        # identity. Updating these keys below preserves their positions, and
        # benchmark-specific extension columns are appended afterwards.
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
                "reference_answer": reference_answer,
                # traces.parquet stores the raw text; responses.parquet and its
                # identity always carry null in this column.
                "trace": None,
            }
        )
        stored_row.update(extra_columns)
        hash_row = {
            column: value
            for column, value in stored_row.items()
            if column != "response_id"
        }
        response_id = _measurement_ids.response_id_from_row(hash_row)
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
    def download(self) -> list[str] | tuple[str, ...]:
        """Fetch the static HTTP artifacts declared in ``sources.downloads``.

        Override this hook only when acquisition requires behavior such as
        authentication, dynamic discovery, repository cloning, or streaming.
        Custom implementations must still cache their inputs beneath
        ``self.raw_dir`` and return every source locator, including on cache
        hits.
        """
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

    @abstractmethod
    def build_subject_item_response_rows(self) -> None:
        """Commit source rows through ``add_subject/item/response``.

        ``main()`` owns benchmark registration, so the hook needs no benchmark
        argument. It must return ``None``; all table rows enter through the
        three explicit add methods.
        """
        raise NotImplementedError

    def _validate_and_write_tables(
        self,
        download_sources: list[str],
    ) -> pd.DataFrame:
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
            resp = df.copy()
            resp["trace"] = None
            # Revalidate the assembled serialization boundary before anything
            # is written. DataFrame construction determines final column order
            # and dtypes; allow_extra covers pass-through columns.
            _tables.validate_table(
                "responses",
                resp,
                include_derived=True,
                allow_extra=True,
                context=self.slug,
            )
            _tables.validate_table("traces", traces, context=self.slug)

        n_subjects = int(counts["subjects"])
        n_items = int(counts["items"])
        n_responses = int(len(df))
        denominator = n_items * n_subjects
        info = self.INFO
        _tables.get_benchmark_id(
            self.slug,
            name=self.name,
            license=info["license"],
            source_url=info["data_source_url"],
            description=info["description"],
            one_line_description=info.get("one_line_description"),
            modality=info["modality"],
            domain=info["domain"],
            multi_single_turn=info.get("multi_single_turn"),
            response_type=info["response_type"],
            response_scale=info["response_scale"],
            categorical=info["categorical"],
            paper_url=info["paper_url"],
            release_date=info["release_date"],
            granularity=info.get("granularity"),
            release=info.get("release"),
            benchmark_features=info.get("benchmark_features"),
            n_unique_responses=(
                int(df["response"].nunique(dropna=True)) if not df.empty else 0
            ),
            n_subjects=n_subjects,
            n_items=n_items,
            n_responses=n_responses,
            n_trials=int(df["trial"].max()) if not df.empty else 0,
            coverage=(
                round(n_responses / denominator, 4) if denominator else 0.0
            ),
            has_ground_truth=bool(counts["items_with_reference_answer"]),
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
                resp.to_parquet(staging_dir / "responses.parquet", index=False)
            if not traces.empty:
                traces.to_parquet(staging_dir / "traces.parquet", index=False)
            if not assets.empty:
                # One file payload per row group permits readers to retrieve a
                # selected asset without decoding unrelated byte blobs.
                assets.to_parquet(
                    staging_dir / "assets.parquet",
                    index=False,
                    row_group_size=1,
                )
            _tables.save(staging_dir)

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
                destination = self.dir / output_name
                if staged_path.exists():
                    staged_path.replace(destination)
                else:
                    destination.unlink(missing_ok=True)
            if not has_staged_assets:
                # Install attachment-free items before removing an older
                # sidecar so a failed replacement cannot leave old manifests
                # dangling.
                self.assets_path.unlink(missing_ok=True)
        (self.raw_dir / "_provenance.json").write_text(
            json.dumps({"sources": download_sources}, indent=2)
        )

        if resp is None:
            print(
                f"[{self.slug}] no per-item responses (granularity={declared}); "
                "wrote registry tables"
                + (" and assets.parquet." if not assets.empty else " only.")
            )
            return df

        return df

    def main(self) -> pd.DataFrame:
        download_sources = self.download()
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
        normalized_sources = list(dict.fromkeys(download_sources))

        raw_has_data = self.raw_dir.exists() and any(
            path.is_file() and path.name != "_provenance.json"
            for path in self.raw_dir.rglob("*")
        )
        if not raw_has_data:
            raise BuildContractError(
                f"{self.slug}: download() must cache at least one upstream "
                f"artifact under {self.raw_dir}"
            )

        # The slug is already the benchmark ID used in item/response identity.
        # Register the complete benchmark row only after all build rows exist.
        self._active_benchmark_id = self.slug
        self._response_rows = []
        self._response_keys = set()
        self._response_ids = set()
        self._response_extra_columns = None
        self._asset_rows = {}
        self._asset_id_by_source_path = {}
        self._item_asset_ids = {}
        try:
            result = self.build_subject_item_response_rows()
            if result is not None:
                raise BuildContractError(
                    f"{self.slug}: build_subject_item_response_rows() must "
                    "return None and commit rows through add_subject(), "
                    "add_item(), and add_response()"
                )
            return self._validate_and_write_tables(normalized_sources)
        finally:
            self._active_benchmark_id = None
