"""Build ResearchCodeBench from the provider's pinned greedy-run release.

See ``metadata.yaml`` for the immutable source manifest and
``curation_record.md`` for provenance, harness, and coverage decisions.
"""

from __future__ import annotations

import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

IMPORT_ROOT = Path(__file__).resolve().parents[3]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from measurement_db.build_base import BenchmarkBuild, BuildContractError, ExactMatcher

# Snippet markers in the annotated pset/<paper>/*.py files (line-anchored, as
# in upstream core/annotation/models/file.py START_PATTERN/END_PATTERN).
_OPEN = re.compile(r'^\s*#\s*<paper2code\s+name="([^"]+)">\s*$')
_CLOSE = re.compile(r'^\s*#\s*</paper2code\s+name="([^"]+)">\s*$')

# Verbatim prompt scaffolding from core/annotation/utils/run_inference.py.
_HDR_WITH_PAPER_PRE = (
    "\nYou are an expert in reproducing research code from a paper.\n\n"
    "Here is the paper that you need to use to complete the code:\n"
)
_HDR_WITHOUT_PAPER = "\nYou are an expert in completing research code.\n\n"
_CODE_PRE = "\n\n\nHere is the code that you need to complete:\n"
_CODE_POST = '''


Please implement the missing code in the TODO blocks. Follow these guidelines carefully:

1. ONLY provide the implementation code that replaces the TODO comments
2. Your implementation must preserve the EXACT indentation level of the TODO block you are replacing
3. Do not include the function/class definitions or any surrounding code
4. Ensure your implementation is complete, functional, and follows best practices
5. If a corresponding paper is given, use it as a reference for reproducing the code
6. ALWAYS wrap your implementation in ```python and ``` markers

For example, if you see this nested TODO block:
```python
class Calculator:
    def calculate_area(self, radius):
        if radius > 0:
            # TODO: Implement block "calculate area"
            # Approximately 2 line(s) of code.
            pass
```

Your answer should preserve the EXACT indentation (12 spaces/3 levels) and be wrapped in code block markers like this:
```python
            area = radius * radius * math.pi
            return area
```

Notice how:
1. The implementation maintains the same indentation level as the TODO comment it replaces
2. The code is wrapped in ```python at the start and ``` at the end
'''


def _decode_text(payload: bytes, *, member: str) -> str:
    """Decode a provider text member like ``Path.read_text()`` did historically."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildContractError(
            f"researchcodebench: archive member is not UTF-8: {member}"
        ) from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_snippets(lines: list[str]) -> list[dict[str, object]]:
    """Mirror upstream ``File.parse_file`` for every named snippet region."""

    snippets: list[dict[str, object]] = []
    for start, line in enumerate(lines):
        opening_match = _OPEN.match(line)
        if opening_match is None:
            continue
        name = opening_match.group(1)
        body: list[str] = []
        end: int | None = None
        for candidate_end in range(start + 1, len(lines)):
            closing_match = _CLOSE.match(lines[candidate_end])
            if closing_match is not None and closing_match.group(1) == name:
                end = candidate_end
                break
            body.append(lines[candidate_end])
        if end is not None:
            snippets.append(
                {"name": name, "start": start, "end": end, "body": body}
            )
    return snippets


def _code_lines(body: list[str]) -> list[str]:
    """Mirror upstream ``Code.get_code_lines`` for TODO line estimates."""

    code_lines: list[str] = []
    for line in body:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" not in line or line.split("#", 1)[0].strip():
            code_lines.append(line)
    return code_lines


def _minimum_indent(body: list[str]) -> str:
    """Mirror upstream ``Snippet.find_minimum_indentation``."""

    levels: list[int] = []
    for line in body:
        if not line.strip():
            continue
        leading_whitespace = 0
        for character in line:
            if character not in (" ", "\t"):
                break
            leading_whitespace += 1
        levels.append(leading_whitespace)
    return " " * min(levels) if levels else ""


def _strip_markers(lines: list[str]) -> list[str]:
    """Remove every paper2code marker while retaining all other lines."""

    return [line for line in lines if not (_OPEN.match(line) or _CLOSE.match(line))]


def _mask_file(file_lines: list[str], snippet: dict[str, object]) -> str:
    """Mirror upstream ``File.build_replaced_code`` for one target snippet."""

    body = snippet["body"]
    if not isinstance(body, list) or not all(isinstance(line, str) for line in body):
        raise BuildContractError("researchcodebench: malformed parsed snippet body")
    name = str(snippet["name"])
    start = int(snippet["start"])
    end = int(snippet["end"])
    indentation = _minimum_indent(body)
    placeholder = [
        f'{indentation}# TODO: Implement block "{name}"\n',
        f"{indentation}# Approximately {len(_code_lines(body))} line(s) of code.\n",
    ]
    replaced = _strip_markers(file_lines[:start] + placeholder + file_lines[end + 1 :])
    return "\n".join(replaced)


def _reference_code(body: list[str]) -> str:
    """Return the real implementation inside a snippet, minus nested markers."""

    return "\n".join(_strip_markers(body))


LAYOUT = {'root': 'ResearchCodeBench-2758001c2ff84fc25c546339d65479ed058b0265',
 'pset_prefix': 'pset',
 'results_member': 'outputs/20llms_greedy/2025-05-12-17-13-20/overall_stats.json'}
EXPECTED_RELEASE = {'papers': 20,
 'items': 212,
 'matched_items': 212,
 'selected_pset_files': 113,
 'raw_subject_configurations': 32,
 'canonical_subjects': 31,
 'responses': 6784,
 'traces': 0,
 'response_counts': {0: 4333, 1: 2451},
 'trial_counts': {1: 6572, 2: 212}}


class ResearchCodeBenchBuild(BenchmarkBuild):
    """Translate the released greedy result matrix without changing its cells."""

    def _load_release_archive(
        self,
    ) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
        """Load the released statistics and selected prompt sources from the pin."""

        # Reproduction captures may add dependency archives beside the release.
        # Select the task/result archive explicitly rather than the first tarball.
        archive_path = self.raw_dir / "researchcodebench-2758001c2ff84fc25c546339d65479ed058b0265.tar.gz"
        archive_root = str(LAYOUT["root"])
        results_member = str(LAYOUT["results_member"])
        pset_prefix = str(LAYOUT["pset_prefix"]).rstrip("/")

        try:
            archive = tarfile.open(archive_path, "r:gz")  # noqa: SIM115
        except (OSError, tarfile.TarError) as exc:
            raise BuildContractError(
                f"researchcodebench: cannot read provider archive {archive_path}"
            ) from exc

        with archive:
            members = archive.getmembers()
            full_results_member = f"{archive_root}/{results_member}"
            try:
                stats_archive_member = archive.getmember(full_results_member)
            except KeyError as exc:
                raise BuildContractError(
                    "researchcodebench: pinned archive has no released overall_stats.json"
                ) from exc
            extracted_stats = archive.extractfile(stats_archive_member)
            if extracted_stats is None:
                raise BuildContractError(
                    "researchcodebench: could not read released overall_stats.json"
                )
            try:
                stats = json.loads(
                    _decode_text(extracted_stats.read(), member=results_member)
                )
            except json.JSONDecodeError as exc:
                raise BuildContractError(
                    "researchcodebench: released overall_stats.json is invalid JSON"
                ) from exc
            if getattr(self, "_local_source", False):
                stats = json.loads((self.raw_dir / "overall_stats.json").read_text())
            if not isinstance(stats, dict) or not isinstance(stats.get("results"), dict):
                raise BuildContractError(
                    "researchcodebench: overall_stats.json has no results mapping"
                )

            released_papers = set(stats["results"])
            source_files: dict[str, dict[str, str]] = {
                paper: {} for paper in stats["results"]
            }
            for member in members:
                if not member.isfile():
                    continue
                member_parts = PurePosixPath(member.name).parts
                if not member_parts or member_parts[0] != archive_root:
                    raise BuildContractError(
                        f"researchcodebench: unexpected archive member {member.name}"
                    )
                relative = PurePosixPath(*member_parts[1:])
                relative_parts = relative.parts
                if (
                    len(relative_parts) < 3
                    or relative_parts[0] != pset_prefix
                    or relative_parts[1] not in released_papers
                ):
                    continue

                relative_text = relative.as_posix()
                keep = (
                    relative_text.endswith(".py")
                    and not relative_text.endswith("paper2code_test.py")
                ) or relative_text.endswith(
                    ("paper2code_paper.tex", "paper2code.yaml")
                )
                if not keep:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise BuildContractError(
                        f"researchcodebench: could not read {member.name}"
                    )
                paper = relative_parts[1]
                paper_relative_path = PurePosixPath(*relative_parts[2:]).as_posix()
                source_files[paper][paper_relative_path] = _decode_text(
                    extracted.read(), member=relative_text
                )

        selected_file_count = sum(len(files) for files in source_files.values())
        expected_file_count = int(EXPECTED_RELEASE["selected_pset_files"])
        if not getattr(self, "_local_source", False) and selected_file_count != expected_file_count:
            raise BuildContractError(
                "researchcodebench: selected pset source count "
                f"{selected_file_count} != expected {expected_file_count}"
            )
        return stats, source_files

    @staticmethod
    def _snippet_index(
        papers: list[str],
        source_files: dict[str, dict[str, str]],
    ) -> dict[tuple[str, str], dict[str, object]]:
        """Index the longest reference occurrence for each paper/snippet key."""

        index: dict[tuple[str, str], dict[str, object]] = {}
        for paper in papers:
            for relative_path, source_text in sorted(source_files[paper].items()):
                if not relative_path.endswith(".py"):
                    continue
                file_lines = source_text.splitlines()
                for snippet in _parse_snippets(file_lines):
                    body = snippet["body"]
                    if not isinstance(body, list):
                        continue
                    reference = _reference_code(body)
                    if not reference.strip():
                        continue
                    key = (paper, str(snippet["name"]))
                    if key not in index or len(reference) > len(str(index[key]["ref"])):
                        index[key] = {
                            "lines": file_lines,
                            "snippet": snippet,
                            "ref": reference,
                        }
        return index

    @staticmethod
    def _context_files_text(
        paper: str,
        source_files: dict[str, dict[str, str]],
    ) -> str:
        """Flatten context files exactly as the released prompt builder did."""

        paper_files = source_files[paper]
        configuration = paper_files.get("paper2code.yaml")
        if configuration is None:
            return ""

        paths: list[str] = []
        in_context_paths = False
        for raw_line in configuration.splitlines():
            line = raw_line.rstrip()
            if line.startswith("context_file_paths:"):
                inline_value = line.split(":", 1)[1].strip()
                if inline_value and inline_value.lower() != "null":
                    paths.append(inline_value)
                in_context_paths = True
                continue
            if in_context_paths:
                list_match = re.match(r"\s*-\s*(.+?)\s*$", line)
                if list_match is not None:
                    paths.append(list_match.group(1))
                elif line and not line.startswith(" "):
                    in_context_paths = False

        flattened: list[str] = []
        for relative_path in paths:
            body = paper_files.get(relative_path)
            if body is not None:
                flattened.append(f"## {relative_path}\n\n{body}\n\n")
        return "".join(flattened)

    def build_subject_item_response_rows(self) -> None:
        stats, source_files = self._load_release_archive()
        results = stats["results"]
        if not isinstance(results, dict):
            raise BuildContractError(
                "researchcodebench: released results must be a mapping"
            )
        if not getattr(self, "_local_source", False) and len(results) != int(EXPECTED_RELEASE["papers"]):
            raise BuildContractError(
                "researchcodebench: released paper count violates expectations"
            )

        snippet_index = self._snippet_index(list(results), source_files)
        paper_text = {
            paper: source_files[paper].get("paper2code_paper.tex")
            for paper in results
        }
        context_text = {
            paper: self._context_files_text(paper, source_files)
            for paper in results
        }

        item_ids: dict[tuple[str, str], str] = {}
        subject_ids: dict[str, str] = {}
        trial_by_subject_item: Counter[tuple[str, str]] = Counter()
        response_counts: Counter[int] = Counter()
        trial_counts: Counter[int] = Counter()
        raw_subject_labels: set[str] = set()
        matched_items = 0
        total_items = 0
        response_count = 0

        for paper, paper_data in results.items():
            if not isinstance(paper_data, dict) or not isinstance(
                paper_data.get("results"), dict
            ):
                raise BuildContractError(
                    f"researchcodebench: malformed paper results for {paper}"
                )
            for model, model_data in paper_data["results"].items():
                raw_subject_labels.add(model)
                if model not in subject_ids:
                    subject_ids[model] = self.add_subject(model)
                subject_id = subject_ids[model]
                if not isinstance(model_data, dict) or not isinstance(
                    model_data.get("results"), dict
                ):
                    raise BuildContractError(
                        f"researchcodebench: malformed model results for {model}"
                    )

                for snippet_name, completions in model_data["results"].items():
                    key = (paper, snippet_name)
                    entry = snippet_index.get(key)
                    reference = str(entry["ref"]) if entry is not None else None
                    if key not in item_ids:
                        total_items += 1
                        if entry is not None:
                            matched_items += 1
                            masked_code = context_text[paper] + _mask_file(
                                entry["lines"], entry["snippet"]
                            )
                            paper_source = paper_text[paper]
                            header = (
                                _HDR_WITH_PAPER_PRE + paper_source + "\n\n"
                                if paper_source is not None
                                else _HDR_WITHOUT_PAPER
                            )
                            content = header + _CODE_PRE + masked_code + _CODE_POST
                        else:
                            # Preserve the legacy fallback, while the release
                            # expectation below makes it an error for this pin.
                            content = (
                                f"ResearchCodeBench task {paper}::{snippet_name}: "
                                "implement the missing code snippet."
                            )
                        item_ids[key] = self.add_item(
                            raw_item_id=f"{paper}::{snippet_name}",
                            content=content,
                            grading_criterion={
                                "reference_answer": reference,
                                "rule": "Assign 1 when the provider's paper-specific tests pass, otherwise 0.",
                            },
                            verifier=ExactMatcher(spec=(
                                "Provider's pinned greedy-run harness: parse the last Python Markdown "
                                "block, insert it into the masked source file, and run the "
                                "paper-specific Python test program with a 60-second timeout. "
                                "Import the released passed flag without rerunning the tests."
                            )),
                            features={"paper": paper},
                        )
                    item_id = item_ids[key]

                    if not isinstance(completions, list):
                        raise BuildContractError(
                            f"researchcodebench: completions are not a list for {key}"
                        )
                    for completion in completions:
                        if not isinstance(completion, dict):
                            raise BuildContractError(
                                f"researchcodebench: malformed completion for {key}"
                            )
                        cell = (subject_id, item_id)
                        trial_by_subject_item[cell] += 1
                        trial = trial_by_subject_item[cell]
                        response = 1.0 if completion.get("passed") else 0.0
                        self.add_response(
                            subject_id=subject_id,
                            item_id=item_id,
                            trial=trial,
                            test_condition=None,
                            interactors=None,
                            response=response,
                            trace=completion.get("completion"),
                        )
                        response_count += 1
                        response_counts[int(response)] += 1
                        trial_counts[trial] += 1

        expected_response_counts = {
            int(value): int(count)
            for value, count in EXPECTED_RELEASE["response_counts"].items()
        }
        expected_trial_counts = {
            int(trial): int(count)
            for trial, count in EXPECTED_RELEASE["trial_counts"].items()
        }
        observed = {
            "items": total_items,
            "matched_items": matched_items,
            "raw_subject_configurations": len(raw_subject_labels),
            "canonical_subjects": len(set(subject_ids.values())),
            "responses": response_count,
        }
        expected = {key: int(EXPECTED_RELEASE[key]) for key in observed}
        if total_items != matched_items:
            raise BuildContractError("researchcodebench: missing full snippet content")
        if not getattr(self, "_local_source", False) and observed != expected:
            raise BuildContractError(
                f"researchcodebench: release shape {observed} != expectations {expected}"
            )
        if not getattr(self, "_local_source", False) and dict(response_counts) != expected_response_counts:
            raise BuildContractError(
                "researchcodebench: binary response counts violate expectations"
            )
        if not getattr(self, "_local_source", False) and dict(trial_counts) != expected_trial_counts:
            raise BuildContractError(
                "researchcodebench: trial assignment violates expectations"
            )

        print(f"[researchcodebench] snippet content matched {matched_items}/{total_items}")


if __name__ == "__main__":
    ResearchCodeBenchBuild(__file__).main_from_args()
