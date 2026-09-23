# Multi-SWE-bench curation record

- **Coverage.** Includes captured experiment outcomes, task definitions, and available prediction patches. Resolved/unresolved membership supplies binary grades; patches supply traces. Repeated runs remain separate observations, with harness configuration recorded on the subject.
- **Curation decisions.** Items retain full prompts, reference patches, test patches, and test lists. Removing earlier truncation and distinguishing grading protocols changed item and response IDs while preserving observation–grade–trace associations. Ambiguous task definitions or duplicate run/task records raise errors rather than silently selecting a record.
- **Limitations.** One captured prediction file (`java__20250426_MopenHands_Gemini-2.5-Pro.jsonl`) contains only a Git LFS pointer: its grades remain, but its traces are unavailable. Original upstream revisions were not recorded for the historical input capture; the saved bytes define that snapshot.

Sources and grading are documented in [metadata.yaml](metadata.yaml); reviewed counts and checks are in [characterization.yaml](characterization.yaml) and [shared dataset checks](../../tests/test_benchmark_datasets.py).
