# SWE-rebench curation record

- **Coverage.** Includes released, non-null `resolved` outcomes for Qwen3-Coder-480B-A35B-Instruct with OpenHands v0.54.0. Repeated trajectories remain separate trials; nonempty final patches supply traces.
- **Curation decisions.** Attempted tasks join to their problem statements, reference patches, test definitions, and environment descriptions. Including grading protocols in item identity separates 35 previously merged items and changes some trial numbers while preserving every retained outcome and trace. The subsequent table-based refactor preserved all five reviewed tables exactly.
- **Limitations.** Captured item and trajectory inputs contain selected columns rather than complete upstream files, and their original capture revisions were not recorded. Missing or blank final patches remain missing traces; final patches do not represent complete agent conversations.

Sources and grading are documented in [metadata.yaml](metadata.yaml); reviewed counts and checks are in [characterization.yaml](characterization.yaml) and [shared dataset checks](../../tests/test_benchmark_datasets.py).
