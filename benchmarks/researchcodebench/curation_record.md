# ResearchCodeBench curation record

- **Coverage.** Includes the provider's released greedy-run matrix for 212 code snippets. Each response retains the released test-pass flag. Later Harbor, OpenReward, and REVERE evaluations and modified forks are separate measurements and are not included.
- **Curation decisions.** Reconstruct prompts and reference code from captured annotated files, paper text, and declared context. The table-based refactor preserved the reviewed items, subjects, grades, IDs, and trials. Duplicate snippet definitions retain the longest reference implementation, with ties resolved by file and line order.
- **Limitations.** Six GPS prompts omit the Markdown paper that the original evaluated models received. Two Gemini preview configurations remain merged into one canonical subject as trials 1 and 2; these are different configurations, not independent repetitions. Released completion traces and complete execution settings are unavailable, and the benchmark-wide license is recorded as unknown.

Sources and grading are documented in [metadata.yaml](metadata.yaml); reviewed counts and checks are in [characterization.yaml](characterization.yaml) and [shared dataset checks](../../tests/test_benchmark_datasets.py).
