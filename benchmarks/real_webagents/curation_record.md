# REAL curation record

- **Coverage.** Includes verified-model API attempts with recorded accuracy and the available task definitions. A response is successful when `evalsFailed` is empty; when that field is absent, `accuracy >= 100` supplies the historical fallback.
- **Curation decisions.** Task rules are joined to attempts by task ID; website is an item attribute. Answer text is preserved except for empty values and the placeholders `Done` and `No response`. The table-based refactor preserved all five output tables, including IDs, grades, trials, and traces.
- **Limitations.** The captured source contains definitions for 112 tasks but none for 121 `v2.` tasks. For those tasks, grading metadata describes the provider-verdict mapping; the original checkers and rubrics remain unavailable. Historical source revisions were not recorded.

Sources and grading are documented in [metadata.yaml](metadata.yaml); reviewed counts and checks are in [characterization.yaml](characterization.yaml) and [shared dataset checks](../../tests/test_benchmark_datasets.py).
