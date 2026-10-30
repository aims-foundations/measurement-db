# OpenBioRQ curation record

- **Coverage.** The pinned author release contains 657 questions and 7,227 attempts from eleven model/tool configurations. Fourteen ungraded attempts remain null; all 7,213 previously graded outcomes are retained.
- **Grading and traces.** Each item carries its published rubric and reference grounding. Independent checks recompute weighted checklist scores, verify the 0.5 success threshold and compare every full prediction record, including tool messages.
- **Scope.** Tool-enabled and no-tool runs are distinct subjects. Results follow the released GLM-5.1 judgments; the author documents differences from the paper's original frontier-model judgments. Private infrastructure details were scrubbed by the author.

[Metadata](metadata.yaml) · [Characterization](characterization.yaml) · [Shared checks](../../tests/test_benchmark_datasets.py)
- Testlet migration (2026-10-29): 7,227 recorded attempts, 53,691 component or retained aggregate observations, and 4,881 canonical items. Complete original question, weighted-rubric, configuration and full-answer reconciliation preserves 7,227 executions. All 14 ungraded checklists now carry explicit unknown criterion observations; 53,691 observations retain native 0, partial-credit and full-credit values across 4,881 items in 657 testlets. Characterization records the independently checked counts and complete table fingerprints. Validated locally against the reviewed source and table contracts.
