# ConfAgents curation record

- **Coverage.** Includes 64 released answer judgments and reasoning traces from four agent frameworks on 16 medical questions.

- **Curation decisions.** Use the released answer-option comparison for grading and retain each framework as a separate subject. Source dataset is an item attribute. Observations and their reasoning traces are preserved through the identity migration.

- **Limitations.** The capture is a small question subset, not the full medical evaluation. Original capture revision and backend model configurations are unknown; framework names are not evidence of a particular underlying model.

[Metadata and sources](metadata.yaml) · [Characterization](characterization.yaml) · [Checks](../../tests/test_benchmark_datasets.py) · [Builder](build.py)
