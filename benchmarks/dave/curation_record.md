# DAVE curation record

- **Coverage.** Imports the released multimodal responses of eight end-to-end models on Epic Kitchens and Ego4D. Preserves complete recorded prompts, structured response payloads, provider binary grades, and original video/audio bytes. The old builder overlooked these item-level results in the official GitHub repository.
- **Matching.** Matches task type, sound, unordered choice text, and correct option to the pinned bank because result files omit clip IDs, shuffle choices, and sometimes reorder rows. Twenty ambiguous bank entries are excluded across all eight models, independently of correctness. A separate audit checks every observation and requires each run to cover every unambiguous stimulus exactly once.
- **Scope.** Pipeline models, random baselines, auxiliary tasks, and modality ablations are captured in raw inputs but are outside this import. Recorded model names are retained; unreleased checkpoint or inference settings are not inferred.

Sources and grading are in [metadata.yaml](metadata.yaml); counts and source checks are in [characterization.yaml](characterization.yaml) and the [shared dataset checks](../../tests/test_benchmark_datasets.py).
