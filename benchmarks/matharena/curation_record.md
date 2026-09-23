# MathArena curation record

- **Coverage.** Includes released attempts from 27 provider output datasets. Final-answer tasks retain the provider's correctness verdict; proof tasks retain each usable judge–criterion score as points divided by maximum points, with the legacy clipping to [0, 1].
- **Curation decisions.** Preserve model/configuration distinctions, competition and grading scope, original attempt indices, full prompts and solution text, and exact image bytes. Corrections restored image questions and 3,377 previously omitted proof-criterion scores, changing affected identities. A proof solution trace is attached only to its first usable criterion.
- **Limitations.** Dataset releases can overlap, and coverage is incomplete. Criterion scores are not equally weighted whole-problem scores; anonymous judge slots do not identify people across competitions. Configuration paths do not establish unreleased inference settings or harness versions, which remain unknown.

Sources and grading are documented in [metadata.yaml](metadata.yaml); reviewed counts and checks are in [characterization.yaml](characterization.yaml) and [shared dataset checks](../../tests/test_benchmark_datasets.py).
