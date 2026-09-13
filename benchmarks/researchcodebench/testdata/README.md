# ResearchCodeBench characterization data

`characterization.json` records reviewed expectations for the generated
ResearchCodeBench tables. It is regression-test data, not an upstream benchmark
artifact, and `build.py` must never read it.

The snapshot covers release shape, response and trial structure, the preserved
Gemini-preview identity collision, the six legacy GPS prompts without paper
text, and logical table fingerprints. Provider-reported counts and task-rate
aggregates live separately in `../metadata.yaml`; `../test.py` reconciles those
claims directly against the immutable archive.

Update this file only after inspecting a successful build and obtaining review
for every semantic change. In particular, do not refresh hashes merely because
a test fails. The 2026-09-07 migration accepted only the repository-wide
response-ID hashing change; prompts, references, values, trials, item IDs, and
subject IDs were checked against the reproduced legacy baseline.

Run the complete characterization with:

```bash
python benchmarks/researchcodebench/build.py
MEASUREMENT_DB_FULL_TEST=1 python benchmarks/researchcodebench/test.py -v
```
