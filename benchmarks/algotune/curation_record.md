# AlgoTune curation record

- **Coverage:** 2,772 published outcomes across 154 tasks and 18 model labels; 2,759 conversation traces contain all 276,526 messages and 3,003 named final source files.
- **Transformation:** Four table stages preserve complete task text, grading code, reported speedups and message order. The existing binary rule remains `final_speedup >= 1`; explicit upstream `N/A` results remain failures.
- **Verification:** All original grades match. Independent source checks cover every outcome, message and final file; five shared dataset tests and ten deliberate corruption checks pass. Source pages and repository inputs are checksum-pinned and downloadable from upstream.
- **Limits:** The public summary omits per-input validity counts and specific failure causes. Thirteen pages have no conversation. No models or submitted programs were executed during curation.

[Sources](metadata.yaml) · [Characterization](characterization.yaml) · [Builder](build.py)
