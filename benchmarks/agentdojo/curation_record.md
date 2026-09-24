# AgentDojo curation record

- **Coverage.** Includes 69,798 utility/attacker-success observations and 69,384 linked full traces from the pinned upstream repository. One interrupted run contributes two null grades; its attempt and messages are retained. Utility=1 means task success; attacker-success=1 means the attacker achieved its goal, including the provider's exception policy.

- **Curation decisions.** Use distinct grading identities and score directions for utility and attacker success. Defense belongs to the subject and attack type to the interactor. Resolve supported literal task expressions, and preserve full JSON histories; removing the old character cap restored 7,332 clipped traces without changing outcomes.

- **Limitations.** Positive grades across the two measures are not a combined agent-performance score. Historical runs do not identify exact task/evaluator versions; captured task declarations can come from later suite versions. Unreleased inference settings remain unknown.

[Metadata and sources](metadata.yaml) · [Characterization](characterization.yaml) · [Checks](../../tests/test_benchmark_datasets.py) · [Builder](build.py)
