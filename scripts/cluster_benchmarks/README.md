# benchmark_clustering

Groups curated benchmarks by domain and generates the canonical domain list.

## Files

| File | |
| --- | --- |
| `cluster_benchmarks.py` | the tool |
| `domain_taxonomy.yaml` | **edit this** — canonical domains, aliases, retired values, signals |
| `benchmarks.csv` | generated snapshot of all 210 benchmarks; refresh with `add --sync` |

Needs `PyYAML`, already in the repo's `requirements.txt`.

## Commands

Every command takes `--csv benchmarks.csv` (the snapshot) or `--root benchmarks`
(the repo's `metadata.yaml` files). They give identical results.

```bash
# the grouped listing
python cluster_benchmarks.py report --csv benchmarks.csv

# the canonical domain list, with usage counts, as paste-ready Python
python cluster_benchmarks.py vocab --csv benchmarks.csv

# CI gate — exits 1 on unknown or retired domain values
python cluster_benchmarks.py check --csv benchmarks.csv
python cluster_benchmarks.py check --csv benchmarks.csv --require-domain

# pull benchmarks into the CSV from the repo
python cluster_benchmarks.py add --slug taubench --root benchmarks
python cluster_benchmarks.py add --slug "harmbench,taubench" --root benchmarks
python cluster_benchmarks.py add --sync --root benchmarks      # everything missing

# propose a domain for benchmarks that declare none
python cluster_benchmarks.py suggest --csv benchmarks.csv
python cluster_benchmarks.py suggest --csv benchmarks.csv --slug taubench

# declared-vs-suggested disagreements, as a review queue
python cluster_benchmarks.py audit --csv benchmarks.csv

# how well the suggester does — run before and after editing signals
python cluster_benchmarks.py eval --csv benchmarks.csv

# rewrite the CSV from the repo
python cluster_benchmarks.py export --root benchmarks --out benchmarks.csv
```

## Adding a benchmark

```bash
python cluster_benchmarks.py add --slug my_new_bench --root benchmarks
```

Reads `benchmarks/my_new_bench/metadata.yaml` and fills in the row. If it
declares no domain, `add` prints suggestions for you to choose from; write your
choice into the CSV's `domain` column, then:

```bash
python cluster_benchmarks.py check --csv benchmarks.csv --require-domain
```

Rows already present are skipped unless you pass `--update`.

## Editing `domain_taxonomy.yaml`

This is the only file you edit by hand.

```yaml
canonical:      # the vocabulary — what `vocab` emits
  - agents_and_tool_use
aliases:        # drifted spelling -> canonical, applied automatically
  tool_use: agents_and_tool_use
retired:        # not domains at all; the reason shows in `check` output
  computer_vision: "a modality, not a domain - recorded in benchmark.modality"
signals:        # regex matched against name + description + tags, for `suggest`
  agents_and_tool_use:
    - \b(agent|tool call|function call|scaffold|harness|browser)\b
```

After editing, run `eval` to confirm the suggester did not get worse, then
`report` and `vocab`.

## A benchmark that fits no domain

There is no catch-all bucket on purpose — a `misc` domain accumulates
benchmarks nobody revisits.

| Situation | What happens |
| --- | --- |
| Value outside the vocabulary | No domain assigned; ERROR; `check` exits 1 |
| Retired value | Same, with the reason shown |
| Declares nothing | Warning; `suggest` offers candidates |
| Declares nothing, no signal matches | `(none - needs a human)`. No guess |

Each routes to one of two edits: add an **alias** if it is a variant of an
existing domain, or add it to **canonical** with signals if it is genuinely new.
Don't create a domain for a single benchmark — use the nearest parent and
revisit when three accumulate.
