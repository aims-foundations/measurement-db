"""Prepare ProgramBench's explicitly identified tests for a continuous matrix preview.

This is a display index, not an item migration. Full native test identifiers are
scoped by testlet; every variant still points to its original canonical item.
Only binary, single-attempt subject/testlet data is supported by this experiment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.render_website.record_packs import pack_records

PAGE_SIZE = 256


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(tables: Path, output: Path, source_revision: str) -> dict:
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    items_file = pq.ParquetFile(tables / "items.parquet", read_dictionary=["content"])
    items = items_file.read(columns=["item_id", "raw_item_id", "testlet_id", "content_hash", "verifier"]).to_pandas()
    if items.testlet_id.isna().any() or items.item_id.duplicated().any():
        raise ValueError("Preview requires explicit testlets and unique canonical items.")
    items["grader"] = items.verifier.map(lambda v: json.loads(json.loads(v)["spec"])["grader_version"])
    items.drop(columns="verifier", inplace=True)
    # ProgramBench's builder records task_id + ':' + the complete native test ID.
    if not all(raw.startswith(task + ":") for raw, task in zip(items.raw_item_id, items.testlet_id)):
        raise ValueError("Unexpected ProgramBench native test identity.")
    items["test"] = [raw[len(task) + 1:] for raw, task in zip(items.raw_item_id, items.testlet_id)]

    # Read each distinct stimulus once, retaining Arrow's dictionary encoding.
    first = items.drop_duplicates("content_hash").index.to_numpy()
    stimuli, offset = {}, 0
    for group in range(items_file.num_row_groups):
        size = items_file.metadata.row_group(group).num_rows
        selected = first[(first >= offset) & (first < offset + size)]
        if len(selected):
            column = items_file.read_row_group(group, columns=["content"]).column(0)
            for position in selected:
                content = json.loads(column[int(position - offset)].as_py())
                prompt = json.dumps(content["messages"], sort_keys=True, ensure_ascii=False)
                stimuli[items.at[position, "content_hash"]] = {
                    "promptHash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    "image": content["environment"]["image"],
                    "cwd": content["environment"]["cwd"],
                }
        offset += size

    subjects = pq.read_table(tables / "subjects.parquet").to_pylist()
    subject_index = pd.Index([s["subject_id"] for s in subjects])
    attempts = pq.read_table(tables / "attempts.parquet", columns=["attempt_id", "subject_id", "raw_attempt_id"]).to_pandas()
    attempts["testlet"] = attempts.raw_attempt_id.str.split("/", n=1).str[1]
    if attempts.duplicated(["subject_id", "testlet"]).any():
        raise ValueError("Multiple attempts require separate display rows; refusing to average them.")
    attempt_lookup = {(r.subject_id, r.testlet): r.attempt_id for r in attempts.itertuples()}

    groups, column_count = [], 0
    item_column = np.empty(len(items), dtype=np.int32)
    item_setup = np.empty(len(items), dtype=np.int16)
    item_group = np.empty(len(items), dtype=np.int16)
    for group_number, (testlet, frame) in enumerate(items.groupby("testlet_id", sort=True)):
        tests = sorted(frame.test.unique())
        test_index = pd.Index(tests)
        positions = test_index.get_indexer(frame.test)
        setup_keys = list(dict.fromkeys(zip(frame.content_hash, frame.grader)))
        setups = [{"contentHash": content, "grader": grader, **stimuli[content]} for content, grader in setup_keys]
        setup_index = {key: n for n, key in enumerate(setup_keys)}
        variants = [[None] * len(setups) for _ in tests]
        for position, row in zip(positions, frame.itertuples()):
            setup = setup_index[row.content_hash, row.grader]
            if variants[position][setup] is not None:
                raise ValueError("More than one canonical item for the same test and setup.")
            variants[position][setup] = row.item_id
            item_setup[row.Index] = setup
        item_column[frame.index] = column_count + positions
        item_group[frame.index] = group_number
        for start in range(0, len(tests), PAGE_SIZE):
            write_json(staging / f"tests/{group_number}/{start // PAGE_SIZE}.json", {
                "start": start, "tests": [
                    {"name": tests[i], "items": variants[i]}
                    for i in range(start, min(start + PAGE_SIZE, len(tests)))
                ],
            })
        groups.append({"id": testlet, "start": column_count, "count": len(tests),
                       "setups": setups, "rowSetups": [-1] * len(subjects)})
        column_count += len(tests)

    scores = np.zeros((len(subjects), column_count), dtype=np.uint8)
    item_index = pd.Index(items.item_id)
    observed_items = np.zeros(len(items), dtype=bool)
    totals = {"observations": 0, "passed": 0, "failed": 0}
    for batch in pq.ParquetFile(tables / "responses.parquet").iter_batches(
        batch_size=131072, columns=["subject_id", "item_id", "attempt_id", "response"]
    ):
        records = batch.to_pandas()
        ip = item_index.get_indexer(records.item_id)
        rows = subject_index.get_indexer(records.subject_id)
        if (ip < 0).any() or (rows < 0).any() or not records.response.isin([0, 1]).all():
            raise ValueError("Unknown identity or non-binary observation.")
        columns = item_column[ip]
        cells = rows * column_count + columns
        if len(np.unique(cells)) != len(cells) or (scores[rows, columns] != 0).any():
            raise ValueError("Multiple observations in an aligned cell; refusing to overwrite.")
        # One setup and source attempt per subject/testlet, checked across batches.
        assignments = pd.DataFrame({"row": rows, "group": item_group[ip],
                                    "setup": item_setup[ip], "attempt": records.attempt_id}).drop_duplicates()
        for row, group, setup, attempt in assignments.itertuples(index=False, name=None):
            descriptor = groups[group]
            previous = descriptor["rowSetups"][row]
            if previous not in (-1, setup) or attempt != attempt_lookup[(subjects[row]["subject_id"], descriptor["id"])]:
                raise ValueError("Inconsistent setup or attempt within a model/testlet.")
            descriptor["rowSetups"][row] = int(setup)
        scores[rows, columns] = records.response.to_numpy(dtype=np.uint8) + 1
        observed_items[ip] = True
        totals["observations"] += len(records)
        totals["passed"] += int(records.response.sum())
        totals["failed"] += int((records.response == 0).sum())
    if not observed_items.all():
        raise ValueError("Unobserved canonical items need explicit handling in this preview.")
    if int(np.count_nonzero(scores)) != totals["observations"]:
        raise ValueError("Aligned counts differ from source observations.")

    transport = pack_records(staging)
    payload = gzip.compress(scores.tobytes(), compresslevel=6, mtime=0)
    (staging / "scores.bin.gz").write_bytes(payload)
    manifest = {"format": "aligned-test-preview-v1", "benchmark": "programbench",
                "sourceRevision": source_revision, "columns": column_count,
                "canonicalItems": len(items), "subjects": subjects, "groups": groups,
                "pageSize": PAGE_SIZE, "pack": transport, **totals,
                "scores": {"file": "scores.bin.gz", "bytes": len(payload),
                           "expandedBytes": scores.nbytes,
                           "sha256": hashlib.sha256(payload).hexdigest()},
                "sourceSha256": {name: digest(tables / name) for name in
                                 ("items.parquet", "subjects.parquet", "responses.parquet", "attempts.parquet")}}
    write_json(staging / "manifest.json", manifest)
    # Only this experiment's output is replaced; canonical tables remain untouched.
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    manifest = prepare(args.tables, args.output, args.source_revision)
    print(json.dumps({key: manifest[key] for key in
                      ("columns", "canonicalItems", "observations", "passed", "failed", "scores")}, indent=2))
