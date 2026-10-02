"""Convert C-SEO's released citation comparisons into item-level measurements."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CSEOBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read native Parquet tables; directory names identify each experimental condition.
        source = self.raw_dir / "release/results"
        paths = sorted(source.rglob("responses.parquet"))
        # Two condition names reuse the same recorded run. Verify each declared alias
        # before excluding it, so a changed source cannot silently lose observations.
        for alias, canonical in self.build_parameters["result_aliases"].items():
            if source / alias in paths and source / canonical in paths:
                if (source / alias).read_bytes() != (source / canonical).read_bytes():
                    raise ValueError("C-SEO result alias differs from its declared original")
                paths.remove(source / alias)
        runs = pd.concat([
            pd.read_parquet(path).rename(columns=self.build_parameters["columns"]).assign(
                source_file=str(path.relative_to(source)), source_row=lambda frame: frame.index,
            ) for path in paths
        ], ignore_index=True)
        paths = runs.source_file.str.split("/", n=3, expand=True)
        runs = runs.assign(
            domain=paths[0], method=paths[1], subject_key=paths[2],
            adoption=paths[3].str.removesuffix("/responses.parquet").replace("responses.parquet", "unspecified"),
            run_key=runs.index,
        )
        if runs[["prompt", "query"]].isna().any().any():
            raise ValueError("C-SEO has an observation without its input")
        # An older video-game export appends this suffix; the actual prompts remain unchanged.
        runs["query_key"] = runs["query"].where(
            runs.domain.ne("videogames"), runs["query"].str.removesuffix(self.build_parameters["pairing"]["query_suffix"]),
        )
        if runs.duplicated(["source_file", "query_key"]).any():
            raise ValueError("C-SEO query normalization creates an ambiguous join")

        # 2. Match each intervention to the same model/domain/query's Original control.
        baseline = runs.loc[runs.method.eq("Original") & runs.adoption.eq("AdoptionMode.NONE")]
        comparisons = runs.loc[runs.method.ne("Original")].merge(
            baseline[["domain", "subject_key", "query_key", "run_key", "prompt", "output", "source_file"]],
            on=["domain", "subject_key", "query_key"], how="left", validate="many_to_one", suffixes=("", "_baseline"),
        )
        # Early releases store a single target as an integer; later ones use lists.
        comparisons["targets"] = comparisons.targets.map(lambda targets: [targets] if pd.api.types.is_scalar(targets) else targets)
        if comparisons.run_key_baseline.isna().any() or not comparisons.targets.map(len).gt(0).all():
            raise ValueError("C-SEO comparison lacks a baseline or target document")
        # One measurement per promoted document, including multi-adopter experiments.
        observations = comparisons.explode("targets", ignore_index=True).rename(columns={"targets": "target_before"})
        observations["target_after"] = observations.target_before
        moved = observations.method.str.startswith("seo_baseline-")
        observations.loc[moved, "target_after"] = observations.loc[moved, "method"].str.rsplit("-", n=1).str[-1].astype(int) - 1
        group_moved = observations.method.eq("seo_baseline_game_theory")
        observations.loc[group_moved, "target_after"] = observations.loc[group_moved].groupby("run_key", sort=False).cumcount()

        # 3. Join target documents to their positions among the first five distinct citations.
        # Keep the native citation lists, including parser errors and out-of-range citations.
        cutoff = self.grading["verifiers"]["rank_difference"]["max_citations"]
        cited = runs[["run_key", "citations"]].assign(citations=runs.citations.str[:cutoff])
        counts = cited.assign(citation_count=cited.citations.map(len))[["run_key", "citation_count"]]
        ranks = cited.explode("citations").dropna(subset=["citations"])
        ranks["rank"] = ranks.groupby("run_key", sort=False).cumcount()
        positions = observations[["run_key", "run_key_baseline", "target_before", "target_after"]].merge(
            ranks.rename(columns={"run_key": "run_key_baseline", "citations": "target_before", "rank": "rank_before"}),
            on=["run_key_baseline", "target_before"], how="left", validate="many_to_one",
        ).merge(
            ranks.rename(columns={"citations": "target_after", "rank": "rank_after"}),
            on=["run_key", "target_after"], how="left", validate="many_to_one",
        ).merge(
            counts.rename(columns={"run_key": "run_key_baseline", "citation_count": "count_before"}),
            on="run_key_baseline", how="left", validate="many_to_one",
        ).merge(counts, on="run_key", how="left", validate="many_to_one")
        # Match the author notebook: absent targets use that output's list length;
        # targets absent from both lists receive zero even when list lengths differ.
        grades = positions.rank_before.fillna(positions.count_before) - positions.rank_after.fillna(positions.citation_count)
        grades = grades.where(positions[["rank_before", "rank_after"]].notna().any(axis=1), 0.)
        grades = grades.where(observations[["output", "output_baseline"]].notna().all(axis=1))

        # 4. Items contain both input contexts and the target indices; outputs stay in traces.
        content = observations[["prompt_baseline", "prompt", "target_before", "target_after"]].rename(columns={
            "prompt_baseline": "baseline_prompt", "prompt": "modified_prompt",
        })
        content[["target_before", "target_after"]] = content[["target_before", "target_after"]].astype(int)
        items = pd.DataFrame({"item_key": observations.index, "raw_item_id": None})
        items["content"] = content.to_json(orient="records", lines=True, force_ascii=False).split("\n")[:-1]
        items["features"] = observations[["domain", "method", "adoption"]].to_dict("records")
        items["grading_criterion"] = [{"rule": self.grading["rule"]}] * len(items)
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["rank_difference"], sort_keys=True))
        subjects = runs[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        subjects["features"] = [self.build_parameters["subject_features"]] * len(subjects)

        # 5. Preserve complete paired outputs and their source locations, without truncation.
        trace_data = observations[["output_baseline", "output", "source_file_baseline", "source_file", "source_row"]].astype(object)
        trace_data = trace_data.where(trace_data.notna(), None)
        traces = pd.DataFrame({"response_key": observations.index})
        traces["trace"] = trace_data.to_json(orient="records", lines=True, force_ascii=False).split("\n")[:-1]

        # 6. Hand local keys to the shared writer, which resolves identities and repeated trials.
        responses = pd.DataFrame({
            "response_key": observations.index, "subject_key": observations.subject_key,
            "item_key": observations.index, "response": grades,
        })
        return {"subjects": subjects, "items": items, "responses": responses, "traces": traces}


if __name__ == "__main__":
    CSEOBench(__file__).main_from_args()
