#!/usr/bin/env python3
"""Transform ResearchCodeBench's captured sources into measurement tables."""

import json
import sys
import tarfile
from pathlib import Path
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, BuildContractError, ExactMatcher


class ResearchCodeBenchBuild(BenchmarkBuild):

    def download(self):
        return self.fetch_sources('release')
    def build_tables(self) -> dict[str, pd.DataFrame]:
        """Reconstruct task prompts and flatten the released result matrix."""
        # 1. Read the result JSON and selected prompt files from the pinned archive.
        inputs = self.build_parameters
        layout, prompt = inputs["archive"], inputs["prompt"]
        archive_path = self.raw_dir / layout["file"]
        with tarfile.open(archive_path, "r:gz") as archive:
            if getattr(self, "_local_source", False):
                stats = json.loads((self.raw_dir / "overall_stats.json").read_text())
            else:
                member = layout["root"] + "/" + layout["results_member"]
                stats = json.load(archive.extractfile(member))
            if not isinstance(stats, dict) or not isinstance(stats.get("results"), dict):
                raise BuildContractError("researchcodebench: overall_stats.json has no results mapping")

            members = pd.DataFrame({"member": [entry.name for entry in archive.getmembers() if entry.isfile()]})
            prefix = layout["root"] + "/"
            if not members.member.str.startswith(prefix).all():
                raise BuildContractError("researchcodebench: unexpected archive root")
            members["path"] = members.member.str.removeprefix(prefix)
            paths = members.path.str.split("/", n=2, expand=True).reindex(columns=[0, 1, 2])
            members = members.assign(section=paths[0], paper=paths[1], path=paths[2])
            selected = members.section.eq(layout["pset_prefix"]) & members.paper.isin(stats["results"])
            is_code = members.path.str.endswith(".py") & ~members.path.str.endswith("paper2code_test.py")
            is_context = members.path.str.endswith(("paper2code_paper.tex", "paper2code.yaml"))
            files = members.loc[selected & (is_code | is_context)].copy()
            if files.duplicated(["paper", "path"]).any():
                raise BuildContractError("researchcodebench: duplicate prompt source files")
            files["payload"] = [archive.extractfile(name).read() for name in files.member]
        try:
            files["text"] = files.payload.str.decode("utf-8")
        except UnicodeDecodeError as exc:
            invalid = files.loc[files.payload.eq(exc.object), "member"].tolist()
            raise BuildContractError(f"researchcodebench: archive member is not UTF-8: {invalid}") from exc
        files["text"] = files.text.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)
        files = files.drop(columns="payload").sort_values(["paper", "path"]).reset_index(drop=True)
        files["file_key"] = files.index

        # 2. Flatten paper -> model -> snippet mappings into one row per result cell.
        papers = pd.DataFrame.from_dict(stats["results"], orient="index").rename_axis("paper").reset_index()
        if "results" not in papers or not papers.results.map(lambda value: isinstance(value, dict)).all():
            raise BuildContractError("researchcodebench: malformed paper results")
        model_rows = papers[["paper"]].assign(entry=papers.results.map(lambda value: list(value.items())))
        model_rows = model_rows.explode("entry", ignore_index=True).dropna(subset=["entry"])
        models = pd.DataFrame(model_rows.entry.tolist(), columns=["model", "details"], index=model_rows.index)
        models = models.join(model_rows[["paper"]]).reset_index(drop=True)
        model_data = pd.json_normalize(models.details.tolist(), max_level=0).reindex(columns=["results"])
        if not model_data.results.map(lambda value: isinstance(value, dict)).all():
            raise BuildContractError("researchcodebench: malformed model results")
        cells = models[["paper", "model"]].assign(entry=model_data.results.map(lambda value: list(value.items())))
        cells = cells.explode("entry", ignore_index=True).dropna(subset=["entry"])
        cell_data = pd.DataFrame(cells.entry.tolist(), columns=["snippet", "completions"], index=cells.index)
        cells = cells.drop(columns="entry").join(cell_data)
        if not cells.completions.map(lambda value: isinstance(value, list)).all():
            raise BuildContractError("researchcodebench: completions must be lists")

        # 3. Expand annotated Python files into lines and pair each opening marker
        # with the first matching closing marker later in the same file.
        code = files.loc[files.path.str.endswith(".py"), ["file_key", "paper", "text"]]
        lines = code.assign(line=code.text.map(str.splitlines)).drop(columns="text").explode("line", ignore_index=True)
        lines = lines.dropna(subset=["line"])
        lines["line_number"] = lines.groupby("file_key", sort=False).cumcount()
        lines["opening"] = lines.line.str.extract(inputs["snippet_markers"]["opening"], expand=False)
        lines["closing"] = lines.line.str.extract(inputs["snippet_markers"]["closing"], expand=False)
        lines["is_marker"] = lines.opening.notna() | lines.closing.notna()
        starts = lines.loc[lines.opening.notna(), ["file_key", "paper", "opening", "line_number"]].rename(
            columns={"opening": "snippet", "line_number": "start"})
        ends = lines.loc[lines.closing.notna(), ["file_key", "closing", "line_number"]].rename(
            columns={"closing": "snippet", "line_number": "end"})
        regions = starts.merge(ends, on=["file_key", "snippet"], how="inner", sort=False)
        regions = regions.loc[regions.end.gt(regions.start)]
        regions = regions.groupby(["file_key", "paper", "snippet", "start"], sort=False).end.min().reset_index()
        regions["snippet_key"] = regions.index

        # 4. Reconstruct reference bodies and the upstream TODO line-count/indentation.
        body = regions.merge(lines[["file_key", "line_number", "line", "is_marker"]], on="file_key", sort=False)
        body = body.loc[body.line_number.gt(body.start) & body.line_number.lt(body.end)].copy()
        body["nonblank"] = body.line.str.strip().ne("")
        body["code_line"] = body.nonblank & ~body.line.str.strip().str.startswith("#")
        body["indent"] = body.line.str.len() - body.line.str.lstrip(" \t").str.len()
        references = body.loc[~body.is_marker].groupby("snippet_key", sort=False).line.agg("\n".join)
        counts = body.groupby("snippet_key", sort=False).code_line.sum()
        indents = body.loc[body.nonblank].groupby("snippet_key", sort=False).indent.min()
        snippets = regions.set_index("snippet_key").assign(reference=references, code_lines=counts, indent=indents)
        snippets = snippets.loc[snippets.reference.fillna("").str.strip().ne("")].copy()
        snippets["indent"] = snippets.indent.fillna(0).astype(int)
        snippets["code_lines"] = snippets.code_lines.fillna(0).astype(int)
        snippets["reference_length"] = snippets.reference.str.len()
        # Preserve the reviewed rule: longest nonempty occurrence; ties follow file/line order.
        snippets = snippets.sort_values("reference_length", ascending=False, kind="stable").drop_duplicates(["paper", "snippet"])
        snippets = snippets.reset_index()

        # Replace the selected region with its TODO block and strip all remaining markers.
        expanded = snippets.merge(lines[["file_key", "line_number", "line", "is_marker"]], on="file_key", sort=False)
        visible = ~expanded.is_marker & (expanded.line_number.lt(expanded.start) | expanded.line_number.gt(expanded.end))
        remaining = expanded.loc[visible, ["snippet_key", "line_number", "line"]]
        indentation = pd.Series(" ", index=snippets.index).str.repeat(snippets.indent)
        placeholders = snippets[["snippet_key", "start"]].rename(columns={"start": "line_number"}).assign(
            line=indentation + '# TODO: Implement block "' + snippets.snippet + '"\n\n'
                 + indentation + "# Approximately " + snippets.code_lines.astype(str) + " line(s) of code.\n"
        )
        masked_lines = pd.concat([remaining, placeholders], ignore_index=True).sort_values(["snippet_key", "line_number"])
        masked_code = masked_lines.groupby("snippet_key", sort=False).line.agg("\n".join)
        snippets["masked_code"] = snippets.snippet_key.map(masked_code)

        # 5. Read declared context paths and join their source text in declaration order.
        configs = files.loc[files.path.eq("paper2code.yaml")]
        context_data = pd.json_normalize(configs.text.map(yaml.safe_load).tolist(), max_level=0)
        context_data = context_data.reindex(columns=["context_file_paths"]).assign(paper=configs.paper.to_numpy())
        contexts = context_data.explode("context_file_paths").dropna(subset=["context_file_paths"])
        contexts = contexts.rename(columns={"context_file_paths": "path"}).merge(
            files[["paper", "path", "text"]], on=["paper", "path"], how="inner", sort=False, validate="many_to_one"
        ).astype({"path": "string", "text": "string"})
        contexts["text"] = "## " + contexts.path + "\n\n" + contexts.text + "\n\n"
        context_text = contexts.groupby("paper", sort=False).text.agg("".join)
        paper_text = files.loc[files.path.eq("paper2code_paper.tex")].set_index("paper").text

        # 6. Join referenced snippets to result cells and assemble complete prompts.
        items = cells[["paper", "snippet"]].drop_duplicates().merge(
            snippets[["paper", "snippet", "reference", "masked_code"]],
            on=["paper", "snippet"], how="left", sort=False, validate="one_to_one"
        )
        missing = items.loc[items.reference.isna(), ["paper", "snippet"]]
        if not missing.empty:
            raise BuildContractError(f"researchcodebench: missing full snippet content:\n{missing.head().to_string(index=False)}")
        header = (prompt["with_paper_prefix"] + items.paper.map(paper_text).astype("string") + "\n\n").fillna(prompt["without_paper"])
        criteria = items[["reference"]].rename(columns={"reference": "reference_answer"}).assign(rule=self.grading["rule"])
        verifier = ExactMatcher(spec=self.grading["verifiers"]["greedy_run"]["description"])
        items = items.assign(
            item_key=items.paper + "::" + items.snippet,
            raw_item_id=items.paper + "::" + items.snippet,
            content=header + prompt["code_prefix"] + items.paper.map(context_text).fillna("") + items.masked_code + prompt["code_suffix"],
            grading_criterion=criteria.to_dict("records"), verifier=verifier,
            features=items[["paper"]].to_dict("records")
        )
        subjects = models[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key

        # 7. Expand completions in source order; shared registration resolves aliases and trials.
        attempts = cells.loc[cells.completions.str.len().gt(0)].explode("completions", ignore_index=True)
        if not attempts.completions.map(lambda value: isinstance(value, dict)).all():
            raise BuildContractError("researchcodebench: malformed completion")
        values = pd.json_normalize(attempts.completions.tolist(), max_level=0).reindex(columns=["passed", "completion"])
        responses = attempts.assign(
            response_key=attempts.index, subject_key=attempts.model,
            item_key=attempts.paper + "::" + attempts.snippet,
            response=values.passed.fillna(False).astype(bool).astype(float), trace=values.completion
        )
        traces = responses.loc[responses.trace.notna(), ["response_key", "trace"]]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ResearchCodeBenchBuild(__file__).main_from_args()
