#!/usr/bin/env python3
"""Cluster measurement-db benchmarks by domain.

Two input sources, same commands for both:

  --csv benchmarks.csv     a spreadsheet of benchmarks (portable, editable)
  --root benchmarks/       a tree of <slug>/metadata.yaml files (the repo)

The taxonomy lives in taxonomy.yaml; this file only applies it.

Two layers, deliberately separated:

  normalize  Deterministic. Reads the declared domain, lowercases, resolves
             aliases, drops retired values. Authoritative - a curator's
             declared domain is the answer.

  suggest    Heuristic. Scores name + description + tags against the taxonomy's
             regex signals. Runs only for benchmarks that declare no domain, or
             in audit mode to flag disagreements. It never writes.

Commands
  report     grouped listing of every benchmark by canonical domain
  check      CI mode: exit 1 on unknown values, retired values, or parse errors
  suggest    propose domains for benchmarks that declare none (or --all)
  audit      compare declared vs suggested, list disagreements
  eval       measure the suggester against all declared labels
  export     write a CSV of every benchmark with declared + canonical domains
  vocab      emit the canonical domain list, with counts, as paste-ready Python
  add        pull one or more benchmarks from the repo into the CSV by slug

Usage
  python cluster_benchmarks.py export --root benchmarks --out benchmarks.csv
  python cluster_benchmarks.py report --csv benchmarks.csv
  python cluster_benchmarks.py check  --csv benchmarks.csv
  python cluster_benchmarks.py suggest --csv benchmarks.csv --slug my_new_bench
  python cluster_benchmarks.py vocab --csv benchmarks.csv
  python cluster_benchmarks.py add --slug taubench --root benchmarks
  python cluster_benchmarks.py add --sync --root benchmarks

This is a review and vocabulary-authoring aid, not part of the build pipeline.
It never writes to any benchmark's metadata.yaml. The canonical domain list it
produces (see the `vocab` command) is the artifact meant to be adopted
elsewhere in the repository.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# CSV columns. `domain` is the declared value and is what you edit;
# `canonical_domain` is computed on export and ignored on read.
CSV_FIELDS = [
    "slug",
    "name",
    "domain",
    "canonical_domain",
    "modality",
    "response_type",
    "multi_single_turn",
    "tags",
    "description",
]
LIST_SEP = ";"


def _split(value: str) -> list[str]:
    return [p.strip() for p in str(value or "").split(LIST_SEP) if p.strip()]


def _join(values) -> str:
    return LIST_SEP.join(str(v) for v in values)


# --------------------------------------------------------------------- taxonomy

@dataclass
class Taxonomy:
    canonical: set[str]
    aliases: dict[str, str]
    retired: dict[str, str]
    signals: dict[str, list]

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        raw = yaml.safe_load(path.read_text())
        canonical = set(raw["canonical"])
        aliases = {k.lower(): v for k, v in (raw.get("aliases") or {}).items()}
        retired = {k.lower(): v for k, v in (raw.get("retired") or {}).items()}

        bad = set(aliases.values()) - canonical
        if bad:
            raise ValueError(f"aliases point at non-canonical domains: {sorted(bad)}")
        both = set(aliases) & canonical
        if both:
            raise ValueError(f"values are both canonical and aliased: {sorted(both)}")

        signals = {}
        for domain, pats in (raw.get("signals") or {}).items():
            if domain not in canonical:
                raise ValueError(f"signals declared for non-canonical domain {domain!r}")
            signals[domain] = [re.compile(p, re.I) for p in pats]
        return cls(canonical, aliases, retired, signals)

    def resolve(self, value: str) -> tuple[str | None, str]:
        """Map one raw label to a canonical domain.

        Returns (canonical_or_None, status): canonical | aliased | retired | unknown.
        """
        key = str(value).strip().lower().replace(" ", "_").replace("&", "").strip("_")
        key = re.sub(r"_+", "_", key)
        if key in self.canonical:
            return key, "canonical"
        if key in self.aliases:
            return self.aliases[key], "aliased"
        if key in self.retired:
            return None, "retired"
        return None, "unknown"


# -------------------------------------------------------------------- benchmark

@dataclass
class Benchmark:
    slug: str
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    declared: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)   # canonical
    issues: list[str] = field(default_factory=list)
    modality: list[str] = field(default_factory=list)
    response_type: str = ""
    turn: str = ""

    @property
    def text(self) -> str:
        return " ".join([self.name, self.description, " ".join(self.tags)])


def _finalize(bench: Benchmark, tax: Taxonomy) -> Benchmark:
    """Resolve declared labels to canonical domains and record any issues."""
    if not bench.declared:
        bench.issues.append("no domain declared")
    seen = set()
    for value in bench.declared:
        canon, status = tax.resolve(value)
        if status == "unknown":
            bench.issues.append(f"unknown domain {value!r}")
        elif status == "retired":
            reason = tax.retired[str(value).strip().lower()]
            bench.issues.append(f"retired domain {value!r}: {reason}")
        else:
            if status == "aliased":
                bench.issues.append(f"aliased {value!r} -> {canon!r}")
            if canon not in seen:
                seen.add(canon)
                bench.domains.append(canon)
    return bench


def load_from_root(root: Path, tax: Taxonomy) -> list[Benchmark]:
    out = []
    for meta_path in sorted(root.glob("*/metadata.yaml")):
        slug = meta_path.parent.name
        if slug.startswith("_"):
            continue
        try:
            doc = yaml.safe_load(meta_path.read_text()) or {}
        except yaml.YAMLError as exc:
            out.append(Benchmark(slug, issues=[f"unparseable metadata.yaml: {exc}"]))
            continue
        b = doc.get("benchmark", doc) or {}
        declared = b.get("domain") or []
        if isinstance(declared, str):
            declared = [declared]
        out.append(_finalize(Benchmark(
            slug=slug,
            name=str(b.get("name") or slug),
            description=str(b.get("description") or ""),
            tags=[str(t) for t in (b.get("tags") or [])],
            declared=[str(d) for d in declared],
            modality=[str(m) for m in (b.get("modality") or [])],
            response_type=str(b.get("response_type") or ""),
            turn=str(b.get("multi_single_turn") or ""),
        ), tax))
    return out


def load_one_from_root(root: Path, slug: str, tax: Taxonomy) -> Benchmark | None:
    """Read one benchmark's metadata.yaml, or None if that folder has none."""
    meta_path = root / slug / "metadata.yaml"
    if not meta_path.exists():
        return None
    for bench in load_from_root(root, tax):
        if bench.slug == slug:
            return bench
    return None


def load_from_csv(path: Path, tax: Taxonomy) -> list[Benchmark]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"slug", "domain"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
        for i, row in enumerate(reader, start=2):
            slug = (row.get("slug") or "").strip()
            if not slug or slug.startswith("#"):
                continue  # blank row or comment
            out.append(_finalize(Benchmark(
                slug=slug,
                name=(row.get("name") or slug).strip(),
                description=(row.get("description") or "").strip(),
                tags=_split(row.get("tags")),
                declared=_split(row.get("domain")),
                modality=_split(row.get("modality")),
                response_type=(row.get("response_type") or "").strip(),
                turn=(row.get("multi_single_turn") or "").strip(),
            ), tax))
    dupes = [s for s, n in Counter(b.slug for b in out).items() if n > 1]
    for b in out:
        if b.slug in dupes:
            b.issues.append("duplicate slug in CSV")
    return out


def write_csv(benches: list[Benchmark], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for b in sorted(benches, key=lambda x: x.slug):
            writer.writerow({
                "slug": b.slug,
                "name": b.name,
                "domain": _join(b.declared),
                "canonical_domain": _join(b.domains),
                "modality": _join(b.modality),
                "response_type": b.response_type,
                "multi_single_turn": b.turn,
                "tags": _join(b.tags),
                "description": " ".join(b.description.split()),
            })


# --------------------------------------------------------------------- suggest

def score(bench: Benchmark, tax: Taxonomy) -> list[tuple[str, int]]:
    """Distinct signal patterns matched per domain, strongest first."""
    hits = [(d, sum(1 for p in pats if p.search(bench.text)))
            for d, pats in tax.signals.items()]
    return sorted([h for h in hits if h[1]], key=lambda kv: (-kv[1], kv[0]))


def suggest(bench: Benchmark, tax: Taxonomy, top: int = 3) -> list[str]:
    ranked = score(bench, tax)
    if not ranked:
        return []
    best = ranked[0][1]
    return [d for d, n in ranked if n >= best - 1][:top]


# --------------------------------------------------------------------- commands

def cmd_report(benches, tax, args) -> int:
    clusters = defaultdict(list)
    for b in benches:
        for d in b.domains:
            clusters[d].append(b.slug)
    unlabeled = [b.slug for b in benches if not b.domains]
    nlab = Counter(len(b.domains) for b in benches)

    print(f"{len(benches)} benchmarks, {len(clusters)} domains, "
          f"{sum(len(v) for v in clusters.values())} assignments")
    print(f"labels per benchmark: {dict(sorted(nlab.items()))}\n")
    for d, slugs in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"## {d} ({len(slugs)})")
        print("   " + ", ".join(sorted(slugs)) + "\n")
    if unlabeled:
        print(f"## (no canonical domain) ({len(unlabeled)})")
        print("   " + ", ".join(sorted(unlabeled)) + "\n")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"clusters": {k: sorted(v) for k, v in clusters.items()},
             "unlabeled": sorted(unlabeled)}, indent=2))
        print(f"wrote {args.json}")
    return 0


def cmd_check(benches, tax, args) -> int:
    blocking = 0
    for b in benches:
        for issue in b.issues:
            warn = issue.startswith("aliased") or issue == "no domain declared"
            if not warn:
                blocking += 1
            print(f"{'warn ' if warn else 'ERROR'} {b.slug}: {issue}")
    undeclared = [b.slug for b in benches if not b.declared]
    if undeclared and args.require_domain:
        blocking += len(undeclared)
        print(f"\nERROR --require-domain: {len(undeclared)} benchmark(s) declare none: "
              + ", ".join(sorted(undeclared)))
    singles = [d for d, n in Counter(d for b in benches for d in b.domains).items() if n == 1]
    if singles:
        print(f"\nwarn  singleton domains (one member each): {sorted(singles)}")
    print(f"\n{blocking} blocking issue(s) across {len(benches)} benchmarks")
    return 1 if blocking else 0


def cmd_suggest(benches, tax, args) -> int:
    if args.slug:
        wanted = {s.strip() for s in ",".join(args.slug).split(",") if s.strip()}
        targets = [b for b in benches if b.slug in wanted]
        if not targets:
            print(f"no benchmark named {sorted(wanted)}", file=sys.stderr)
            return 1
    else:
        targets = [b for b in benches if args.all or not b.declared]
    if not targets:
        print("every benchmark already declares a domain; use --all to re-score")
        return 0
    for b in targets:
        ranked = score(b, tax)
        picks = suggest(b, tax, args.top)
        detail = ", ".join(f"{d}({n})" for d, n in ranked[:5]) or "no signal matched"
        print(f"{b.slug}")
        print(f"  suggested: {picks or ['(none - needs a human)']}")
        print(f"  evidence : {detail}\n")
    if any(not t.description for t in targets):
        print("note: suggestions use name + description + tags; rows with an empty\n"
              "      description give the suggester almost nothing to work with.")
    return 0


def cmd_audit(benches, tax, args) -> int:
    n = 0
    labeled = [b for b in benches if b.domains]
    for b in labeled:
        picks = set(suggest(b, tax, args.top))
        if picks and not (picks & set(b.domains)):
            n += 1
            print(f"{b.slug}\n  declared : {b.domains}\n  suggested: {sorted(picks)}\n")
    print(f"{n} disagreement(s) of {len(labeled)} labeled")
    return 0


def cmd_eval(benches, tax, args) -> int:
    labeled = [b for b in benches if b.domains]
    if not labeled:
        print("no labeled benchmarks to evaluate against")
        return 0
    hit = tp = fp = fn = 0
    for b in labeled:
        picks, truth = set(suggest(b, tax, args.top)), set(b.domains)
        hit += bool(picks & truth)
        tp += len(picks & truth)
        fp += len(picks - truth)
        fn += len(truth - picks)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"labeled benchmarks      : {len(labeled)}")
    print(f"at least one label hit  : {hit} ({hit / len(labeled):.0%})")
    print(f"micro precision / recall: {prec:.2f} / {rec:.2f}")
    print("\nSuggestions are a review aid, not an authority. Treat a miss as a prompt\n"
          "to check the taxonomy signals, not to relabel the benchmark.")
    return 0


def cmd_add(benches, tax, args) -> int:
    """Populate CSV rows from the repo by slug.

    `benches` is the current CSV contents; the repo is the source of the new
    rows, so nothing here is typed by hand or invented.
    """
    root = args.root or Path("benchmarks")
    if not root.exists():
        print(f"no such directory: {root}\nPass --root pointing at the repo's "
              "benchmarks/ folder.", file=sys.stderr)
        return 1

    have = {b.slug for b in benches}
    in_repo = {p.parent.name for p in root.glob("*/metadata.yaml")
               if not p.parent.name.startswith("_")}

    if args.sync:
        wanted = sorted(in_repo - have)
        if not wanted:
            print(f"nothing to add: all {len(in_repo)} benchmarks in {root} "
                  "are already in the CSV")
            return 0
        print(f"{len(wanted)} benchmark(s) in {root} missing from the CSV")
    else:
        wanted = [s.strip() for s in ",".join(args.slug or []).split(",") if s.strip()]
        if not wanted:
            print("pass --slug NAME (comma-separated for several) or --sync",
                  file=sys.stderr)
            return 1

    added, skipped, missing = [], [], []
    for slug in wanted:
        if slug in have and not args.update:
            skipped.append(slug)
            continue
        bench = load_one_from_root(root, slug, tax)
        if bench is None:
            missing.append(slug)
            continue
        benches = [b for b in benches if b.slug != slug]
        benches.append(bench)
        added.append(bench)

    for slug in missing:
        near = difflib.get_close_matches(slug, sorted(in_repo), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        print(f"ERROR no {root}/{slug}/metadata.yaml.{hint}", file=sys.stderr)
    for slug in skipped:
        print(f"skip  {slug}: already in the CSV (use --update to overwrite)")

    if not added:
        return 1 if missing else 0

    write_csv(benches, args.csv)
    print(f"\nadded {len(added)} row(s) to {args.csv}:")
    for b in added:
        print(f"  {b.slug}: domain={b.declared or '(none declared)'}"
              f" -> canonical={b.domains or '(none)'}")

    undeclared = [b for b in added if not b.declared]
    if undeclared:
        print("\nno domain declared upstream - suggestions to review:")
        for b in undeclared:
            picks = suggest(b, tax, args.top)
            print(f"  {b.slug}: {picks or ['(none - needs a human)']}")
        print("\nWrite the chosen value into the CSV's `domain` column, then run\n"
              "  python cluster_benchmarks.py check --csv "
              f"{args.csv} --require-domain")
    return 1 if missing else 0


def cmd_vocab(benches, tax, args) -> int:
    """Emit the canonical vocabulary this taxonomy defines, with usage counts.

    The point of the tool: the list below is generated from curated data plus a
    reviewed alias table, rather than accumulated by hand one benchmark at a time.
    """
    used = Counter(d for b in benches for d in b.domains)
    unused = sorted(tax.canonical - set(used))

    print("# Canonical benchmark domains.")
    print(f"# Generated by scripts/benchmark_clustering over {len(benches)} benchmarks.")
    print("# Regenerate: python cluster_benchmarks.py vocab --csv benchmarks.csv")
    print("DOMAINS = (")
    for domain, n in sorted(used.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f'    "{domain}",'.ljust(32) + f"# {n}")
    for domain in unused:
        print(f'    "{domain}",'.ljust(32) + "# 0 - declared in the taxonomy, unused")
    print(")")

    singles = sorted(d for d, n in used.items() if n == 1)
    if singles or unused:
        print(f"\n# review before adopting:", file=sys.stderr)
    if singles:
        print(f"#   one member only: {singles}", file=sys.stderr)
    if unused:
        print(f"#   no members: {unused}", file=sys.stderr)
    return 0


def cmd_export(benches, tax, args) -> int:
    out = Path(args.out)
    write_csv(benches, out)
    print(f"wrote {out} - {len(benches)} benchmarks, {len(CSV_FIELDS)} columns")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",
                    choices=["report", "check", "suggest", "audit", "eval",
                             "export", "vocab", "add"])
    # add needs both: --root is the source, --csv the destination.
    ap.add_argument("--csv", type=Path, help="read/write benchmarks as a CSV")
    ap.add_argument("--root", type=Path, help="read benchmarks from <slug>/metadata.yaml")
    ap.add_argument("--taxonomy", type=Path, default=HERE / "domain_taxonomy.yaml")
    ap.add_argument("--out", default="benchmarks.csv", help="export: output path")
    ap.add_argument("--slug", action="append",
                    help="suggest/add: a benchmark slug; repeatable or comma-separated")
    ap.add_argument("--sync", action="store_true",
                    help="add: pull every benchmark in --root missing from the CSV")
    ap.add_argument("--update", action="store_true",
                    help="add: overwrite rows already in the CSV")
    ap.add_argument("--all", action="store_true", help="suggest: re-score everything")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", help="report: also write clusters to this path")
    ap.add_argument("--require-domain", action="store_true",
                    help="check: also fail when a benchmark declares no domain")
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)

    if args.command != "add" and args.csv and args.root:
        print("pass --csv or --root, not both (add takes both: --root is the "
              "source, --csv the destination)", file=sys.stderr)
        return 1

    if args.command == "add":
        args.csv = args.csv or Path("benchmarks.csv")
        benches = load_from_csv(args.csv, tax) if args.csv.exists() else []
        return cmd_add(benches, tax, args)

    if args.csv:
        if not args.csv.exists():
            print(f"no such file: {args.csv}", file=sys.stderr)
            return 1
        benches = load_from_csv(args.csv, tax)
    else:
        root = args.root or Path("benchmarks")
        if not root.exists():
            print(f"no such directory: {root}\nPass --csv or --root.", file=sys.stderr)
            return 1
        benches = load_from_root(root, tax)

    if not benches:
        print("no benchmarks found", file=sys.stderr)
        return 1

    return {"report": cmd_report, "check": cmd_check, "suggest": cmd_suggest,
            "audit": cmd_audit, "eval": cmd_eval, "export": cmd_export,
            "vocab": cmd_vocab, "add": cmd_add}[args.command](
                benches, tax, args)


if __name__ == "__main__":
    raise SystemExit(main())
