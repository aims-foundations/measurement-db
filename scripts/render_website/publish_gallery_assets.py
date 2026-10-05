"""Publish gallery card images, institution logos, and audio to Hugging Face.

Matrices, prompts, and answers are read from the source tables at runtime.
Use --dry-run to inspect an upload without publishing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_REPO = "aims-foundations/measurement-db-gallery"

# Subdirectories of website/public/benchmarks/ reported separately. Anything
# else directly under the folder (the top-level <slug> card images) is grouped
# as "<top-level>".
ASSET_DIRS = ("item-images", "institutions", "audio")

# Exclude obsolete exports from both the scan and the upload.
IGNORED_DIRS = {"items", ".cache", "traces", "data", "matrices", "joined"}
IGNORE_PATTERNS = [*(f"{directory}/**" for directory in sorted(IGNORED_DIRS)), "**/.DS_Store"]

# Every publishable file must end in one of these. Same belt-and-braces posture
# as upload_to_hf.py's allowlist: a stray .py, .parquet or .env under public/
# should abort the push loudly rather than ship. Note .jpeg as well as .jpg —
# 8 card images use those (cybench.jpeg and 7 .jpg).
# .opus: tau_voice call recordings, one clip per response cell (~3 GB).
ALLOWED_SUFFIXES = {".json.gz", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".opus"}

# A full media publish includes thousands of audio clips.
MIN_EXPECTED_FILES = 5_000


def suffix_of(p: Path) -> str:
    """'.json.gz' for double-extensions, else the plain suffix."""
    return ".json.gz" if p.name.endswith(".json.gz") else p.suffix.lower()


def collect(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (publishable files, offenders that violate ALLOWED_SUFFIXES).

    Walks *everything* under root except IGNORED_DIRS, deliberately mirroring
    what upload_large_folder will actually ship. An earlier version scanned only
    ASSET_DIRS plus a `*.png` glob, which meant the allowlist below was
    validating a different set than the one being uploaded — it silently missed
    the 8 .jpg/.jpeg card images, and would equally have missed a stray .py or
    .parquet dropped at the top level.
    """
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not IGNORED_DIRS.intersection(p.relative_to(root).parts)
        and p.name != ".DS_Store"
        # .gitattributes is repo plumbing HF manages itself.
        and p.name != ".gitattributes"
    ]
    offenders = [p for p in files if suffix_of(p) not in ALLOWED_SUFFIXES]
    return files, offenders


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--web-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "website")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--create", action="store_true",
                    help="create the dataset repo (gated) before uploading")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and report; upload nothing")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--allow-small", action="store_true",
                    help=f"skip the {MIN_EXPECTED_FILES}-file sanity floor")
    args = ap.parse_args()

    root = args.web_dir / "public" / "benchmarks"
    if not root.is_dir():
        sys.exit(f"ERROR: {root} does not exist — nothing to publish.")

    files, offenders = collect(root)
    if offenders:
        shown = "\n  ".join(str(p.relative_to(root)) for p in offenders[:20])
        sys.exit(
            f"ERROR: {len(offenders)} file(s) are not publishable web assets "
            f"(allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}):\n  {shown}"
        )
    if not files:
        sys.exit(f"ERROR: no publishable files under {root}.")

    total = sum(p.stat().st_size for p in files)
    print(f"{len(files):,} files, {human(total)} under {root}")
    for name in ASSET_DIRS:
        sub = [p for p in files if p.is_relative_to(root / name)]
        if sub:
            print(f"  {name + '/':14} {len(sub):>7,} files  "
                  f"{human(sum(p.stat().st_size for p in sub)):>10}")
    cards = [p for p in files if p.parent == root]
    print(f"  {'<top-level>':14} {len(cards):>7,} files  "
          f"{human(sum(p.stat().st_size for p in cards)):>10}")
    other = [p for p in files
             if p.parent != root
             and not any(p.is_relative_to(root / d) for d in ASSET_DIRS)]
    if other:
        print(f"  {'<other>':14} {len(other):>7,} files  "
              f"{human(sum(p.stat().st_size for p in other)):>10}")

    if len(files) < MIN_EXPECTED_FILES and not args.allow_small:
        sys.exit(
            f"\nERROR: only {len(files):,} files found, expected at least "
            f"{MIN_EXPECTED_FILES:,}. This usually means the assets were never "
            f"staged in this checkout. Stage the media (or pass --allow-small if "
            f"this really is intentional)."
        )

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"\nauthenticated as {who.get('name')} → {args.repo}")

    if args.create:
        # Created PRIVATE deliberately. The API cannot set gating at creation
        # time, so a repo created with the default visibility would sit public
        # and ungated until someone changed it by hand — briefly publishing the
        # very assets the gate exists to protect. Private is the safe starting
        # point; relax it to gated: manual in the repo settings afterwards.
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"repo ready (PRIVATE): https://huggingface.co/datasets/{args.repo}")
        print("NEXT: in repo Settings, switch visibility to public and set "
              "'Gated: manual' if you want the same access policy as "
              "aims-foundations/measurement-db.")

    api.upload_large_folder(
        repo_id=args.repo,
        folder_path=str(root),
        repo_type="dataset",
        ignore_patterns=IGNORE_PATTERNS,
        num_workers=args.num_workers,
    )

    sha = api.repo_info(args.repo, repo_type="dataset").sha
    out = args.web_dir / "content" / "generated" / "gallery-revision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": args.repo, "revision": sha}, indent=2) + "\n")
    print(f"\npublished at {sha}")
    print(f"wrote {out.relative_to(args.web_dir.parent)}")
    print("Commit that file and redeploy so the site serves the new revision.")


if __name__ == "__main__":
    main()
