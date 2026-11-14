"""Explicitly scoped preparation/publication; never rebuilds charts or evaluations."""
import argparse
import json
from pathlib import Path
import re

from .export import export, verify, digest, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--benchmarks', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--publish-branch')
    parser.add_argument('--publish-base')
    parser.add_argument('--website-manifest', type=Path)
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{40}', args.source_revision):
        parser.error('Source revision must be an immutable commit')
    if args.publish_branch and (not args.publish_branch.startswith('preview/') or not args.website_manifest):
        parser.error('Publishing requires an explicit preview/ branch and website manifest')
    if any(not re.fullmatch('[a-z0-9][a-z0-9_-]*', slug) for slug in args.benchmarks):
        parser.error('Invalid benchmark slug')
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error('Duplicate benchmark selection')
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for slug in args.benchmarks:
        path = args.tables / slug / 'traces.parquet'
        directory = args.output / slug / (digest(path)+'-'+identity())
        manifest = export(path, directory)
        checks = verify(path, directory)
        results[slug] = dict(manifest=manifest, verification=checks, directory=str(directory))
        (args.output/'verification.json').write_text(json.dumps(results, indent=2))
        print(slug, json.dumps(checks), flush=True)
    if not args.publish_branch:
        return
    from huggingface_hub import HfApi, CommitOperationAdd
    api, repo = HfApi(), 'aims-foundations/measurement-db'
    for slug, result in results.items():
        source = f'{slug}/formatted_tables/traces.parquet'
        info = api.get_paths_info(repo, [source], revision=args.source_revision, repo_type='dataset')[0]
        if not info.lfs or info.lfs.sha256 != result['manifest']['source_sha256']:
            raise ValueError(f'{slug}: local source differs from the pinned HF table')
    api.create_branch(repo, branch=args.publish_branch, revision=args.publish_base or args.source_revision,
                      repo_type='dataset', exist_ok=True)
    parent = api.dataset_info(repo, revision=args.publish_branch).sha
    operations, descriptors = [], {}
    for slug, result in results.items():
        m, directory = result['manifest'], Path(result['directory'])
        remote = f"website-trace-previews/{slug}/{m['source_sha256']}/{m['exporter']}"
        for file in sorted(directory.iterdir()):
            operations.append(CommitOperationAdd(path_in_repo=f'{remote}/{file.name}', path_or_fileobj=file))
        descriptors[slug] = dict(source=f'https://huggingface.co/datasets/{repo}/resolve/{args.source_revision}/{slug}/formatted_tables/traces.parquet',
            directory=remote, **{key:m[key] for key in ['format','source_sha256','exporter','rows','packs','indexes','page_bytes']})
    result = api.create_commit(repo, repo_type='dataset', revision=args.publish_branch,
        parent_commit=parent, operations=operations, commit_message='Stage bounded structured trace display previews')
    # Prove the published pack bytes, not just successful upload.
    for slug, value in results.items():
        local = Path(value['directory']); remote = descriptors[slug]['directory']
        files = list(local.iterdir())
        infos = {x.path:x for x in api.get_paths_info(repo, [f'{remote}/{f.name}' for f in files],
                    revision=result.oid, repo_type='dataset')}
        for file in files:
            info = infos[f'{remote}/{file.name}']
            if info.lfs and digest(file) != info.lfs.sha256:
                raise ValueError(f'Published preview checksum mismatch: {slug}/{file.name}')
        descriptors[slug]['revision'] = result.oid
    current = json.loads(args.website_manifest.read_text()) if args.website_manifest.exists() else {}
    current.update(descriptors)
    args.website_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.website_manifest.write_text(json.dumps(current, indent=2)+'\n')
    (args.output/'publication.json').write_text(json.dumps(dict(branch=args.publish_branch, revision=result.oid,
        source_revision=args.source_revision, changed_files=[o.path_in_repo for o in operations]), indent=2))
    print('Published verified trace preview artifacts:', result.oid, flush=True)


if __name__ == '__main__':
    main()
