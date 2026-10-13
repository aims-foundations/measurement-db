"""Stage the local packing study for a preview; does not generate benchmark data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stage(source, packed, website):
    target = website / 'public/benchmark-data'
    if target.exists() and any(target.iterdir()):
        raise ValueError('Preview benchmark-data destination must be empty')
    count = chunks = 0
    for directory in sorted(packed.iterdir()):
        if not directory.is_dir():
            continue
        slug = directory.name
        original = source / slug
        output = target / slug
        pack_out = output / 'lookup-pack'
        pack_out.mkdir(parents=True)
        for path in sorted(original.glob('*.json.gz')):
            shutil.copyfile(path, output / path.name)
        index = json.loads((directory / 'index.json').read_text())
        index['format'] = 'gallery-lookup-pack-v1'
        for pack in index['packs']:
            path = directory / pack['file']
            payload = path.read_bytes()
            if len(payload) != pack['bytes'] or digest(payload) != pack['sha256']:
                raise ValueError(f'Pack integrity failure for {slug}')
            pack['file'] = pack['sha256'] + '.bin'
            (pack_out / pack['file']).write_bytes(payload)
        for family in index['families'].values():
            chunks += len(family['chunks'])
        encoded = json.dumps(index, sort_keys=True, separators=(',', ':')).encode()
        name = f'index.{digest(encoded)}.json'
        (pack_out / name).write_bytes(encoded)
        descriptor = output / 'source.json.gz'
        obj = json.loads(gzip.decompress(descriptor.read_bytes()))
        obj['lookupPack'] = {'index': f'lookup-pack/{name}'}
        descriptor.write_bytes(gzip.compress(json.dumps(obj, separators=(',', ':')).encode(), mtime=0))
        count += 1
    result = {'benchmarks': count, 'indexed_chunks': chunks,
              'files': sum(p.is_file() for p in target.rglob('*')),
              'bytes': sum(p.stat().st_size for p in target.rglob('*') if p.is_file())}
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--website', type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.packed, args.website)
