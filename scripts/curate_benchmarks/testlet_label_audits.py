"""Independent checks for component labels recovered from native predictions."""

import hashlib
import json
import re
import unicodedata


def source_attempts(directory, metadata):
    if directory.name == 'arcagi':
        from .native_result_audits import _arcagi_source_records
        native, _, _, _ = _arcagi_source_records(directory, metadata)
        for (version, task, digest), record in native.items():
            puzzle = json.loads(record['item'])[1]
            checks = []
            for position, pair in enumerate(record['record']):
                candidates = [a for a in pair.values() if a is not None]
                indices = {a['metadata']['pair_index'] for a in candidates
                    if a['metadata'].get('pair_index') is not None}
                index = next(iter(indices)) if indices else position
                grade = float(any(a['answer'] == puzzle['test'][index]['output'] for a in candidates))
                checks.append((version + '::' + task + '::' + str(index), task, grade))
            yield '/'.join((version, task, digest)), record['grade'], checks
    elif directory.name == 'care_enzymes':
        from .native_result_audits import _care_source_records
        native, _ = _care_source_records(directory, metadata)
        # The legacy export is appended after the main released CSV collection.
        ordered = sorted(native, key=lambda key: (native[key]['collection'] != 'paper', key))
        for index, key in enumerate(ordered):
            entry = native[key]
            content = json.dumps(entry['inputs'], ensure_ascii=False, sort_keys=True)
            task = hashlib.sha256(unicodedata.normalize('NFC', content).encode()).hexdigest()
            method = entry['method']
            separator = None if entry['collection'] != 'paper' else (
                '; ' if method in {'BLAST', 'Foldseek'} else ',' if method == 'Pika' or 'ChatGPT' in method else None)
            top = entry['record']['0'] or '0.0.0.0'
            candidates = top.split(separator) if separator is not None else [top]
            candidates = [v if v.count('.') == 3 else '0.0.0.0' for v in candidates]
            checks = [(task + '::' + str(i), task, float(target in candidates))
                for i, target in enumerate(entry['reference'].split(';'))]
            yield str(index), entry['grade'], checks
    elif directory.name == 'afrieval':
        yield from _afrieval(directory, metadata)
    elif directory.name == 'proteininvbench':
        yield from _proteininvbench(directory, metadata)
    else:
        raise ValueError(directory.name)


def _sentences(path):
    current = []
    for line in path.read_text().splitlines():
        if line.strip():
            current.append(line.split())
        elif current:
            yield current
            current = []
    if current:
        yield current


def _afrieval(directory, metadata):
    raw = directory / 'raw'
    parameters = metadata['build']['parameters']
    references = {}
    for path in sorted((raw / 'references').rglob('test.txt')):
        version = path.relative_to(raw / 'references').parts[0]
        references[version, path.parent.name] = list(_sentences(path))
    run = 0
    for group in parameters.values():
        if group.get('kind') != 'prediction':
            continue
        for path in sorted(raw.glob(group['glob'])):
            coordinates = re.fullmatch(group['pattern'], str(path.relative_to(raw))).groupdict()
            language, version = coordinates['language'], group['version']
            gold = references[version, language]
            sentences = list(_sentences(path))
            assert len(sentences) == len(gold), 'Incomplete native prediction file'
            for sentence, targets in zip(sentences, gold):
                tokens = [row[0] for row in sentence]
                reference_tokens = [row[0] for row in targets]
                if coordinates['model'] == parameters['protocol']['digit_normalized_model'] and version == 'v1':
                    reference_tokens = [re.sub(r'\d', '0', token) for token in reference_tokens]
                    assert [row[1] for row in sentence] == [row[1] for row in targets]
                assert tokens == reference_tokens, 'Native token sequence mismatch'
                predicted, reference = [r[-1] for r in sentence], [r[1] for r in targets]
                content = json.dumps(dict(task=parameters['protocol']['instruction'], language=language, tokens=tokens), ensure_ascii=True)
                task = hashlib.sha256(content.encode()).hexdigest()
                checks = [(task + '::' + str(i), task, float(a == b))
                    for i, (a, b) in enumerate(zip(predicted, reference, strict=True))]
                yield str(run), float(predicted == reference), checks
                run += 1


def _proteininvbench(directory, metadata):
    """Check target alignment and native padding before comparing token grades."""
    import io
    import tarfile
    import numpy as np
    from .read_native_pickle import read_native_pickle

    raw = directory / 'raw'
    profile = metadata['grading']['verifiers']['native']
    assert hashlib.sha256((raw / profile['file']).read_bytes()).hexdigest() == profile['sha256']
    layouts = dict(StructGNN='finite_padded', GraphTrans='finite_padded', GCA='finite_padded',
        ProteinMPNN='full', GVP='full', AlphaDesign='finite_compact', PiFold='finite_compact', KWDesign='finite_compact')
    assert metadata['build']['parameters']['preprocessing'] == layouts
    inputs = {}
    with tarfile.open(raw / 'data.tar.gz', 'r|gz') as archive:
        for member in archive:
            if member.name not in {'data/cath4.2/chain_set.jsonl', 'data/cath4.3/chain_set.jsonl'}:
                continue
            dataset = member.name.split('/')[1].upper()
            for line in archive.extractfile(member):
                row = json.loads(line)
                assert (dataset, row['name']) not in inputs
                inputs[dataset, row['name']] = row
    with tarfile.open(raw / 'model_zoom.tar.gz', 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('/results.pt'):
                continue
            _, dataset, model, _ = member.name.split('/')
            if dataset not in {'CATH4.2', 'CATH4.3'}:
                continue
            source = read_native_pickle(io.BytesIO(archive.extractfile(member).read()))
            vocabulary = source['tokenizer'].state['all_tokens']
            assert source['tokenizer'].state['_token_to_id'] == {token: i for i, token in enumerate(vocabulary)}
            assert len(source['title']) == len(set(source['title']))
            for index, (name, target, probability) in enumerate(zip(
                    source['title'], source['true_seq'], source['pred_probs'], strict=True), 1):
                entry = inputs[dataset, name]
                coordinates = np.stack([np.asarray(entry['coords'][a], dtype=float) for a in ['N', 'CA', 'C', 'O']], axis=1)
                valid = [i for i in range(len(entry['seq'])) if np.isfinite(coordinates[i].sum())]
                positions = list(range(len(entry['seq']))) if layouts[model] == 'full' else valid
                if layouts[model] == 'finite_padded':
                    positions = positions + [-1] * (len(entry['seq']) - len(valid))
                expected = [vocabulary.index(entry['seq'][i]) if i >= 0 else 0 for i in positions]
                assert target.tolist() == expected, 'Native protein target does not align with the backbone'
                assert probability.shape == (len(target), len(vocabulary))
                predicted = probability.argmax(axis=1)
                grades = [float(a == b) for a, b in zip(predicted, expected, strict=True)]
                aggregate = float(np.float32(sum(grades) / len(grades)))
                yield member.name + ':' + str(index), aggregate, [
                    (name + '::' + str(position), name, grade) for position, grade in enumerate(grades)]
