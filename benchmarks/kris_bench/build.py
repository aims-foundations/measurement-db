"""Tabulate KRIS-Bench's original images, recorded instructions and judge ratings."""

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class KrisBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate original task annotations and retain ordered image roles.
        parts = []
        for path in sorted(self.raw_dir.glob(parameters['paths']['annotations'])):
            records = pd.Series(json.loads(path.read_text()), dtype=object).rename_axis('source_id').reset_index(name='annotation')
            records = records.join(pd.json_normalize(records.annotation, max_level=0))
            records['category'] = path.parent.name
            records['annotation_file'] = str(path.relative_to(self.raw_dir))
            parts.append(records)
        bank = pd.concat(parts, ignore_index=True)
        bank['ori_img'] = bank.ori_img.map(lambda value: value if isinstance(value, list) else [value])
        bank['gt_img'] = bank.gt_img.fillna('')
        attachments = []
        for row in bank.itertuples():
            slots = [('input', name) for name in row.ori_img]
            if row.gt_img:
                slots.append(('grading', row.gt_img))
            images = []
            for role, name in slots:
                source = Path(row.annotation_file).parent / name
                with (self.raw_dir / source).open('rb') as stream:
                    header = stream.read(12)
                media_type = next(mime for signature, mime in parameters['image_signatures'].items()
                    if header.startswith(bytes.fromhex(signature)))
                images.append(dict(source_path=str(source), path=f'{role}/{row.category}/{name}', role=role, media_type=media_type))
            attachments.append(images)
        bank['attachments'] = attachments

        # 2. Read native ratings and index complete generated images without extraction.
        parts, outputs = [], []
        for path in sorted(self.raw_dir.glob(parameters['paths']['results'])):
            model = path.name.removesuffix('.tar.gz')
            with tarfile.open(path, 'r|gz') as archive:
                for member in archive:
                    name = Path(member.name)
                    if not member.isfile() or name.name.startswith('.'):
                        continue
                    if name.name in parameters['judge_files']:
                        records = pd.Series(json.load(archive.extractfile(member)), dtype=object).rename_axis('source_id').reset_index(name='native_record')
                        parts.append(records.assign(subject_key=model, category=name.parent.name,
                            source_file=str(path.relative_to(self.raw_dir)), member=member.name,
                            judge=parameters['judge_files'][name.name]))
                    elif name.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                        with archive.extractfile(member) as stream:
                            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                        outputs.append(dict(subject_key=model, category=name.parent.name,
                            source_id=name.stem.split('-')[0], output_filename=name.name,
                            output=dict(source_file=str(path.relative_to(self.raw_dir)), member=member.name,
                                bytes=member.size, sha256=digest)))
        records = pd.concat(parts, ignore_index=True)
        records = records.join(pd.json_normalize(records.native_record))
        records = records.merge(bank, on=['category', 'source_id'], how='left', validate='many_to_one', suffixes=('', '_bank'))
        records = records.merge(pd.DataFrame(outputs), on=['subject_key', 'category', 'source_id'], how='left', validate='many_to_one')
        if records.annotation.isna().any() or records.output.isna().any():
            raise ValueError('A saved rating lacks its original task or generated image')
        for row in records.itertuples():
            if Path(row.output_filename).stem != row.source_id and row.output_filename != row.ori_img[0]:
                raise ValueError('A suffixed output does not name its task’s first input frame')

        # 3. Unpivot present score fields, preserving explicit nulls and invalid ratings.
        records['instruction_text'] = records.native_record.map(lambda row: row.get('ins_en', row.get('instruction')))
        records['explanation_text'] = records.native_record.map(lambda row: row.get('explain_en', row.get('explain')))
        if records.instruction_text.isna().any() or records.explanation_text.isna().any():
            raise ValueError('A judge record has no complete historical task wording')
        fields = list(parameters['score_fields'])
        responses = records.melt(id_vars=[column for column in records.columns if column not in fields],
            value_vars=fields, var_name='metric', value_name='native_score')
        responses = responses.loc[[field in record for field, record in zip(responses.metric, responses.native_record)]].copy()
        responses['response'] = pd.to_numeric(responses.native_score)
        if responses.response.isin([float('inf'), float('-inf')]).any():
            raise ValueError('An infinite upstream rating is not a missing grade')
        valid = responses.response.isin([1, 2, 3, 4, 5])
        responses['grade_status'] = 'recorded'
        responses.loc[responses.response.isna(), 'grade_status'] = 'missing_upstream_score'
        responses.loc[responses.response.notna() & ~valid, 'grade_status'] = 'invalid_upstream_score'
        responses.loc[~valid, 'response'] = None
        responses['response_key'] = responses.source_file + '/' + responses.member + '#' + responses.source_id + '/' + responses.metric
        responses['item_key'] = [json.dumps([r.category, r.source_id, r.instruction_text, r.explanation_text, r.judge, r.metric], ensure_ascii=False)
            for r in responses.itertuples()]

        # 4. Keep historical instructions, grading dimensions and judges in item identity.
        items = responses.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.category + '::' + items.source_id
        items['content'] = [json.dumps(dict(multimedia_elements=[dict(content_type='text/plain', text=row.instruction_text)]
            + [dict(content_type=image['media_type'], location=image['path']) for image in row.attachments if image['role'] == 'input']),
            ensure_ascii=False) for row in items.itertuples()]
        items['features'] = [dict(category=row.category, input_scope=parameters['labels']['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=self.grading['rule'] + '\nDimension: ' + parameters['score_fields'][row.metric],
            **(dict(reference_answer=row.explanation_text) if row.explanation_text else {})) for row in items.itertuples()]
        items['verifier'] = [Judge(judged_by='llm', spec=json.dumps(dict(self.grading['verifiers'][row.judge], metric=row.metric), sort_keys=True))
            for row in items.itertuples()]
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(**parameters['subject_features'], source_model=model) for model in subjects.subject_key]

        # 5. Preserve complete native records and exact output-image associations.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, member=row.member, source_id=row.source_id,
            judge=row.judge, metric=row.metric, native_record=row.native_record, grade_status=row.grade_status,
            annotation_file=row.annotation_file, annotation=row.annotation, output=row.output), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects, 'items': items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    KrisBench(__file__).main_from_args()
