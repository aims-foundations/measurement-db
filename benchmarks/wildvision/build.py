"""Tabulate WildVision's original answers, judge records and image-based tasks."""

import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class WildVision(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths, labels = parameters['paths'], parameters['labels']

        # 1. Concatenate the native answer and judgment tables, retaining full records.
        tables = {}
        for name in ['answers', 'judgments']:
            frames = []
            for path in sorted(self.raw_dir.glob(paths[name])):
                frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
                frame = frame.astype(object).where(frame.notna(), None)
                frames.append(frame.assign(native_record=frame.to_dict('records'),
                    source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
            tables[name] = pd.concat(frames, ignore_index=True)
        answers, judgments = tables['answers'], tables['judgments']
        if not judgments.games.map(len).eq(1).all() or not judgments.judge.eq(labels['judge']).all():
            raise ValueError('Unexpected grading occasion or judge in the pinned release')
        responses = answers.merge(judgments, on=['question_id', 'model'], how='outer',
            validate='one_to_one', suffixes=('_answer', '_judge'), indicator=True)
        if not responses['_merge'].eq('both').all():
            raise ValueError('Every native judgment must correspond to an original model output')
        responses['response_key'] = responses.index
        responses['grade_code'] = responses.games.str[0].str['score']
        responses['response'] = responses.grade_code.map(parameters['scores']).astype(float)

        # 2. Reshape the HF exports and verify that their overlapping observations agree.
        exports = {}
        for name in ['hf_answers', 'hf_judgments']:
            frames = []
            for path in sorted(self.raw_dir.glob(paths[name])):
                frame = pd.read_parquet(path)
                frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
            exports[name] = pd.concat(frames, ignore_index=True)
        items, released = exports['hf_answers'], exports['hf_judgments']
        metadata = ['question_id', 'instruction', 'language']
        models = items.columns.difference([*metadata, 'image', 'source_file', 'source_row']).to_list()
        if set(models) != set(released.columns) - {*metadata, 'image', 'source_file', 'source_row'}:
            raise ValueError('The HF answer and judgment releases name different models')
        for table in [items, released]:
            table['image_hash'] = table.image.map(lambda image: hashlib.sha256(image['bytes']).hexdigest())
        shared = items.merge(released, on=[*metadata, 'image_hash'], how='outer',
            validate='one_to_one', suffixes=('_answer', '_judge'), indicator=True)
        if not shared['_merge'].eq('both').all():
            raise ValueError('The HF exports disagree about task text, language or original images')
        hf_answers = items.melt(id_vars=[*metadata, 'source_file', 'source_row'], value_vars=models,
            var_name='model', value_name='hf_output')
        hf_judgments = released.melt(id_vars=['question_id', 'source_file', 'source_row'], value_vars=models,
            var_name='model', value_name='hf_verdict')
        evidence = hf_answers.merge(hf_judgments, on=['question_id', 'model'], validate='one_to_one',
            suffixes=('_answer', '_judge'))
        evidence['hf_evidence'] = evidence.to_dict('records')
        evidence = evidence.merge(responses[['question_id', 'model', 'output', 'instruction', 'language', 'grade_code']],
            on=['question_id', 'model'], how='left', validate='one_to_one', suffixes=('', '_native'))
        if (not evidence.hf_output.eq(evidence.output).all()
            or not evidence.instruction.eq(evidence.instruction_native).all()
            or not evidence.language.eq(evidence.language_native).all()
            or not evidence.hf_verdict.eq(evidence.grade_code.map(parameters['verdicts'])).all()):
            raise ValueError('A native answer or judgment differs from its overlapping HF export')
        responses = responses.merge(evidence[['question_id', 'model', 'hf_evidence']],
            on=['question_id', 'model'], how='left', validate='one_to_one')

        # 3. Link every native task and its fixed comparison answer to the original image.
        baseline = answers.loc[answers.model.eq(labels['opponent']), ['question_id', 'output']]
        items = items.merge(baseline.rename(columns={'output':'comparison_answer'}),
            on='question_id', how='left', validate='one_to_one')
        if items.comparison_answer.isna().any():
            raise ValueError('Missing fixed reference-model answer')
        responses = responses.merge(items[[*metadata, 'image_hash', 'comparison_answer']], on=metadata,
            how='left', validate='many_to_one')
        if responses.image_hash.isna().any():
            raise ValueError('A native answer refers to a different question or language')
        template = yaml.safe_load((self.raw_dir / paths['judge_config']).read_text())['prompt_template'][0]
        available = responses.loc[responses.games.str[0].str['user_prompt'].notna()]
        expected_prompts = [[dict(type='text', text=template.format(question_1=row.instruction,
            answer_1=row.comparison_answer, answer_2=row.output)), dict(type='image', image=row.image_hash)]
            for row in available.itertuples()]
        if not available.games.str[0].str['user_prompt'].eq(pd.Series(expected_prompts, index=available.index)).all():
            raise ValueError('A native judge prompt uses different answers, ordering or image bytes')
        items['item_key'] = items.question_id
        items['raw_item_id'] = items.question_id
        items['content'] = [json.dumps(dict(multimedia_elements=[
            dict(content_type='image/png', location='image.png'),
            dict(content_type='text/plain', text=text)]), ensure_ascii=False) for text in items.instruction]
        items['attachments'] = [[dict(data=image['bytes'], path='image.png', media_type='image/png', role='input')]
            for image in items.image]
        items['features'] = [dict(language=quote(language, safe=' /-._')) for language in items.language]
        items['grading_criterion'] = [dict(rule=json.dumps(dict(description=self.grading['rule'],
            comparison_model=labels['opponent'], comparison_answer=answer), ensure_ascii=False)) for answer in items.comparison_answer]
        items['verifier'] = Judge(judge=labels['judge'], judged_by='llm',
            spec=json.dumps(self.grading['verifiers']['preference'], sort_keys=True))

        # 4. Project original subjects, preference labels and complete source-linked traces.
        subjects = answers[['model']].drop_duplicates().rename(columns={'model':'subject_key'})
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(model_identifier=quote(model, safe=' /-._'),
            harness=labels['harness']) for model in subjects.subject_key]
        responses['subject_key'] = responses.model
        responses['item_key'] = responses.question_id
        responses['interactors'] = 'opponent=' + labels['opponent']
        responses['grade_status'] = responses.response.notna().map({True:'published_preference', False:'unavailable_verdict'})
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(answer_source=dict(file=row.source_file_answer, row=int(row.source_row_answer)),
            judgment_source=dict(file=row.source_file_judge, row=int(row.source_row_judge)),
            answer=row.native_record_answer, judgment=row.native_record_judge,
            hf_evidence=row.hf_evidence if isinstance(row.hf_evidence, dict) else None, grade_status=row.grade_status),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {'subjects':subjects,
            'items':items[['item_key', 'raw_item_id', 'content', 'attachments', 'features', 'grading_criterion', 'verifier']],
            'responses':responses[['response_key', 'subject_key', 'item_key', 'response', 'interactors']], 'traces':traces}


if __name__ == '__main__':
    WildVision(__file__).main_from_args()
