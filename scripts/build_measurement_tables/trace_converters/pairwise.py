"""One evaluated model's comparison side, with the recorded judge output."""
from .source_fields import SourceTrace

CONVERTER = 'pairwise@1'


def convert(source, *, reference):
    out = SourceTrace(source, CONVERTER, json_encoded=isinstance(source, str))
    record = out.source['native_record']
    if reference not in {record['generator_1'], record['generator_2']}:
        raise ValueError('A pairwise record does not contain its declared fixed reference')
    # This is the same subject orientation as the benchmark builder. When the
    # generator labels coincide, the fixed bank still occupies the first side.
    side = 2 if record['generator_1'] == reference else 1
    if not isinstance(record[f'output_{side}'], str):
        raise TypeError('The evaluated pairwise output must be its original recorded string')
    step = {'type': 'llm'}
    out.project(step, 'model', ['native_record', f'generator_{side}'])
    out.project(step, 'content', ['native_record', f'output_{side}'], json_text=True)
    out.add(step)
    completion = record.get('raw_completion')
    if completion is not None:
        prefix = ['native_record', 'raw_completion']
        if isinstance(completion, dict) and isinstance(completion.get('message'), dict) and completion['message'].get('content') is not None:
            path = prefix + ['message', 'content']
        elif isinstance(completion, dict) and completion.get('text') is not None:
            path = prefix + ['text']
        else:
            path = prefix
        out.project_section('evaluation', ['judge', 'content'], path, json_text=True)
        if record.get('annotator') is not None:
            out.project_section('evaluation', ['judge', 'participant'], ['native_record', 'annotator'])
    # price_per_example, time_per_example and raw_completion token counts belong
    # to the judge, not to generation of the evaluated model's answer.
    return out.value(), {'llm': 1, 'comparison_sides': 1, 'judge_outputs': int(completion is not None)}
