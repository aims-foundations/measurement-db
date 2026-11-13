"""A recorded answer or unstructured transcript is one complete text step."""
from .source_fields import SourceTrace

CONVERTER = 'text@1'


def convert(source):
    if not isinstance(source, str):
        raise TypeError('Text traces require the original recorded string')
    out = SourceTrace(source, CONVERTER)
    step = {'type': 'llm'}
    out.project(step, 'content', [])
    out.add(step)
    return out.value(), {'llm': 1, 'text_records': 1}
