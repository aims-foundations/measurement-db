"""A recorded answer or unstructured transcript is one complete text step."""
import hashlib

from ..trace_format import trace

CONVERTER = 'text@1'


def convert(source):
    if not isinstance(source, str):
        raise TypeError('Text traces require the original recorded string')
    # The original input is already the complete content field. Keep its hash
    # for the independent audit instead of storing a second copy of the text.
    native = {'source_encoding': 'literal_text',
              'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest()}
    return trace(CONVERTER, [{'type': 'llm', 'content': source}], native=native), {'llm': 1, 'text_records': 1}
