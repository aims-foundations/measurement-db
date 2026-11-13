"""Attach source references to displayed fields while retaining the native input.

The independent source audit checks these references against the saved input.
References also distinguish recorded values from labels and other presentation.
"""
from copy import deepcopy
import json

from ..trace_format import trace


def get(value, path):
    for key in path:
        value = value[key]
    return value


class SourceTrace:
    def __init__(self, source, converter, *, json_encoded=False):
        self.source = json.loads(source) if json_encoded else source
        self.native = {'source': deepcopy(source), 'source_encoding': 'json' if json_encoded else 'native'}
        self.converter, self.steps = converter, []

    def project(self, step, target, path, *, json_text=False):
        value = get(self.source, path)
        encoding = 'json' if json_text and not isinstance(value, str) else None
        displayed = json.dumps(value, ensure_ascii=False, allow_nan=False) if encoding else value
        keys = [target] if isinstance(target, str) else target
        parent = step
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = displayed
        reference = {'source': path, 'target': keys}
        if encoding:
            reference['encoding'] = encoding
        step.setdefault('native', {}).setdefault('source_fields', []).append(reference)
        return displayed

    def add(self, step, **provenance):
        if provenance:
            step.setdefault('native', {}).update(provenance)
        self.steps.append(step)
        return step

    def value(self, **kwargs):
        return trace(self.converter, self.steps, native=self.native, **kwargs)
