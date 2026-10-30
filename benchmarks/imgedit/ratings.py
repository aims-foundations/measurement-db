"""Read explicitly labeled ratings; retain missing judgments without guessing."""

import re


def rubric_dimensions(prompt):
    dimensions = re.findall(r'^([^:\n]+): A number from 1 to 5\.', prompt, re.MULTILINE)
    if len(dimensions) != 3 or len(set(dimensions)) != 3:
        raise ValueError('Expected three named ImgEdit rubric dimensions')
    return dimensions


def read_ratings(text, dimensions, aliases):
    """Allow Markdown and adjacent explanations around an explicit integer.

    The published average parser rejected such formatting. Its original result
    is retained separately; this parser only recovers the literal named scores.
    """
    result, pending = {}, None
    for line in text.splitlines():
        line = re.sub(r'^\s*(?:#{1,6}\s+|[-*]\s+|[1-3]\.\s+)', '', line)
        line = line.replace('**', '').strip()
        label, separator, value = line.partition(':')
        label = aliases.get(label, label)
        if separator and label in dimensions:
            pending = label
        elif pending is not None and not separator:
            value = line
        else:
            pending = None
            continue
        if not value.strip():
            continue
        score = re.fullmatch(r'\s*([1-5])(?:\s*/\s*5)?(?:\.(?:\s+.*)?|\s+[-—–(].*)?\s*', value)
        if score is None:
            if re.match(r'\s*[+\-\d.]', value):
                raise ValueError('A named ImgEdit rating has ambiguous or unsupported numeric text')
            # Some replies repeat each label before a prose explanation.
            pending = None
            continue
        if pending in result:
            raise ValueError('Repeated ImgEdit dimension rating')
        result[pending] = float(score[1])
        pending = None
    return {dimension: result.get(dimension) for dimension in dimensions}
