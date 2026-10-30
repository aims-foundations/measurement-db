"""Read explicit native check explanations without rerunning a memory test."""

import math
import re


_VERDICTS = {
    'Checklist correct': 1.,
    'Checklist Incorrect': 0.,
    'The answer makes reference to the expected answer, which is not numerical.': 1.,
    'The answer does not make reference to the expected answer, which is not numerical.': 0.,
    'The answer makes reference to the expected answer, which is numerical, and the numbers match.': 1.,
    'The answer makes reference to the expected answer, which is numerical, and the numbers do not match.': 0.,
}


def component_grades(task, record):
    """Return source reference/check/grade triples, or [] when detail is unavailable.

    Published manual revisions may disagree with the automatic explanations.
    In that case preserve the published aggregate rather than silently importing
    old check grades as its components. Dynamic Restaurant prose is not a scorecard.
    """
    targets, reasons = record['expected_responses'], record['reasoning']
    if task in {'Delayed Recall', 'Instruction Recall', 'Trigger Response'}:
        if len(targets) != len(reasons) or not targets:
            return []
        grades = []
        for target, reason in zip(targets, reasons):
            grade = _VERDICTS.get(reason)
            if task == 'Trigger Response':
                if reason == f"'{target}' is in the response.":
                    grade = 1.
                elif reason == f"'{target}' is not in the response.":
                    grade = 0.
            if grade is None:
                return []
            grades.append(grade)
        if not math.isclose(sum(grades) / len(grades) * record['max_score'], record['score'], abs_tol=1e-9):
            return []
        return [(str(i), target, grade) for i, (target, grade) in enumerate(zip(targets, grades))]
    if task == 'NameList':
        # The early release spells out all missed names and the native count.
        match = re.fullmatch(r'(\d+) names out of (\d+) were found\. Names (.+) were not in the response\.', '\n'.join(reasons))
        if match and int(match[2]) == len(targets) == record['max_score']:
            missing = match[3].split(', ')
            if len(set(targets)) == len(targets) and set(missing) <= set(targets):
                grades = [float(target not in missing) for target in targets]
                if sum(grades) == int(match[1]) == record['score']:
                    return [(str(i), target, grade) for i, (target, grade) in enumerate(zip(targets, grades))]
    if task == 'Spy Meeting' and len(targets) == 3 and len(reasons) >= 3:
        grades = []
        for target, reason in zip(targets, reasons):
            if reason == 'Answer contains expected keyword(s)':
                grades.append(1.)
            elif reason == f'{target} not found in answer.':
                grades.append(0.)
            else:
                return []
        extras = reasons[3:]
        if not all(re.fullmatch(r'Answer also contains `.+`, which indicates that the agent has recalled something incorrect\.', r) for r in extras):
            return []
        confusion = bool(extras)
        if math.isclose(max((sum(grades) - confusion) / 3, 0.), record['score'], abs_tol=1e-9):
            return [(str(i), target, grade) for i, (target, grade) in enumerate(zip(targets, grades))] + [
                ('no_conflicting_keywords', 'No incorrect keyword from the native distractor lists', float(not confusion))]
    return []
