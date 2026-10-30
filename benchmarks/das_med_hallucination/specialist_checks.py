"""Read subtype detections without treating skipped or uncertain judges as clean."""


def subtype_grades(judgment, subtypes):
    grades = dict.fromkeys(subtypes)
    if judgment is None:
        return grades
    seen = set()
    for decision in judgment['agent_decisions']:
        root = str(decision['code'])
        if root in seen or root not in {code[0] for code in subtypes}:
            raise ValueError('Repeated or unknown DAS specialist')
        seen.add(root)
        if type(decision['called']) is not bool:
            raise ValueError('DAS specialist invocation must be explicit')
        classification = decision.get('classification')
        if not decision['called']:
            if classification is not None:
                raise ValueError('An uncalled DAS specialist has a classification')
            continue
        if classification is None or classification == '0.5':
            continue
        codes = [code for code in subtypes if code.startswith(root)]
        if classification == '0':
            detected = set()
        elif isinstance(classification, list) and all(code in codes for code in classification):
            detected = set(classification)
        else:
            raise ValueError('DAS classification is outside the specialist vocabulary')
        grades.update({code: float(code in detected) for code in codes})
    return grades
