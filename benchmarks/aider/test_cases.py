"""Recover named Catch test outcomes from the final recorded test cycle."""

import json
import re


def case_names(files):
    """Read literal TEST_CASE names, preserving strings while skipping comments."""
    names = {}
    for filename, source in files.items():
        source = re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
            lambda match: '' if match.group().startswith('/') else match.group(), source, flags=re.S)
        for match in re.finditer(r'TEST_CASE\s*\(\s*("(?:\\.|[^"\\])*")', source):
            name = json.loads(match.group(1))
            if name in names:
                raise ValueError('Duplicate native Catch test name')
            names[name] = filename
    return names


def final_outcomes(history, cycles, names):
    """Use complete rosters for passes; otherwise retain only named failures.

    The harness appends a plain fenced CMake log after each completed test
    cycle. A missing log (for example a timeout) cannot be replaced by an older
    cycle. Compilation failures do not imply individual test failures.
    """
    logs = re.findall(r'(?ms)^```\n(?=-- |\[|CMake |make)(.*?)\n```', history)
    if len(logs) != len(cycles):
        return {}
    final = logs[-1]
    passed = re.search(r'All tests passed \(\d+ assertions? in (\d+) test cases?\)', final)
    if passed:
        return dict.fromkeys(names, 1.) if int(passed.group(1)) == len(names) else {}
    summary = re.search(r'(?m)^test cases:\s+(\d+)\s+\|([^\n]+)', final)
    if not summary:
        return {}
    failures = re.search(r'(\d+) failed', summary.group(2))
    if not failures:
        return {}
    failed, unknown = set(), False
    normalized = {' '.join(name.split()): name for name in names}
    if len(normalized) != len(names):
        return {}
    for header in re.findall(r'(?ms)^-{20,}\n(.*?)\n-{20,}\n', final):
        key = ' '.join(header.split())
        if key not in normalized:
            lines = header.splitlines()
            # Catch indents SECTION labels below the actual test case name.
            if len(lines) > 1 and all(line.startswith('  ') for line in lines[1:]):
                key = ' '.join(lines[0].split())
        if key in normalized:
            failed.add(normalized[key])
        else:
            unknown = True
    if len(failed) > int(failures.group(1)):
        return {}
    complete = int(summary.group(1)) == len(names) and len(failed) == int(failures.group(1)) and not unknown
    if complete:
        return {name: float(name not in failed) for name in names}
    return {name: 0. for name in names if name in failed}
