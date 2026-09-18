"""Report trace syntax findings without changing observations or executing code.

Declare the outer format with --format. Text traces are checked only for labeled
Markdown code fences; prose (including refusals) is not assumed to be code.
JSON histories also expose fenced content and Python/Bash tool-call arguments.
Unsupported languages are reported as unchecked, not as passing validation.
Syntax findings do not cause a failing exit status: unsuccessful model outputs
remain observations. Storage-envelope requirements belong in benchmark tests.
"""
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ALIASES = {'py': 'python', 'python3': 'python', 'js': 'javascript',
           'sh': 'bash', 'shell': 'bash'}
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})\s*([^\s`~]*)[^\n]*\n', re.MULTILINE)


def parse_syntax(code, language):
    language = ALIASES.get(language.lower(), language.lower())
    try:
        if language == 'json':
            def reject_constant(value):
                raise ValueError(f'{value} is not a JSON number')
            json.loads(code, parse_constant=reject_constant)
        elif language == 'python':
            # Compilation validates context (e.g. top-level return), but does
            # not evaluate imports, function calls, or any other submitted code.
            compile(ast.parse(code), '<trace>', 'exec')
        elif language in ('javascript', 'bash'):
            executable = shutil.which('node' if language == 'javascript' else 'bash')
            if executable is None:
                return {'status': 'unchecked', 'message': f'No {language} syntax checker installed'}
            variants = (['--check', '--input-type=commonjs'], ['--check', '--input-type=module']) if language == 'javascript' else (['--noprofile', '--norc', '-n'],)
            errors = []
            for arguments in variants:
                result = subprocess.run([executable, *arguments], input=code, text=True,
                    capture_output=True, timeout=10, env={'PATH': os.defpath, 'LANG': 'C.UTF-8'})
                if result.returncode == 0:
                    return {'status': 'valid'}
                # Omit source excerpts from the report.
                lines = result.stderr.splitlines()
                error = next((line for line in lines if 'SyntaxError:' in line),
                             lines[0] if lines else 'Syntax checker failed')
                errors.append(error)
            return {'status': 'invalid', 'message': '; '.join(dict.fromkeys(errors))}
        else:
            return {'status': 'unchecked', 'message': f'No checker configured for {language or "unlabeled code"}'}
    except json.JSONDecodeError as exc:
        return {'status': 'invalid', 'message': exc.msg, 'line': exc.lineno, 'column': exc.colno}
    except SyntaxError as exc:
        return {'status': 'invalid', 'message': exc.msg, 'line': exc.lineno, 'column': exc.offset}
    except ValueError as exc:
        return {'status': 'invalid', 'message': str(exc)}
    except (RecursionError, subprocess.TimeoutExpired, OSError) as exc:
        return {'status': 'unchecked', 'message': type(exc).__name__}
    return {'status': 'valid'}


def audit_trace(trace, outer_format='text'):
    findings = []

    def check(code, language, location):
        result = dict(location=location, language=ALIASES.get(language.lower(), language.lower()),
                      **parse_syntax(code, language))
        findings.append(result)
        return result['status'] == 'valid'

    def fences(text, location):
        position = 0
        while match := FENCE.search(text, position):
            delimiter, language = match.groups()
            closer = re.compile(r'^ {0,3}' + re.escape(delimiter[0]) +
                                '{' + str(len(delimiter)) + r',}[ \t]*$', re.MULTILINE)
            end = closer.search(text, match.end())
            code = text[match.end():end.start() if end else len(text)]
            line = text.count('\n', 0, match.start()) + 1
            block = f'{location}:fence@{line}'
            if not end:
                findings.append(dict(location=block, language=language, status='invalid',
                                     message='Unclosed Markdown code fence'))
            check(code, language, block)
            position = end.end() if end else len(text)

    def walk(value, location):
        if isinstance(value, str):
            fences(value, location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f'{location}[{index}]')
        elif isinstance(value, dict):
            function = value.get('function') or value.get('name')
            arguments = value.get('arguments', value.get('args'))
            if isinstance(function, dict):
                arguments = function.get('arguments')
                function = function.get('name')
            if function in ('python', 'bash'):
                if isinstance(arguments, str):
                    if check(arguments, 'json', location + '.arguments'):
                        arguments = json.loads(arguments)
                key = 'code' if function == 'python' else 'cmd'
                if isinstance(arguments, dict) and isinstance(arguments.get(key), str):
                    check(arguments[key], function, location + '.arguments.' + key)
            for key, child in value.items():
                walk(child, f'{location}.{key}')

    if outer_format == 'text':
        fences(trace, '$')
    elif check(trace, outer_format, '$') and outer_format == 'json':
        walk(json.loads(trace), '$')
    return findings


def audit_parquet(path, outer_format='text'):
    import pyarrow.parquet as pq
    counts = Counter()
    observations = []
    for batch in pq.ParquetFile(path).iter_batches(columns=['response_id', 'trace']):
        for row in batch.to_pylist():
            counts['traces'] += 1
            checks = audit_trace(row['trace'], outer_format)
            counts['checks'] += len(checks)
            counts.update(check['status'] for check in checks)
            if not checks:
                counts['prose_or_untyped_traces'] += 1
            problems = [check for check in checks if check['status'] != 'valid']
            if problems:
                observations.append(dict(response_id=row['response_id'], findings=problems))
    return dict(source=str(Path(path).resolve()), outer_format=outer_format,
                policy='Flag syntax findings; preserve every source trace, observation and grade.',
                summary=dict(counts), observations=observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path)
    parser.add_argument('--format', default='text', choices=['text', 'json', 'python', 'javascript', 'bash'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix != '.json' or args.output.resolve() == args.traces.resolve():
        parser.error('Choose a separate .json report; never overwrite a source table')
    report = audit_parquet(args.traces, args.format)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
