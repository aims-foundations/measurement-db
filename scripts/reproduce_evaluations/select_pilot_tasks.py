"""Freeze a seeded, outcome-independent pilot sample from a named task inventory."""
import hashlib
import json
from pathlib import Path


def select_tasks(slug, rows, *, seed, scope, strata=False):
    """Use SHA-256 random priorities; task IDs break the negligible collision tie."""
    rows = sorted(rows, key=lambda r: r['task_id'])
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Task IDs must be unique before selection')
    if len(rows) < 3:
        raise ValueError('Three eligible tasks required')
    def priority(kind, value):
        return hashlib.sha256(f'{seed}\0{slug}\0{kind}\0{value}'.encode()).hexdigest(),value
    if strata:
        groups = sorted({r['stratum'] for r in rows},key=lambda s:priority('stratum',s))[:3]
        if len(groups)!=3:raise ValueError('At least three strata required')
        selected=[min((r for r in rows if r['stratum']==g),key=lambda r:priority('task',r['task_id'])) for g in groups]
    else:
        selected=sorted(rows,key=lambda r:priority('task',r['task_id']))[:3]
    return dict(seed=seed,benchmark=slug,scope=scope,algorithm='SHA256(seed NUL benchmark NUL kind NUL identifier), ascending priority',
                design='Three random strata, one random task per stratum' if strata else 'Three tasks without replacement from the complete eligible list',
                eligible_count=len(rows),eligible_tasks=rows,selected_tasks=selected,
                replacement_policy='No replacement for missing assets, setup failures, grade failures or task difficulty',
                goal='Environment execution and replay of native grades; model task success is not the acceptance criterion')


def freeze_selection(path, selection):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(selection,indent=2,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text()!=payload:raise ValueError('A different frozen selection already exists')
    else:
        with path.open('x') as out:out.write(payload)
