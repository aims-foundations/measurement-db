"""Read explicit item membership and describe bounded slices of each testlet."""
from __future__ import annotations

import itertools
from pathlib import Path

import pyarrow.parquet as pq


def item_order(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    source = pq.ParquetFile(path)
    columns = [name for name in ("item_id", "testlet_id") if name in source.schema_arrow.names]
    bank = source.read(columns=columns).to_pydict()
    groups: dict[str, list[str]] = {}
    independent = []
    for identity, testlet in zip(bank["item_id"], bank.get("testlet_id", [None] * len(bank["item_id"])), strict=True):
        if testlet is None:
            independent.append(identity)
        elif not isinstance(testlet, str) or not testlet.strip():
            raise ValueError("testlet_id must be a nonempty string or null")
        else:
            groups.setdefault(testlet, []).append(identity)
    # Keep recorded item order within a testlet. Ungrouped releases retain their
    # existing ID order, so their prepared bundles need not be regenerated.
    groups = dict(sorted(groups.items()))
    return [item for values in groups.values() for item in values] + sorted(independent), groups


def page_testlets(items, membership, groups, *, before=(), after=()) -> list[dict]:
    result = []
    for testlet, members in itertools.groupby(items, key=membership.get):
        visible = list(members)
        if testlet is None:
            continue
        complete = groups[testlet]
        result.append({
            "testletId": testlet, "itemIds": visible, "totalItems": len(complete),
            "continuedBefore": visible[0] != complete[0] or visible[0] in before,
            "continuedAfter": visible[-1] != complete[-1] or visible[-1] in after,
        })
    return result
