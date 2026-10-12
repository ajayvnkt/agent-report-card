"""Format 6: CSV with ARL column names, one row per event (args/result/outcome/meta as JSON)."""
from __future__ import annotations

import csv
import io

from ..schema import FIELD_NAMES, Event
from ._common import ParseResult, Source, json_or_raw

NAME = "csv"
LABEL = "CSV with ARL columns"

csv.field_size_limit(1 << 30)


def _header(head: str) -> list[str]:
    first = next((l for l in head.split("\n") if l.strip()), "")
    try:
        row = next(csv.reader([first]))
    except (csv.Error, StopIteration):
        return []
    return [c.strip().lower() for c in row]


def sniff(records: list, head: str) -> float:
    cols = _header(head)
    if len(cols) < 2 or ("type" not in cols and "event" not in cols):
        return 0.0
    known = sum(1 for c in cols if c in FIELD_NAMES)
    score = 0.5 + 0.1 * min(known, 4)
    if "run_id" in cols:
        score += 0.05
    return min(score, 0.95)


def parse(src: Source) -> ParseResult:
    pr = ParseResult()
    with src.open_text() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            reader.fieldnames = [c.strip().lower() for c in reader.fieldnames]
            unknown = [c for c in reader.fieldnames if c not in FIELD_NAMES and c != "event"]
            if unknown:
                pr.warn(f"CSV columns not in ARL kept under meta: {', '.join(unknown)}")
        for row in reader:
            lineno = reader.line_num
            if not any((v or "").strip() for v in row.values() if isinstance(v, str)):
                continue
            if None in row:  # more cells than headers
                pr.warn("row has more cells than the header; extra cells ignored")
                row.pop(None)
            d = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                if k in ("args", "result", "outcome", "meta", "claimed_actions"):
                    d[k] = json_or_raw(v)
                else:
                    d[k] = v
            try:
                pr.events.append(Event.from_dict(d))
            except ValueError as exc:
                pr.skip(lineno, ",".join(str(v) for v in row.values())[:80], str(exc))
    return pr


def dump(events: list[Event]) -> str:
    import json
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELD_NAMES, lineterminator="\n")
    w.writeheader()
    for e in events:
        d = e.to_dict()
        row = {}
        for k, v in d.items():
            if k in ("args", "result", "outcome", "meta", "claimed_actions"):
                row[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, bool):
                row[k] = "true" if v else "false"
            else:
                row[k] = v
        w.writerow(row)
    return buf.getvalue()
