"""Format 1: ARL JSONL (native). Also accepts a JSON array of ARL events."""
from __future__ import annotations

import json

from ..schema import EVENT_TYPES, Event, normalize_type
from ._common import ParseResult, Source, dict_records, frac, iter_json_records

NAME = "arl"
LABEL = "ARL JSONL (canonical)"


def _looks_arl(r: dict) -> bool:
    t = r.get("type", r.get("event"))
    return normalize_type(t) is not None and ("run_id" in r or "seq" in r or t in EVENT_TYPES) \
        and "messages" not in r and "turns" not in r and "span_id" not in r


def sniff(records: list, head: str) -> float:
    d = dict_records(records)
    if not d:
        return 0.0
    f = frac(d, _looks_arl)
    strong = frac(d, lambda r: r.get("type") in EVENT_TYPES and "run_id" in r)
    return round((0.7 + 0.28 * strong) * (0.5 + 0.5 * f), 3) if f >= 0.5 else 0.0


def parse(src: Source) -> ParseResult:
    pr = ParseResult()
    for lineno, rec in iter_json_records(src, pr):
        if not isinstance(rec, dict):
            pr.skip(lineno, json.dumps(rec)[:80], "not a JSON object")
            continue
        other = _other_format(rec)
        if other:
            pr.skip(lineno, json.dumps(rec, ensure_ascii=False)[:80],
                    f"looks like {other}, not an ARL event -- put each format in its own file")
            pr.warn(f"file mixes formats: some lines look like {other}; they were skipped")
            continue
        try:
            pr.events.append(Event.from_dict(rec))
        except ValueError as exc:
            pr.skip(lineno, json.dumps(rec, ensure_ascii=False)[:80], str(exc))
    return pr


def _other_format(r: dict) -> str | None:
    if "type" in r or "event" in r:
        return None
    if isinstance(r.get("messages"), list):
        return "an OpenAI chat-messages run"
    if "role" in r and ("content" in r or "tool_calls" in r):
        return "an OpenAI chat message"
    if ("span_id" in r or "spanId" in r) and ("trace_id" in r or "traceId" in r):
        return "an OpenTelemetry span"
    if "resourceSpans" in r:
        return "an OTLP JSON export"
    return None


def dump(events: list[Event]) -> str:
    return "".join(e.to_json() + "\n" for e in events)
