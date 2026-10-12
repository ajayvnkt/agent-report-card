"""Format 3: Python `logging` text lines.

    <ISO ts> <LEVEL> agent_run run_id=<id> seq=<n> type=<type> [tool=..] [args=<json>] ...

Values are bare tokens (no spaces) or JSON (object / array / "quoted string"). Lines without the
`agent_run` marker are skipped and counted; unknown keys are kept under `meta`.
"""
from __future__ import annotations

import json
import re

from ..schema import Event
from ._common import ParseResult, Source

NAME = "pylog"
LABEL = "Python logging text (agent_run key=value lines)"

MARKER_RE = re.compile(r"(?:^|\s)agent_run(?:\s|:|$)")
KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)=")
TS_RE = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)\]?")
LEVELS = ("DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL")
STRUCT_KEYS = ("args", "result", "outcome", "meta", "claimed_actions")
STR_KEYS = ("content", "error", "ts", "run_id", "call_id", "tool", "agent", "model", "task_type",
            "status", "type")
_DEC = json.JSONDecoder()


def sniff(records: list, head: str) -> float:
    lines = [l for l in head.split("\n") if l.strip()][:400]
    if not lines:
        return 0.0
    hits = [l for l in lines if MARKER_RE.search(l) and "type=" in l]
    if not hits:
        return 0.0
    strong = sum(1 for l in hits if "run_id=" in l) / len(hits)
    return round(min(0.97, 0.75 + 0.2 * strong + 0.02 * min(len(hits), 1)), 3)


def parse_kv(s: str) -> tuple[dict, list[str]]:
    """Parse `k=v k2={"a":1} k3="text with spaces"` into a dict. Returns (dict, problems)."""
    out: dict = {}
    problems: list[str] = []
    i, n = 0, len(s)
    while i < n:
        m = KEY_RE.search(s, i)
        if not m:
            break
        key = m.group(1)
        j = m.end()
        if j < n and s[j] in "{[\"":
            try:
                val, end = _DEC.raw_decode(s, j)
                out[key] = ("json", val)
                i = end
                continue
            except ValueError:
                problems.append(f"bad JSON value for {key}")
        end = j
        while end < n and not s[end].isspace():
            end += 1
        out[key] = ("bare", s[j:end])
        i = end
    return out, problems


def _convert(key: str, kind: str, raw):
    if kind == "json":
        return raw
    if key in STRUCT_KEYS:
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    if key == "claimed_success":
        return raw.lower() in ("true", "1", "yes")
    return raw  # numeric fields are coerced by Event.from_dict


def parse_line(line: str) -> tuple[dict | None, list[str]]:
    m = MARKER_RE.search(line)
    if not m:
        return None, []
    prefix, rest = line[: m.start()], line[m.end():]
    kv, problems = parse_kv(rest)
    d = {k: _convert(k, kind, v) for k, (kind, v) in kv.items()}
    if "ts" not in d:
        tm = TS_RE.match(prefix)
        if tm:
            d["ts"] = tm.group(1).replace(",", ".")
    return d, problems


CONTINUATION_RE = re.compile(r"^(\s+\S|Traceback \(most recent call last\)|[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b(:|$)|"
                             r"During handling of the above exception|The above exception was the direct cause)")


def parse(src: Source) -> ParseResult:
    pr = ParseResult()
    last: Event | None = None      # event a traceback / continuation line belongs to
    n_cont = 0
    for lineno, line in src.lines():
        if not line.strip():
            continue
        d, problems = parse_line(line)
        if d is None:
            if last is not None and not TS_RE.match(line) and CONTINUATION_RE.match(line):
                meta = dict(last.meta or {})
                meta["traceback"] = (meta.get("traceback", "") + "\n" + line).lstrip("\n")
                last.meta = meta
                n_cont += 1
                continue
            last = None
            pr.skip(lineno, line, "no agent_run marker")
            continue
        for p in problems:
            pr.warn(p)
        if "type" not in d:
            last = None
            pr.skip(lineno, line, "agent_run line without type=")
            continue
        try:
            last = Event.from_dict(d)
            pr.events.append(last)
        except ValueError as exc:
            last = None
            pr.skip(lineno, line, str(exc))
    if n_cont:
        pr.warn(f"attached {n_cont} traceback / continuation line(s) to the agent_run event before them (meta.traceback)")
    return pr


# ----------------------------------------------------------------------------- export
_BARE_OK = re.compile(r"^[^\s\"{\[=][^\s=]*$")


def _fmt(key: str, v) -> str:
    if key in STRUCT_KEYS:
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    s = str(v)
    if key in ("content", "error") or not _BARE_OK.match(s):
        return json.dumps(s, ensure_ascii=False)
    return s


def dump(events: list[Event]) -> str:
    lines = []
    for e in events:
        d = e.to_dict()
        ts = d.pop("ts", None) or "1970-01-01T00:00:00Z"  # format needs a prefix ts
        level = "ERROR" if e.type == "error" or e.status in ("error", "timeout") else \
            "WARNING" if e.status == "refused" else "INFO"
        parts = [f"{ts} {level} agent_run"]
        for k in ("run_id", "seq", "type"):
            if k in d:
                parts.append(f"{k}={_fmt(k, d.pop(k))}")
        for k, v in d.items():
            parts.append(f"{k}={_fmt(k, v)}")
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")
