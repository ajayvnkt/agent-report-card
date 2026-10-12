"""Shared input plumbing for adapters: byte/path sources, tolerant JSON record streaming, reports."""
from __future__ import annotations

import io
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..schema import Event

HEAD_BYTES = 256 * 1024
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class Source:
    """A log file given as a path or as bytes. Text is decoded as UTF-8 (BOM stripped, invalid
    bytes replaced) with universal newlines (CRLF / CR -> LF). Lines are streamed, never all
    held in memory, for line-oriented formats."""

    def __init__(self, data: bytes | None = None, path: str | Path | None = None, name: str = ""):
        if data is None and path is None:
            raise ValueError("Source needs data or path")
        self.data = data
        self.path = Path(path) if path is not None else None
        self.name = name or (self.path.name if self.path else "upload")
        self._head: str | None = None

    @property
    def stem(self) -> str:
        n = Path(self.name).name
        for suf in (".jsonl", ".json", ".ndjson", ".log", ".txt", ".csv"):
            if n.lower().endswith(suf):
                return n[: -len(suf)] or "run"
        return n.split(".")[0] or "run"

    def _binary(self):
        return open(self.path, "rb") if self.data is None else io.BytesIO(self.data)

    def open_text(self) -> io.TextIOWrapper:
        return io.TextIOWrapper(self._binary(), encoding="utf-8-sig", errors="replace", newline=None)

    def lines(self) -> Iterator[tuple[int, str]]:
        with self.open_text() as fh:
            for i, line in enumerate(fh, 1):
                yield i, line.rstrip("\n")

    def text(self) -> str:
        with self.open_text() as fh:
            return fh.read()

    def head(self) -> str:
        if self._head is None:
            with self._binary() as fh:
                raw = fh.read(HEAD_BYTES)
            t = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            self._head = t
        return self._head


@dataclass
class ParseResult:
    events: list[Event] = field(default_factory=list)
    warnings: Counter = field(default_factory=Counter)
    skipped_lines: int = 0
    skipped_examples: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings[msg] += 1

    def skip(self, lineno: int | None, line: str, why: str = "not parseable") -> None:
        self.skipped_lines += 1
        if len(self.skipped_examples) < 5:
            loc = f"line {lineno}" if lineno else "record"
            self.skipped_examples.append(f"{loc} ({why}): {line[:80]!r}")

    def warning_list(self, cap: int = 50) -> list[str]:
        out = [m if n == 1 else f"{m} (x{n})" for m, n in self.warnings.most_common()]
        if len(out) > cap:
            out = out[:cap] + [f"... and {len(out) - cap} more warning types"]
        return out


def loads_lenient(s: str):
    """json.loads, then retry after dropping trailing commas. Returns (value, repaired)."""
    try:
        return json.loads(s), False
    except ValueError:
        pass
    t = s.rstrip().rstrip(",").rstrip()
    t = TRAILING_COMMA_RE.sub(r"\1", t)
    return json.loads(t), True  # may raise


def _concatenated(text: str):
    """Yield JSON values from text holding several values back to back (pretty-printed dumps)."""
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        try:
            v, j = dec.raw_decode(text, i)
        except ValueError:
            return
        yield v
        i = j


def _first_nonblank_line(src: Source) -> str:
    for line in src.head().split("\n"):
        if line.strip():
            return line.strip()
    return ""


def json_kind(src: Source) -> str:
    """'array' | 'object' | 'none' from the first non-blank character."""
    h = src.head().lstrip()
    if h.startswith("["):
        return "array"
    if h.startswith("{"):
        return "object"
    return "none"


def iter_json_records(src: Source, pr: ParseResult, keep_lists: bool = False) -> Iterator[tuple[int | None, object]]:
    """Yield (lineno, value) for each JSON record in a JSON document, JSON Lines file, or a file
    of concatenated pretty-printed JSON values. A top-level array yields its items. Garbage lines
    are skipped (counted in pr.skipped_lines), trailing commas repaired with a warning."""
    kind = json_kind(src)
    jsonl = False
    if kind == "object":
        try:
            json.loads(_first_nonblank_line(src).rstrip(","))
            jsonl = True
        except ValueError:
            jsonl = False
    if kind == "none":
        jsonl = True
    if jsonl:
        for lineno, line in src.lines():
            s = line.strip()
            if not s:
                continue
            try:
                v, repaired = loads_lenient(s)
            except ValueError:
                pr.skip(lineno, s, "not valid JSON")
                continue
            if repaired:
                pr.warn("repaired trailing comma(s) in JSON line")
            if isinstance(v, list) and not keep_lists:
                for item in v:
                    yield lineno, item
            else:
                yield lineno, v
        return
    text = src.text()
    try:
        doc, repaired = loads_lenient(text)
        if repaired:
            pr.warn("repaired trailing comma(s) in JSON document")
        items = doc if isinstance(doc, list) else [doc]
        for item in items:
            yield None, item
        return
    except ValueError:
        pass
    vals = list(_concatenated(text))
    if vals:
        pr.warn("file is several concatenated JSON values (not one document)")
        for v in vals:
            for item in (v if isinstance(v, list) else [v]):
                yield None, item
        return
    # Last resort: treat as JSON Lines with garbage.
    for lineno, line in enumerate(text.split("\n"), 1):
        s = line.strip().rstrip(",")
        if not s or s in ("[", "]"):
            continue
        try:
            v, _ = loads_lenient(s)
        except ValueError:
            pr.skip(lineno, s, "not valid JSON")
            continue
        yield lineno, v


def sample_records(src: Source, limit: int = 20) -> list:
    """Up to `limit` JSON records from the head of the file (for format sniffing only)."""
    h = src.head()
    kind = json_kind(src)
    out: list = []
    if kind == "array":
        start = h.index("[") + 1
        dec = json.JSONDecoder()
        i = start
        while len(out) < limit:
            while i < len(h) and h[i] in " \t\n,":
                i += 1
            if i >= len(h) or h[i] == "]":
                break
            try:
                v, i = dec.raw_decode(h, i)
            except ValueError:
                break
            out.append(v)
        return out
    if kind == "object":
        first = _first_nonblank_line(src).rstrip(",")
        try:
            json.loads(first)
        except ValueError:
            try:
                v, _ = json.JSONDecoder().raw_decode(h.lstrip())
                return [v]
            except ValueError:
                return []
    for line in h.split("\n"):
        s = line.strip()
        if not s:
            continue
        try:
            v, _ = loads_lenient(s)
        except ValueError:
            continue
        out.extend(v if isinstance(v, list) else [v])
        if len(out) >= limit:
            break
    return out[:limit]


def dict_records(records: list) -> list[dict]:
    return [r for r in records if isinstance(r, dict)]


def frac(records: list, pred) -> float:
    d = dict_records(records)
    if not d:
        return 0.0
    return sum(1 for r in d if pred(r)) / len(d)


def json_or_raw(s):
    """Decode a JSON string value if it is JSON, else return the original string."""
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except ValueError:
        return s


def dump_value_as_text(v) -> str:
    """Encode a tool result as text so that json_or_raw() gives back exactly `v`."""
    if isinstance(v, str):
        try:
            json.loads(v)
        except ValueError:
            return v
        return json.dumps(v, ensure_ascii=False)
    return json.dumps(v, ensure_ascii=False)
