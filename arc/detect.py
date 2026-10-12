"""Auto-detect a log file's format (content sniffing, extension only as a tie-breaker) and
normalise it to ARL events."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS, BY_NAME
from .adapters._common import Source, sample_records
from .schema import Event, finalize_events, validate_event

EXT_HINTS = {
    ".csv": "csv", ".log": "pylog", ".txt": "pylog",
}
MIN_CONFIDENCE = 0.5

ACCEPTED_FORMATS_HELP = """Accepted formats (auto-detected from the content):
  1. ARL JSONL (canonical)   one event per line: {"run_id","seq","type", ...}
  2. OpenAI chat messages    JSONL/JSON of {"run_id", "messages":[...]} or a bare messages list
  3. Python logging text     "<ts> INFO agent_run run_id=.. seq=.. type=tool_call tool=.. args={..}"
  4. OpenTelemetry spans     OTLP JSON ({"resourceSpans":[...]}) or a list/JSONL of flat spans
  5. CSV                     header row with ARL column names (run_id,seq,type,tool,args,result,...)
  6. Langfuse export         traces with "observations" (GENERATION / TOOL / SPAN / EVENT), as a JSON
                             list, {"data":[...]} or JSONL; observations alone are grouped by traceId
See docs/LOG_FORMAT.md for the full specification and examples."""


class UnrecognizedFormatError(ValueError):
    """No adapter recognised the input. The message lists the accepted formats."""


def _make_source(path_or_bytes: Any, filename: str | None) -> Source:
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return Source(data=bytes(path_or_bytes), name=filename or "upload")
    if isinstance(path_or_bytes, (str, os.PathLike)):
        p = Path(path_or_bytes)
        if not p.exists():
            raise FileNotFoundError(f"log file not found: {p}")
        return Source(path=p, name=filename or p.name)
    # file-like (e.g. Streamlit UploadedFile)
    if hasattr(path_or_bytes, "getvalue"):
        data = path_or_bytes.getvalue()
    elif hasattr(path_or_bytes, "read"):
        data = path_or_bytes.read()
    else:
        raise TypeError(f"expected a path, bytes or file-like object, got {type(path_or_bytes).__name__}")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return Source(data=data, name=filename or getattr(path_or_bytes, "name", "upload"))


def sniff(src: Source) -> list[tuple[str, float]]:
    """All adapters' confidence for this source, best first."""
    head = src.head()
    records = sample_records(src)
    hint = EXT_HINTS.get(Path(src.name).suffix.lower())
    scores = []
    for i, a in enumerate(ADAPTERS):
        try:
            c = float(a.sniff(records, head))
        except Exception:  # a sniffer must never break detection
            c = 0.0
        if c > 0 and hint == a.NAME:
            c = min(1.0, c + 0.02)
        scores.append((a.NAME, round(c, 3), -i))
    scores.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return [(n, c) for n, c, _ in scores]


def detect_format(path_or_bytes: Any, filename: str | None = None) -> tuple[str, float]:
    src = _make_source(path_or_bytes, filename)
    name, conf = sniff(src)[0]
    if conf < MIN_CONFIDENCE:
        raise UnrecognizedFormatError(_no_match_message(src))
    return name, conf


def _no_match_message(src: Source) -> str:
    head = src.head().strip()
    if not head:
        return f"'{src.name}' is empty -- nothing to analyse.\n\n{ACCEPTED_FORMATS_HELP}"
    first = head.split("\n", 1)[0][:120]
    return (f"Could not recognise the format of '{src.name}'. First line: {first!r}\n\n"
            f"{ACCEPTED_FORMATS_HELP}")


def load_any(path_or_bytes: Any, filename: str | None = None, fmt: str | None = None) -> tuple[list[Event], dict]:
    """Load a log file in any supported format and normalise it to ARL events.

    path_or_bytes: a path, raw bytes, or a file-like object (e.g. Streamlit's UploadedFile).
    filename: original name (used for messages, generated run ids and as a weak format hint).
    fmt: force a format name (arl/openai/pylog/otel/csv/langfuse) instead of auto-detecting.

    Returns (events, report) where report = {format, format_label, confidence, n_runs, n_events,
    skipped_lines, skipped_examples, warnings[]}. Raises UnrecognizedFormatError (a ValueError)
    with an actionable message when nothing matches or nothing could be parsed.
    """
    src = _make_source(path_or_bytes, filename)
    if fmt:
        if fmt not in BY_NAME:
            raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(BY_NAME)}")
        name, conf = fmt, 1.0
    else:
        ranked = sniff(src)
        name, conf = ranked[0]
        if conf < MIN_CONFIDENCE:
            raise UnrecognizedFormatError(_no_match_message(src))
    adapter = BY_NAME[name]
    pr = adapter.parse(src)
    warnings: list[str] = []
    if not pr.events:
        detail = "; ".join(pr.skipped_examples[:3])
        raise UnrecognizedFormatError(
            f"'{src.name}' looks like {adapter.LABEL}, but no events could be read from it"
            + (f" ({pr.skipped_lines} line(s) skipped, e.g. {detail})" if pr.skipped_lines else "")
            + f".\n\n{ACCEPTED_FORMATS_HELP}")
    missing_run = sum(1 for e in pr.events if not e.run_id)
    if missing_run:
        warnings.append(f"{missing_run} event(s) had no run_id; assigned run_id '{src.stem}'")
    events = finalize_events(pr.events, default_run_id=src.stem, warnings=warnings)
    problems: dict[str, int] = {}
    for e in events:
        for p in validate_event(e):
            problems[p] = problems.get(p, 0) + 1
    for p, n in problems.items():
        warnings.append(f"{n} event(s): {p}")
    if pr.skipped_lines:
        warnings.append(f"skipped {pr.skipped_lines} unreadable line(s)/record(s), e.g. "
                        + "; ".join(pr.skipped_examples[:3]))
    warnings.extend(pr.warning_list())
    report = {
        "format": name,
        "format_label": adapter.LABEL,
        "confidence": conf,
        "n_runs": len({e.run_id for e in events}),
        "n_events": len(events),
        "skipped_lines": pr.skipped_lines,
        "skipped_examples": list(pr.skipped_examples),
        "warnings": warnings,
    }
    return events, report
