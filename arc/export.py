"""Write ARL events back out to the other formats (round-trip support)."""
from __future__ import annotations

from pathlib import Path

from .adapters import arl, csv_arl, langfuse, openai_messages, otel, pylog
from .schema import Event

WRITERS = {
    "arl": arl.dump,
    "openai": openai_messages.dump,
    "pylog": pylog.dump,
    "otel": otel.dump,
    "otlp": lambda evs: otel.dump(evs, otlp=True),
    "csv": csv_arl.dump,
    "langfuse": langfuse.dump,
}
EXTENSIONS = {"arl": ".jsonl", "openai": ".jsonl", "pylog": ".log", "otel": ".json", "otlp": ".json", "csv": ".csv",
              "langfuse": ".json"}


def export(events: list[Event], fmt: str) -> str:
    """Serialise events to `fmt` (arl | openai | pylog | otel | otlp | csv | langfuse) and return the text."""
    if fmt not in WRITERS:
        raise ValueError(f"cannot export to {fmt!r}; choose one of {', '.join(WRITERS)}")
    return WRITERS[fmt](events)


def write(events: list[Event], fmt: str, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(export(events, fmt), encoding="utf-8")
    return p
