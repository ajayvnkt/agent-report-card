"""One module per supported input format. Each exposes NAME, LABEL, sniff(records, head) -> float
and parse(Source) -> ParseResult; formats that can be written also expose dump(events) -> str."""
from . import arl, csv_arl, langfuse, openai_messages, otel, pylog

# Order = tie-break priority when two sniffers return the same confidence.
ADAPTERS = [arl, openai_messages, otel, langfuse, pylog, csv_arl]
BY_NAME = {a.NAME: a for a in ADAPTERS}

__all__ = ["ADAPTERS", "BY_NAME"]
