"""Agent Report Card -- reusable analysis for tool-using AI agents.

One call:
    rep = analyze_file("my_logs.jsonl", policy="my_policy.yaml")   # -> pipeline.Report
    rep.action_items, rep.runs[0].success_probability, rep.accuracy, rep.to_markdown()
    CLI: python -m arc analyze my_logs.jsonl --policy my_policy.yaml --out report.md

Ingestion API:
    events, report = load_any("my_logs.jsonl")      # any supported format -> ARL events
    text = export(events, "otel")                     # ARL -> arl | openai | pylog | otel | otlp | csv
    ARLLogger(...)                                    # write ARL from your own agent
"""
from .analyze import AnalysisResult, RunSummary, summarize_runs  # (module arc.analyze keeps analyze())
from .arl_logger import ARLLogger
from .detect import ACCEPTED_FORMATS_HELP, UnrecognizedFormatError, detect_format, load_any
from .export import export, write
from .pipeline import Report, analyze_events
from .pipeline import run as analyze_file
from .policy import load_policy
from .schema import (EVENT_TYPES, STATUSES, Event, finalize_events, group_runs, pair_tool_calls,
                     validate_event)

__version__ = "2.1.0"
__all__ = [
    "AnalysisResult", "Report", "RunSummary", "analyze_events", "analyze_file", "load_policy",
    "summarize_runs",
    "ARLLogger", "ACCEPTED_FORMATS_HELP", "EVENT_TYPES", "STATUSES", "Event", "UnrecognizedFormatError",
    "detect_format", "export", "finalize_events", "group_runs", "load_any", "pair_tool_calls",
    "validate_event", "write",
]
