"""Command line:

    python -m arc analyze <file> [--policy p.yaml] [--out report.json|report.md] [--format FMT] [--no-model]
    python -m arc convert <file> --to arl|openai|pylog|otel|otlp|csv|langfuse --out <path> [--format FMT]
    python -m arc formats
    python -m arc train <file-with-outcomes> [--policy p.yaml] [--out model.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_analyze(a) -> int:
    from .pipeline import run
    rep = run(a.file, policy=a.policy, model=None if a.no_model else "default", fmt=a.format)
    if a.out:
        out = Path(a.out)
        text = rep.to_markdown() if out.suffix.lower() in (".md", ".markdown") else rep.to_json()
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(rep.to_markdown())
    return 0


def _cmd_convert(a) -> int:
    from .detect import load_any
    from .export import write
    events, rep = load_any(a.file, fmt=a.format)
    p = write(events, a.to, a.out)
    print(f"{rep['format_label']} -> {a.to}: {rep['n_runs']} runs, {rep['n_events']} events -> {p}", file=sys.stderr)
    for w in rep["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    from .detect import ACCEPTED_FORMATS_HELP, UnrecognizedFormatError
    ap = argparse.ArgumentParser(prog="python -m arc", description="Agent Report Card -- analyse tool-using AI agent logs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="analyse a log file and print/write the report")
    p.add_argument("file")
    p.add_argument("--policy", help="policy YAML/JSON (default: generic built-in policy)")
    p.add_argument("--out", help="report.json or report.md (default: markdown to stdout)")
    p.add_argument("--format", help="force the input format (arl|openai|pylog|otel|csv|langfuse)")
    p.add_argument("--no-model", action="store_true", help="use the rule-of-thumb score instead of the trained model")
    p.set_defaults(func=_cmd_analyze)
    c = sub.add_parser("convert", help="convert a log file to another format (e.g. to canonical ARL)")
    c.add_argument("file")
    c.add_argument("--to", required=True, choices=["arl", "openai", "pylog", "otel", "otlp", "csv", "langfuse"])
    c.add_argument("--out", required=True)
    c.add_argument("--format")
    c.set_defaults(func=_cmd_convert)
    f = sub.add_parser("formats", help="list the accepted input formats")
    f.set_defaults(func=lambda a: print(ACCEPTED_FORMATS_HELP) or 0)
    t = sub.add_parser("train", help="train the success model (see arc/model.py)", add_help=False)
    t.set_defaults(func=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        from .model import _main
        return _main(argv)
    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except UnrecognizedFormatError as e:
        print(str(e), file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:     # bad policy, unknown --format, ...
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
