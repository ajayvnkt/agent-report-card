"""Write examples/: the first 3 demo runs in every supported format, plus policy files.

    python scripts/make_examples.py
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc import export, load_any  # noqa: E402
OUT = ROOT / "examples"
SPEC = [("arl", "example_arl.jsonl"), ("openai", "example_openai_messages.jsonl"), ("pylog", "example_python.log"),
        ("otel", "example_otel_spans.json"), ("otlp", "example_otlp.json"), ("csv", "example_events.csv"),
        ("langfuse", "example_langfuse.json")]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    events, _ = load_any(ROOT / "sample_logs/demo_agent_runs.jsonl")
    keep = list(dict.fromkeys(e.run_id for e in events))[:3]
    evs = [e for e in events if e.run_id in keep]
    for fmt, name in SPEC:
        (OUT / name).write_text(export(evs, fmt), encoding="utf-8")
    for name in ("skydesk_policy.yaml", "shopdesk_policy.yaml"):
        shutil.copy(ROOT / "sample_logs" / name, OUT / name)
    doc = (ROOT / "docs/POLICY_FORMAT.md").read_text(encoding="utf-8")
    m = re.search(r"Minimal example.*?```ya?ml\n(.*?)```", doc, flags=re.S)
    (OUT / "policy_template.yaml").write_text(m.group(1) if m else "", encoding="utf-8")
    shutil.copy(ROOT / "arc/arl_logger.py", OUT / "arl_logger.py")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
