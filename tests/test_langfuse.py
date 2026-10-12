"""Langfuse adapter: detection, mapping of real export shapes, ARL round trips and cross-format consistency.

Run with:  .venv/bin/python -m pytest -q tests/test_langfuse.py
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arc import export, load_any, pipeline  # noqa: E402
from arc.adapters._common import Source  # noqa: E402
from arc.detect import sniff  # noqa: E402
from arc.schema import pair_tool_calls, validate_event  # noqa: E402

FX = ROOT / "tests" / "fixtures"
S = ROOT / "sample_logs"
DEMO = S / "demo_agent_runs.jsonl"
POLICY = S / "skydesk_policy.yaml"
TOL = 1e-6
LF_FIXTURES = ["langfuse_traces.json", "langfuse_observations.jsonl", "langfuse_openai_integration.json"]


def by_run(events):
    out = {}
    for e in events:
        out.setdefault(e.run_id, []).append(e)
    return out


def scores(path_or_bytes, name="x.json"):
    src = Source(path=path_or_bytes) if isinstance(path_or_bytes, Path) else Source(data=path_or_bytes, name=name)
    return dict(sniff(src))


def strip_arc(text: str) -> bytes:
    """The writer's output without metadata.arc: what a plain Langfuse export of the same runs holds."""
    traces = json.loads(text)
    for t in traces:
        t.get("metadata", {}).pop("arc", None)
        for o in t["observations"]:
            o.get("metadata", {}).pop("arc", None)
    return json.dumps(traces).encode()


# ============================================================================ detection
@pytest.mark.parametrize("fname,n_runs", [("langfuse_traces.json", 2), ("langfuse_observations.jsonl", 2),
                                          ("langfuse_openai_integration.json", 1)])
def test_langfuse_fixtures_detected(fname, n_runs):
    events, rep = load_any(FX / fname)
    assert rep["format"] == "langfuse" and rep["confidence"] >= 0.9
    assert rep["n_runs"] == n_runs
    for e in events:
        assert validate_event(e) == [], (e, validate_event(e))


@pytest.mark.parametrize("fname", LF_FIXTURES)
def test_langfuse_not_confused_with_otel_or_openai(fname):
    sc = scores(FX / fname)
    assert sc["langfuse"] >= 0.9
    for other in ("otel", "openai", "arl", "pylog", "csv"):
        assert sc[other] < 0.5, (other, sc)


@pytest.mark.parametrize("fname", ["otel_otlp.json", "otel_flat.jsonl", "otel_console.json", "openai_runs.jsonl",
                                   "openai_list.json", "openai_bare_messages.json", "arl_basic.jsonl",
                                   "arl_events.json", "csv_basic.csv", "pylog_basic.log"])
def test_other_formats_never_look_like_langfuse(fname):
    sc = scores(FX / fname)
    assert sc["langfuse"] == 0.0
    assert max(sc, key=sc.get) != "langfuse"


def test_exported_langfuse_and_otel_detected_as_themselves():
    events, _ = load_any(DEMO)
    for fmt, want in (("langfuse", "langfuse"), ("otel", "otel"), ("otlp", "otel"), ("openai", "openai")):
        sc = scores(export(events, fmt).encode())
        assert max(sc, key=sc.get) == want, (fmt, sc)
        assert all(v < 0.5 for k, v in sc.items() if k != want), (fmt, sc)


def test_data_page_jsonl_and_observations_only_shapes():
    traces = json.loads(export(load_any(FX / "arl_basic.jsonl")[0], "langfuse"))
    plain = json.loads(strip_arc(json.dumps(traces)))
    want = [e.to_dict() for e in load_any(json.dumps(plain).encode(), filename="list.json")[0]]
    page = json.dumps({"data": plain, "meta": {"page": 1, "totalItems": len(plain)}}).encode()
    jsonl = "\n".join(json.dumps(t) for t in plain).encode()
    for data, name in ((page, "page.json"), (jsonl, "traces.jsonl")):
        ev, rep = load_any(data, filename=name)
        assert rep["format"] == "langfuse" and [e.to_dict() for e in ev] == want
    # observations exported on their own -> grouped by traceId, same tool sequence and statuses
    obs = [o for t in plain for o in t["observations"]]
    ev, rep = load_any(json.dumps(obs).encode(), filename="observations.json")
    assert rep["format"] == "langfuse" and rep["n_runs"] == len(plain)
    a, b = by_run(load_any(FX / "arl_basic.jsonl")[0]), by_run(ev)
    for rid in a:
        assert [(e.tool, e.status) for e in a[rid] if e.type == "tool_result"] == \
               [(e.tool, e.status) for e in b[rid] if e.type == "tool_result"]


# ============================================================================ mapping
def test_trace_mapping_tools_generations_scores():
    ev, rep = load_any(FX / "langfuse_traces.json")
    r = by_run(ev)["lf-refund-1"]
    assert [e.type for e in r] == ["run_start", "user_input", "llm_request", "llm_response", "tool_call",
                                   "tool_result", "llm_request", "llm_response", "tool_call", "tool_result",
                                   "llm_request", "llm_response", "final_answer", "run_end"]
    st = r[0]
    assert (st.agent, st.task_type, st.model) == ("refund-agent", "refund", "gpt-5-mini")
    assert st.meta == {"session_id": "sess-77", "user_id": "user-12"}
    assert r[1].content == "Please refund booking QX12, the flight was cancelled."  # last user message
    assert (r[3].tokens_in, r[3].tokens_out, r[3].cost_usd, r[3].latency_ms) == (900, 40, 0.00031, 1500)
    assert r[7].cost_usd == 0.00042 and r[7].tokens_in == 1000  # legacy "usage" + calculatedTotalCost
    # tool_calls inside the generation output are not doubled: the TOOL observations ran them
    assert [e.tool for e in r if e.type == "tool_call"] == ["verify_customer", "issue_refund"]
    assert r[4].call_id == "call_v" and r[4].args == {"booking_ref": "QX12"} and r[5].result == {"verified": True}
    assert r[9].status == "refused" and r[9].error.startswith("PolicyError") and r[9].result is None
    assert r[12].claimed_success is True and r[12].meta == {"claimed_success_inferred": True}
    assert r[13].outcome == {"verified_success": 0, "source": "payments ledger"}
    assert any("AGENT" in w for w in rep["warnings"])  # container observation ignored, and said so


def test_escalation_tool_last_generation_answer_and_metadata_outcome():
    ev, _ = load_any(FX / "langfuse_traces.json")
    r = by_run(ev)["lf-bill-2"]
    assert r[0].task_type == "billing"  # the only tag
    assert [e.type for e in r] == ["run_start", "user_input", "llm_request", "llm_response", "tool_call",
                                   "tool_result", "escalation", "llm_request", "llm_response", "final_answer",
                                   "run_end"]  # observations given out of order -> sorted by startTime
    assert r[6].content == "bereavement + duplicate charge"
    fa = r[9]
    assert fa.content.startswith("I'm very sorry") and fa.claimed_success is False  # hand-off, not done
    assert r[-1].outcome == {"verified_success": 1}
    rep = pipeline.run(FX / "langfuse_traces.json", policy=POLICY)
    run = {x.run_id: x for x in rep.runs}["lf-bill-2"]
    assert run.escalated and run.has_final_answer


def test_level_error_warning_and_status_inference():
    ev, rep = load_any(FX / "langfuse_observations.jsonl")
    assert rep["skipped_lines"] == 1
    r = by_run(ev)["lf-it-9"]
    calls = pair_tool_calls(r)
    got = [(c.tool, None if res is None else res.status) for c, res in calls]
    # SPAN with metadata.tool, TOOL with a JSON-string input, "tool:" SPAN with WARNING, unanswered TOOL
    assert got == [("lookup_user", "ok"), ("reset_password", "ok"), ("notify_it", "timeout"), ("close_ticket", None)]
    assert calls[1][0].args == {"user": "a"}
    err = next(e for e in r if e.type == "error")
    assert err.error == "agent crashed: out of retries" and err.status == "error"
    assert r[0].task_type == "password_reset" and r[0].agent == "it-helpdesk"
    fa = next(e for e in r if e.type == "final_answer")
    assert fa.claimed_success is True and not (fa.meta or {}).get("claimed_success_inferred")  # explicit
    assert r[-1].outcome["verified_success"] == 1  # boolean score named "success"
    orphan = by_run(ev)["lf-orphan"]  # observations whose trace record is missing still form a run
    assert [e.type for e in orphan] == ["run_start", "tool_call", "tool_result", "run_end"]


@pytest.mark.parametrize("level,msg,status", [
    ("ERROR", "PolicyError: refund over limit", "refused"),
    ("ERROR", "TimeoutError: upstream", "timeout"),
    ("ERROR", "ValueError: bad booking ref", "error"),
    ("ERROR", None, "error"),
    ("WARNING", "Denied by guardrail", "refused"),
    ("DEFAULT", None, "ok"),
])
def test_error_level_mapping(level, msg, status):
    o = {"id": "o1", "traceId": "t1", "type": "TOOL", "name": "issue_refund", "startTime": "2026-09-01T10:00:00Z",
         "endTime": "2026-09-01T10:00:01Z", "input": {"amount": 5}, "level": level}
    if msg:
        o["statusMessage"] = msg
    if level == "DEFAULT":
        o["output"] = {"refund_id": "rf_1"}
    ev, rep = load_any(json.dumps([o]).encode(), filename="obs.json")
    res = next(e for e in ev if e.type == "tool_result")
    assert res.status == status
    if level == "ERROR":
        assert res.error == (msg or "Error")


@pytest.mark.parametrize("score,expected", [
    ({"name": "verified_success", "value": 1}, 1), ({"name": "verified_success", "value": 0}, 0),
    ({"name": "success", "value": True}, 1), ({"name": "task_success", "stringValue": "false"}, 0),
    ({"name": "helpfulness", "value": 0.2}, None),
])
def test_scores_to_outcome(score, expected):
    t = {"id": "t1", "name": "a", "timestamp": "2026-09-01T10:00:00Z", "input": "hi", "scores": [score],
         "observations": []}
    ev, _ = load_any(json.dumps([t]).encode(), filename="t.json")
    oc = ev[-1].outcome
    assert ev[-1].type == "run_end"
    assert (oc or {}).get("verified_success") == expected


def test_generation_tool_calls_with_results_in_next_input():
    ev, _ = load_any(FX / "langfuse_openai_integration.json")
    r = ev
    assert [e.type for e in r] == ["run_start", "user_input", "llm_request", "llm_response", "tool_call",
                                   "tool_result", "llm_request", "llm_response", "final_answer", "run_end"]
    assert r[4].tool == "rebook_flight" and r[4].args == {"flight": "SD12"}
    assert r[5].call_id == "c1" and r[5].result == {"new_flight": "SD14", "seat": "12C"} and r[5].status == "ok"
    assert r[8].content == "You are rebooked on SD14, seat 12C." and r[8].claimed_success is True


def test_log_format_doc_example_parses():
    md = (ROOT / "docs" / "LOG_FORMAT.md").read_text(encoding="utf-8")
    m = re.search(r"## Langfuse exports.*?```json\n(.*?)```", md, flags=re.S)
    assert m, "Langfuse example missing from docs/LOG_FORMAT.md"
    ev, rep = load_any(m.group(1).encode(), filename="doc.json")
    assert rep["format"] == "langfuse" and rep["n_runs"] == 1
    res = [e for e in ev if e.type == "tool_result"]
    assert [(e.tool, e.status) for e in res] == [("verify_customer", "ok"), ("issue_refund", "refused")]
    fa = next(e for e in ev if e.type == "final_answer")
    assert fa.claimed_success is True and ev[-1].outcome == {"verified_success": 0, "source": "langfuse score verified_success"}
    run = pipeline.run(m.group(1).encode(), "doc.json", policy=POLICY).runs[0]
    assert run.claimed_success and run.task_type == "refund"


# ============================================================================ round trips
@pytest.mark.parametrize("fixture", ["arl_basic.jsonl", "arl_messy.jsonl", "arl_events.json", "openai_runs.jsonl",
                                     "pylog_basic.log", "otel_otlp.json", "otel_flat.jsonl", "csv_basic.csv",
                                     "csv_malformed.csv"] + LF_FIXTURES)
def test_roundtrip_lossless(fixture):
    events, _ = load_any(FX / fixture)
    back, rep = load_any(export(events, "langfuse").encode(), filename="rt.json")
    assert rep["format"] == "langfuse"
    assert [e.to_dict() for e in back] == [e.to_dict() for e in events]


def test_roundtrip_demo_and_full_sample(bulk_samples):
    for p in (DEMO, bulk_samples / "skydesk_arl.jsonl"):
        events, _ = load_any(p)
        back, rep = load_any(export(events, "langfuse").encode(), filename="rt.json")
        assert rep["warnings"] == [] and [e.to_dict() for e in back] == [e.to_dict() for e in events]


def test_example_file_is_current():
    events, _ = load_any(DEMO)
    keep = list(dict.fromkeys(e.run_id for e in events))[:3]
    want = export([e for e in events if e.run_id in keep], "langfuse")
    assert (ROOT / "examples" / "example_langfuse.json").read_text(encoding="utf-8") == want
    ev, rep = load_any(ROOT / "examples" / "example_langfuse.json")
    assert rep["format"] == "langfuse" and rep["n_runs"] == 3


# ============================================================================ cross-format consistency
def _same_analysis(a, b, label):
    A, B = {r.run_id: r for r in a.runs}, {r.run_id: r for r in b.runs}
    assert set(A) == set(B), label
    for rid, x in A.items():
        y = B[rid]
        assert x.tools_called == y.tools_called, (label, rid)
        assert [c.status for c in x.tool_calls] == [c.status for c in y.tool_calls], (label, rid)
        assert sorted((f.code, f.severity, f.tool) for f in x.findings) == \
               sorted((f.code, f.severity, f.tool) for f in y.findings), (label, rid)
        assert (x.claimed_success, x.escalated, x.has_final_answer) == \
               (y.claimed_success, y.escalated, y.has_final_answer), (label, rid)
        assert (x.outcome, x.task_type, x.agent, x.model) == (y.outcome, y.task_type, y.agent, y.model), (label, rid)
        assert abs(x.success_probability - y.success_probability) <= TOL, (label, rid)
        assert x.features() == pytest.approx(y.features(), abs=TOL), (label, rid)
    key = lambda rep: [(i.code, i.title, i.n_runs, i.severity) for i in rep.action_items]  # noqa: E731
    assert key(a) == key(b), label


@pytest.mark.parametrize("plain", [False, True], ids=["writer-output", "plain-langfuse-fields-only"])
def test_demo_runs_same_report_as_arl(plain):
    """The 16 demo runs as a Langfuse export give the same tool sequences, findings, probabilities (1e-6) and
    action items as the ARL file -- also with metadata.arc removed, i.e. from plain Langfuse fields alone."""
    events, _ = load_any(DEMO)
    text = export(events, "langfuse")
    data = strip_arc(text) if plain else text.encode()
    _same_analysis(pipeline.run(DEMO, policy=POLICY), pipeline.run(data, "demo_langfuse.json", policy=POLICY),
                   "plain" if plain else "writer")


def test_full_sample_plain_langfuse_matches_arl(bulk_samples):
    p = bulk_samples / "skydesk_arl.jsonl"
    events, _ = load_any(p)
    arl = pipeline.run(p, policy=POLICY)
    lf = pipeline.run(strip_arc(export(events, "langfuse")), "skydesk_langfuse.json", policy=POLICY)
    assert lf.source["format"] == "langfuse" and len(lf.runs) == 300
    _same_analysis(arl, lf, "skydesk")
    assert lf.accuracy["auc"] == pytest.approx(arl.accuracy["auc"], abs=TOL)
