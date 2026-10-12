"""Tests for the arc ingestion layer: schema, adapters, detection, export round-trips, logger.

Run with:  .venv/bin/python -m pytest -q tests/test_adapters.py
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arc  # noqa: E402
from arc import Event, export, load_any  # noqa: E402
from arc.detect import UnrecognizedFormatError  # noqa: E402
from arc.export import EXTENSIONS  # noqa: E402
from arc.schema import finalize_events, infer_claimed_success, pair_tool_calls, validate_event  # noqa: E402

FX = ROOT / "tests" / "fixtures"


def by_run(events):
    out = {}
    for e in events:
        out.setdefault(e.run_id, []).append(e)
    return out


# ============================================================================ detection
@pytest.mark.parametrize("fname,fmt,n_runs", [
    ("arl_basic.jsonl", "arl", 2),
    ("arl_messy.jsonl", "arl", 1),
    ("arl_events.json", "arl", 1),
    ("openai_runs.jsonl", "openai", 2),
    ("openai_list.json", "openai", 2),
    ("openai_bare_messages.json", "openai", 1),
    ("openai_malformed.jsonl", "openai", 2),
    ("pylog_basic.log", "pylog", 2),
    ("pylog_malformed.log", "pylog", 1),
    ("otel_otlp.json", "otel", 1),
    ("otel_flat.jsonl", "otel", 2),
    ("otel_console.json", "otel", 1),
    ("csv_basic.csv", "csv", 1),
    ("csv_malformed.csv", "csv", 2),
])
def test_every_fixture_detected_and_parsed(fname, fmt, n_runs):
    events, rep = load_any(FX / fname)
    assert rep["format"] == fmt
    assert rep["confidence"] >= 0.5
    assert rep["n_runs"] == n_runs
    assert rep["n_events"] == len(events) > 0
    for k in ("format", "confidence", "n_runs", "n_events", "skipped_lines", "warnings"):
        assert k in rep
    for e in events:
        assert validate_event(e) == [], (e, validate_event(e))
    for evs in by_run(events).values():
        assert [e.seq for e in evs] == list(range(len(evs)))


def test_detection_uses_content_not_extension(tmp_path):
    data = (FX / "arl_basic.jsonl").read_bytes()
    for name in ("logs.csv", "logs.txt", "logs.log", "noext"):
        p = tmp_path / name
        p.write_bytes(data)
        assert load_any(p)[1]["format"] == "arl"
    p = tmp_path / "spans.txt"
    p.write_bytes((FX / "otel_otlp.json").read_bytes())
    assert load_any(p)[1]["format"] == "otel"


def test_bytes_and_filelike_inputs():
    data = (FX / "openai_runs.jsonl").read_bytes()
    ev1, r1 = load_any(data, filename="x.jsonl")
    ev2, r2 = load_any(io.BytesIO(data), filename="x.jsonl")
    assert r1["format"] == r2["format"] == "openai"
    assert [e.to_dict() for e in ev1] == [e.to_dict() for e in ev2]


def test_unrecognised_format_has_actionable_message():
    with pytest.raises(UnrecognizedFormatError) as ei:
        load_any(FX / "unknown.txt")
    msg = str(ei.value)
    assert "docs/LOG_FORMAT.md" in msg and "OpenAI" in msg and "OpenTelemetry" in msg and "CSV" in msg
    assert "Dear diary" in msg  # shows what it saw
    with pytest.raises(UnrecognizedFormatError, match="empty"):
        load_any(FX / "empty.jsonl")
    with pytest.raises(ValueError):
        load_any(b"\n\n   \n", filename="blank.jsonl")


def test_forced_format_and_bad_name():
    ev, rep = load_any(FX / "arl_basic.jsonl", fmt="arl")
    assert rep["format"] == "arl" and rep["confidence"] == 1.0
    with pytest.raises(ValueError, match="unknown format"):
        load_any(FX / "arl_basic.jsonl", fmt="xml")


# ============================================================================ ARL + robustness
def test_arl_basic_matches_spec_example():
    ev, rep = load_any(FX / "arl_basic.jsonl")
    r1 = by_run(ev)["r-001"]
    assert [e.type for e in r1] == ["run_start", "user_input", "tool_call", "tool_result", "tool_call",
                                    "tool_result", "final_answer", "run_end"]
    assert r1[5].status == "error" and r1[5].error.startswith("PolicyError")  # explicit status kept
    assert r1[6].claimed_success is True and r1[7].outcome["verified_success"] == 0
    r2 = by_run(ev)["r-002"]
    assert "🙂" in r2[1].content and "€500" in r2[1].content
    vr = next(e for e in r2 if e.type == "tool_result" and e.tool == "verify_identity")
    assert vr.status == "refused"  # false from a verify-type tool
    pairs = pair_tool_calls(r2)
    assert [(c.tool, r is None) for c, r in pairs] == [("lookup_order", False), ("verify_identity", False),
                                                       ("issue_refund", True)]
    assert rep["warnings"] == []


def test_arl_messy_bom_crlf_garbage_trailing_commas_missing_ids():
    ev, rep = load_any(FX / "arl_messy.jsonl")
    assert rep["skipped_lines"] == 2  # garbage line + unknown event type
    assert any("trailing comma" in w for w in rep["warnings"])
    assert any("no run_id" in w for w in rep["warnings"])
    assert {e.run_id for e in ev} == {"arl_messy"}
    assert ev[0].agent == "café-bot"  # BOM stripped, unicode intact
    assert [e.type for e in ev] == ["run_start", "user_input", "tool_call", "tool_result", "final_answer", "run_end"]
    assert ev[2].args == {"plan": "pro"}
    assert ev[3].status == "error" and ev[3].error == "Error: subscription not found" and ev[3].result is None
    assert ev[3].call_id == ev[2].call_id  # paired by latest unanswered same-tool call
    assert ev[4].claimed_success is True and ev[4].meta["claimed_success_inferred"] is True
    assert all("\r" not in (e.content or "") for e in ev)


def test_arl_json_array_out_of_order_seq_and_timeout():
    ev, rep = load_any(FX / "arl_events.json")
    assert [e.type for e in ev] == ["run_start", "tool_call", "tool_result", "final_answer"]
    assert ev[2].status == "timeout"
    assert ev[3].claimed_success is False


def test_missing_seq_uses_file_order_and_missing_run_id_generated():
    lines = [{"type": "run_start"}, {"type": "tool_call", "tool": "a"}, {"type": "tool_result", "tool": "a", "result": 1}]
    data = "\n".join(json.dumps(x) for x in lines).encode()
    ev, rep = load_any(data, filename="single_run.jsonl")
    assert [e.seq for e in ev] == [0, 1, 2]
    assert {e.run_id for e in ev} == {"single_run"}


def test_huge_file_streams(tmp_path):
    p = tmp_path / "big.jsonl"
    n_runs = 4000
    with p.open("w") as fh:
        for r in range(n_runs):
            rid = f"r{r}"
            fh.write(json.dumps({"run_id": rid, "seq": 0, "type": "run_start", "agent": "a"}) + "\n")
            for k in range(5):
                fh.write(json.dumps({"run_id": rid, "seq": 1 + 2 * k, "type": "tool_call", "tool": "t",
                                     "args": {"k": k, "pad": "x" * 100}}) + "\n")
                fh.write(json.dumps({"run_id": rid, "seq": 2 + 2 * k, "type": "tool_result", "tool": "t",
                                     "result": k}) + "\n")
            fh.write(json.dumps({"run_id": rid, "seq": 11, "type": "run_end"}) + "\n")
    ev, rep = load_any(p)
    assert rep["n_runs"] == n_runs and rep["n_events"] == n_runs * 12


# ============================================================================ schema
@pytest.mark.parametrize("tool,result,error,status,expected", [
    ("x", "PolicyError: nope", None, None, "refused"),
    ("x", "NotFound: id 7", None, None, "error"),
    ("x", "Error: boom", None, None, "error"),
    ("x", "Denied", None, None, "refused"),
    ("x", None, "TimeoutError: upstream", None, "timeout"),
    ("x", None, "ValueError: bad", None, "error"),
    ("verify_identity", False, None, None, "refused"),
    ("is_open", False, None, None, "ok"),
    ("x", {"error": "rate limited"}, None, None, "error"),
    ("x", {"ok": True}, None, None, "ok"),
    ("x", "Everything fine", None, None, "ok"),
    ("x", None, None, "Success", "ok"),
    ("x", None, "PolicyError: x", "failed", "error"),  # explicit status wins
])
def test_status_inference(tool, result, error, status, expected):
    e = Event(run_id="r", type="tool_result", tool=tool, result=result, error=error, status=status)
    finalize_events([e], "r")
    assert e.status == expected


def test_event_from_dict_coercion_and_meta():
    e = Event.from_dict({"run": 7, "type": "Tool-Response", "seq": "3", "tool": "t", "custom": 1,
                         "args": '{"a": 1}', "claimed_success": "yes", "latency_ms": "12.5"})
    assert e.run_id == "7" and e.type == "tool_result" and e.seq == 3 and e.args == {"a": 1}
    assert e.meta == {"custom": 1} and e.claimed_success is True and e.latency_ms == 12.5
    with pytest.raises(ValueError):
        Event.from_dict({"type": "banana"})
    with pytest.raises(ValueError):
        Event.from_dict(["not", "a", "dict"])
    assert Event.from_dict(e.to_dict()).to_dict() == e.to_dict()


def test_validate_event_flags_problems():
    assert "tool_call without tool name" in validate_event(Event(run_id="r", type="tool_call"))
    assert any("outcome only" in p for p in validate_event(Event(run_id="r", type="final_answer", outcome={})))
    assert any("0/1" in p for p in validate_event(Event(run_id="r", type="run_end", outcome={"verified_success": 5})))


def test_pairing_by_call_id_then_latest_unanswered_same_tool():
    evs = [Event("r", "tool_call", tool="a", call_id="1"), Event("r", "tool_call", tool="a"),
           Event("r", "tool_call", tool="b"), Event("r", "tool_result", tool="a", call_id="1", result=1),
           Event("r", "tool_result", tool="a", result=2), Event("r", "tool_result", tool="b", result=3)]
    finalize_events(evs, "r")
    pairs = pair_tool_calls(evs)
    assert [(c.tool, r.result) for c, r in pairs] == [("a", 1), ("a", 2), ("b", 3)]
    assert evs[4].call_id == evs[1].call_id and evs[5].call_id == evs[2].call_id


def test_claimed_success_inference():
    assert infer_claimed_success("Your refund has been issued.") is True
    assert infer_claimed_success("Sorry, I couldn't complete that.") is False
    assert infer_claimed_success("I have escalated this to a specialist.") is False
    assert infer_claimed_success("Your balance is 42.") is None
    assert infer_claimed_success("") is None


# ============================================================================ per-format specifics
def test_openai_mapping():
    ev, rep = load_any(FX / "openai_runs.jsonl")
    r = by_run(ev)["oa-1"]
    assert [e.type for e in r] == ["run_start", "user_input", "llm_response", "tool_call", "tool_result",
                                   "tool_call", "tool_call", "tool_result", "tool_result", "final_answer", "run_end"]
    assert r[0].meta["system_prompt"] == "You are a billing agent." and r[0].task_type == "refund"
    assert r[4].result == {"id": "INV-7", "amount": 99.5, "paid": True}
    refund = next(e for e in r if e.type == "tool_result" and e.call_id == "call_b")
    assert refund.tool == "issue_refund" and refund.status == "refused"
    assert r[9].claimed_success is True and r[-1].outcome == {"verified_success": 0}
    r2 = by_run(ev)["oa-2"]
    assert r2[1].content == "What is my balance?"  # list-of-parts content
    assert r2[3].result == 42


def test_openai_bare_list_and_malformed():
    ev, rep = load_any(FX / "openai_bare_messages.json")
    assert {e.run_id for e in ev} == {"openai_bare_messages"}
    tr = next(e for e in ev if e.type == "tool_result")
    assert tr.status == "error" and tr.error == "no availability"
    ev, rep = load_any(FX / "openai_malformed.jsonl")
    assert rep["skipped_lines"] == 1
    call = next(e for e in ev if e.run_id == "bad-1" and e.type == "tool_call")
    assert call.args == {"_raw": "{not json"}
    unanswered = [c for c, r in pair_tool_calls(by_run(ev)["bad-2"]) if r is None]
    assert len(unanswered) == 1


def test_pylog_mapping_and_skips():
    ev, rep = load_any(FX / "pylog_basic.log")
    assert rep["skipped_lines"] == 2
    r = by_run(ev)["py-1"]
    assert r[0].model == "gpt-5-mini" and r[0].task_type == "address_change"
    assert r[1].content == "Please update my address to 12 Rue de la Paix, Paris"
    assert r[2].args == {"street": "12 Rue de la Paix", "city": "Paris"}
    assert r[3].status == "error" and r[3].latency_ms == 480.0 and r[3].call_id == "u1"
    assert r[4].claimed_success is True
    assert r[5].outcome == {"verified_success": 0, "source": "crm diff"} and r[5].meta == {"host": "web-3"}
    assert r[0].ts == "2026-09-02T08:00:01.120Z"
    assert by_run(ev)["py-2"][2].result == {"name": "Zoë", "tier": "gold"}
    ev, rep = load_any(FX / "pylog_malformed.log")
    assert rep["skipped_lines"] == 3
    assert any(e.status == "refused" for e in ev)
    assert ev[-1].meta["extra_key"] == "whatever"


def test_otel_mapping():
    ev, rep = load_any(FX / "otel_otlp.json")
    assert [e.type for e in ev] == ["run_start", "user_input", "llm_response", "tool_call", "tool_result",
                                    "final_answer", "run_end"]
    assert ev[0].agent == "travel-agent" and ev[0].task_type == "booking"
    assert ev[2].tokens_in == 1200 and ev[2].model == "gpt-5-mini"
    assert ev[4].status == "error" and "flight full" in ev[4].error
    assert ev[5].claimed_success is True and ev[6].outcome == {"verified_success": 0}
    ev, rep = load_any(FX / "otel_flat.jsonl")
    r = by_run(ev)["t-flat-1"]
    assert [e.tool for e in r if e.type == "tool_call"] == ["reset_password", "notify_it"]  # ordered by time
    assert r[3].status == "refused"
    r2 = by_run(ev)["t-flat-2"]  # trace with only a tool span (no root)
    assert [e.type for e in r2] == ["run_start", "tool_call", "tool_result", "run_end"]
    assert r2[2].result == {"found": False}
    ev, rep = load_any(FX / "otel_console.json")
    assert [e.type for e in ev] == ["run_start", "tool_call", "tool_result", "final_answer", "run_end"]


def test_csv_mapping():
    ev, rep = load_any(FX / "csv_basic.csv")
    assert ev[1].content == "Pay invoice 12, amount €1,200"
    assert ev[2].args == {"invoice": 12, "amount": 1200} and ev[3].result == {"payment_id": "pay_9"}
    assert ev[4].tokens_in == 1500 and ev[4].cost_usd == 0.0004
    assert ev[5].claimed_success is True and ev[6].outcome == {"verified_success": 1}
    ev, rep = load_any(FX / "csv_malformed.csv")
    assert rep["skipped_lines"] == 1
    nf = next(e for e in ev if e.type == "tool_result")
    assert nf.status == "error" and nf.error.startswith("NotFound")


# ============================================================================ full synthetic sample
@pytest.fixture(scope="module")
def skydesk(bulk_samples):
    ev, rep = load_any(bulk_samples / "skydesk_arl.jsonl")
    return ev, rep


# ============================================================================ round trips
LOSSLESS = ("arl", "csv", "pylog")


def _full(e, drop_ts):
    d = e.to_dict()
    if drop_ts:
        d.pop("ts", None)
    return d


def _otel_proj(e):
    d = e.to_dict()
    for k in ("ts", "meta", "seq", "run_id"):
        d.pop(k, None)
    if e.type == "llm_response":
        d.pop("model", None)
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _openai_proj(e):
    if e.type not in ("run_start", "user_input", "tool_call", "tool_result", "final_answer", "run_end", "llm_response"):
        return None
    if e.type == "llm_response" and not e.content:
        return None
    if e.type == "final_answer" and e.content is None:
        return None  # nothing to render as an assistant message
    if e.type == "run_end" and not e.outcome:
        return None  # chat transcripts have no explicit end marker
    d = {"type": e.type}
    keep = {"run_start": ("agent", "task_type"), "user_input": ("content",),
            "llm_response": ("content",), "tool_call": ("tool", "call_id", "args"),
            "tool_result": ("tool", "call_id", "result", "error"), "final_answer": ("content",),
            "run_end": ("outcome",)}[e.type]
    for k in keep:
        d[k] = getattr(e, k)
    if e.type == "tool_result":
        d["failed"] = e.status != "ok"
    return d


def _roundtrip(events, fmt):
    text = export(events, fmt)
    back, rep = load_any(text.encode("utf-8"), filename=f"rt{EXTENSIONS[fmt]}")
    return back, rep


@pytest.mark.parametrize("fixture", ["arl_basic.jsonl", "arl_messy.jsonl", "openai_runs.jsonl", "pylog_basic.log",
                                     "otel_otlp.json", "csv_basic.csv"])
@pytest.mark.parametrize("fmt", ["arl", "csv", "pylog", "otel", "otlp", "openai"])
def test_roundtrip(fixture, fmt):
    events, _ = load_any(FX / fixture)
    back, rep = _roundtrip(events, fmt)
    assert rep["format"] == ("otel" if fmt == "otlp" else fmt)
    a, b = by_run(events), by_run(back)
    assert list(a) == list(b)
    for rid in a:
        if fmt in LOSSLESS:
            drop_ts = fmt == "pylog"
            got = [_full(e, drop_ts) for e in b[rid]]
            want = [_full(e, drop_ts) for e in a[rid]]
            assert got == want
            if fmt == "pylog":
                assert [e.ts for e in b[rid] if a[rid][e.seq].ts] == [e.ts for e in a[rid] if e.ts]
        elif fmt in ("otel", "otlp"):
            assert Counter(map(_otel_proj, a[rid])) == Counter(map(_otel_proj, b[rid]))
            assert [e.call_id for e in a[rid] if e.type == "tool_call"] == \
                   [e.call_id for e in b[rid] if e.type == "tool_call"]
        else:
            pa = [p for p in map(_openai_proj, a[rid]) if p]
            pb = [p for p in map(_openai_proj, b[rid]) if p]
            assert pa == pb


@pytest.mark.parametrize("fmt", ["arl", "csv", "pylog", "otel"])
def test_roundtrip_full_sample(skydesk, fmt):
    ev, src_rep = skydesk
    back, rep = _roundtrip(ev, fmt)
    assert rep["n_runs"] == src_rep["n_runs"] and rep["n_events"] == len(ev)
    if fmt == "arl":
        assert [e.to_dict() for e in back] == [e.to_dict() for e in ev]
    elif fmt == "csv":  # CSV cannot tell "" from missing, nor 2825 from 2825.0
        def norm(e):
            return {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                    for k, v in e.to_dict().items() if v != ""}
        assert [norm(e) for e in back] == [norm(e) for e in ev]
    elif fmt == "pylog":
        assert [_full(e, True) for e in back] == [_full(e, True) for e in ev]
    else:
        a, b = by_run(ev), by_run(back)
        for rid in a:
            assert Counter(map(_otel_proj, a[rid])) == Counter(map(_otel_proj, b[rid])), rid


def test_export_rejects_unknown_format():
    with pytest.raises(ValueError):
        export([], "yaml")


# ============================================================================ ARL logger
def test_arl_logger_end_to_end(tmp_path):
    p = tmp_path / "logs" / "runs.jsonl"
    log = arc.ARLLogger(p, run_id="lg-1", agent="refund-agent", model="gpt-5-mini", task_type="refund")
    log.user_input("Refund booking XYZ9")
    log.llm_response("Checking.", tokens_in=100, tokens_out=5)

    @log.tool
    def verify_customer(booking_ref: str) -> bool:
        return True

    @log.tool(name="issue_refund")
    def refund(booking_ref, amount=0):
        raise PermissionError("PolicyError: amount exceeds fare")

    assert verify_customer("XYZ9") is True
    with pytest.raises(PermissionError):
        refund("XYZ9", amount=420)
    log.final_answer("Your refund has been issued.", claimed_success=True, claimed_actions=["issue_refund"])
    log.end(outcome={"verified_success": 0, "source": "test"})
    assert log.end() is None  # idempotent

    with arc.ARLLogger(p, run_id="lg-2", agent="refund-agent") as log2:
        cid = log2.tool_call("lookup", {"id": 1})
        log2.tool_result("lookup", {"found": True}, call_id=cid)

    ev, rep = load_any(p)
    assert rep["format"] == "arl" and rep["n_runs"] == 2 and rep["warnings"] == []
    r = by_run(ev)["lg-1"]
    assert [e.type for e in r] == ["run_start", "user_input", "llm_response", "tool_call", "tool_result",
                                   "tool_call", "tool_result", "final_answer", "run_end"]
    assert r[3].args == {"booking_ref": "XYZ9"} and r[4].result is True and r[4].status == "ok"
    assert r[5].args == {"booking_ref": "XYZ9", "amount": 420}
    assert r[6].status == "refused" and r[6].error.startswith("PermissionError")
    assert r[6].call_id == r[5].call_id and r[6].latency_ms is not None
    assert all(e.ts for e in r)
    assert by_run(ev)["lg-2"][-1].type == "run_end"


def test_arl_logger_is_dependency_free():
    src = (ROOT / "arc" / "arl_logger.py").read_text()
    imports = [l.split()[1] for l in src.splitlines() if l.startswith(("import ", "from "))]
    stdlib = {"__future__", "functools", "inspect", "itertools", "json", "os", "threading", "time", "uuid",
              "datetime", "typing"}
    assert set(imports) <= stdlib, imports
