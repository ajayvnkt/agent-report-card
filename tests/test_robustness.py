"""Adversarial / robustness regression tests: hostile inputs never crash the pipeline or the app,
and every bug found in the adversarial QA pass stays fixed.

Bugs covered (one test each, at least):
  1. OpenAI tool messages whose content is a list of parts hid tool errors ("PolicyError" inside
     [{"type":"text","text":...}] was read as a successful result).
  2. A tool_result with no logged tool_call was ignored, so its failure / policy breach was invisible.
  3. call_id mismatches: an unknown call_id left the call "unanswered"; a call_id that pointed at a
     call of ANOTHER tool paired the two (verify_customer got issue_refund's error).
  4. Replies that claimed an action that never ran were missed when the log had no claimed_actions
     (e.g. "I've also added a USD 50 voucher" with no issue_voucher call).
  5. Policy selectors with a bad regex ("re:(") passed loading and crashed the analysis; non-mapping
     require_before / caps raised AttributeError; misspelt keys were silently ignored; a missing
     policy file path was parsed as YAML text.
  6. Python tracebacks after an agent_run line were counted as "unreadable lines".
  7. "Cost per likely-successful run" exploded when (almost) no run was expected to succeed.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from arc.analyze import claimed_tools_from_text
from arc.detect import UnrecognizedFormatError, load_any
from arc.pipeline import _usd, cost_block, run
from arc.policy import load_policy, policy_warnings

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_logs"
SKY_POLICY = str(SAMPLES / "skydesk_policy.yaml")
APP = str(ROOT / "app.py")


def _jsonl(rows) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()


def _arl(run_id, *events):
    return [{"run_id": run_id, "seq": i, **e} for i, e in enumerate(events)]


# --------------------------------------------------------------------------- never crash

UNREADABLE = {
    "empty.jsonl": b"",
    "whitespace.jsonl": b"   \n\n\t \n",
    "garbage.log": bytes(random.Random(1).randrange(256) for _ in range(5000)),
    "no_agent_run.log": b"2026-01-01 INFO server started\n2026-01-01 WARN disk low\nhello\n",
    "not_a_run.json": b'{"hello": "world", "items": [1, 2, 3]}',
    "scalars.json": b'[1, 2, "x", null]',
    "header_only.csv": b"run_id,seq,type\n",
}


@pytest.mark.parametrize("name", sorted(UNREADABLE))
def test_unreadable_files_raise_a_clear_error(name):
    with pytest.raises(UnrecognizedFormatError) as ei:
        run(UNREADABLE[name], filename=name)
    msg = str(ei.value)
    assert name in msg and "Accepted formats" in msg


ODD_BUT_READABLE = {
    "arl_missing_fields.jsonl": _jsonl([
        {"seq": 0, "type": "run_start"},
        {"run_id": "x", "type": "tool_call", "tool": "t"},
        {"run_id": "x", "seq": 3},
        {"run_id": "x", "seq": 4, "type": "bogus"},
        {"run_id": "x", "seq": 5, "type": "final_answer", "content": "All done!"},
    ]),
    "dup_and_bad_seq.jsonl": _jsonl([
        {"run_id": "d", "seq": 0, "type": "run_start"},
        {"run_id": "d", "seq": 1, "type": "tool_call", "tool": "t", "call_id": "1"},
        {"run_id": "d", "seq": 1, "type": "tool_result", "tool": "t", "call_id": "1", "result": "ok"},
        {"run_id": "d", "seq": 1, "type": "final_answer", "content": "done"},
        {"run_id": "d", "seq": "abc", "type": "run_end"},
        {"run_id": "d", "seq": -5, "type": "run_end"},
        {"run_id": "d", "seq": 1e30, "type": "run_end"},
    ]),
    "unicode.jsonl": _jsonl(_arl(
        "ü-🚀-مرحبا",
        {"type": "run_start", "agent": "وكيل", "task_type": "استرداد 💸"},
        {"type": "user_input", "content": "‮RTL טקסט \u0000 nul $x$ | **b**"},
        {"type": "tool_call", "tool": "工具<script>alert(1)</script>", "call_id": "1", "args": {"k": "🤖"}},
        {"type": "tool_result", "tool": "工具<script>alert(1)</script>", "call_id": "1", "result": "| pipe `x`"},
        {"type": "final_answer", "content": "تم ✅ Done | `x`"},
        {"type": "run_end", "outcome": {"verified_success": "yes"}},
    )),
    "extra_cols.csv": b'run_id,seq,type,tool,args,result,weird,claimed_success\nr1,0,run_start,,,,x,\n'
                      b'r1,1,tool_call,t,"{""a"":1}",,y,\nr1,2,tool_result,t,,"{bad json",z,\nr1,3,final_answer,,,,,maybe\n',
    "missing_cols.csv": b"run_id,type\nr1,run_start\nr1,final_answer\n",
    "ragged.csv": b'run_id,seq,type,content\nr1,0,run_start\nr1,1,final_answer,"done, all good",extra,cols\n"unterminated\n',
    "otlp_cycle.json": json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [
        {"traceId": "C", "spanId": "A", "parentSpanId": "B", "name": "execute_tool a",
         "attributes": [{"key": "gen_ai.tool.name", "value": {"stringValue": "a"}}]},
        {"traceId": "C", "spanId": "B", "parentSpanId": "A", "name": "execute_tool b",
         "attributes": [{"key": "gen_ai.tool.name", "value": {"stringValue": "b"}}]}]}]}]}).encode(),
}


@pytest.mark.parametrize("name", sorted(ODD_BUT_READABLE))
def test_odd_inputs_are_analysed_with_sane_numbers(name):
    rep = run(ODD_BUT_READABLE[name], filename=name)
    assert rep.overview["n_runs"] >= 1
    for r in rep.runs:
        assert 0.0 <= r.success_probability <= 1.0
        assert r.cost_usd >= 0
    json.loads(rep.to_json())
    assert rep.to_markdown().startswith("# Agent Report Card")


def test_otlp_int_bool_attrs_and_nested_parents():
    def a(k, v):
        key = "boolValue" if isinstance(v, bool) else "intValue" if isinstance(v, int) else "stringValue"
        return {"key": k, "value": {key: str(v) if key == "intValue" else v}}
    spans = [
        {"traceId": "T1", "spanId": "S0", "name": "invoke_agent bot",
         "attributes": [a("gen_ai.agent.name", "bot"), a("arc.claimed_success", True),
                        a("arc.final_answer", "Refund issued"), a("arc.outcome.verified_success", 0)]},
        {"traceId": "T1", "spanId": "S1", "parentSpanId": "S0", "name": "step", "attributes": []},
        {"traceId": "T1", "spanId": "S2", "parentSpanId": "S1", "name": "chat m",
         "attributes": [a("gen_ai.operation.name", "chat"), a("gen_ai.usage.input_tokens", 100)]},
        {"traceId": "T1", "spanId": "S3", "parentSpanId": "S2", "name": "execute_tool issue_refund",
         "status": {"code": 2, "message": "boom"},
         "attributes": [a("gen_ai.tool.name", "issue_refund"), a("gen_ai.tool.call.id", "c1")]},
    ]
    rep = run(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}).encode(), filename="o.json")
    r = rep.runs[0]
    assert r.claimed_success and r.outcome == 0 and r.n_error == 1
    assert "FALSE_SUCCESS_AFTER_FAILURE" in r.finding_codes


def test_50k_events_under_20_seconds():
    rng = random.Random(0)
    rows = []
    for k in range(5000):
        ev = [{"type": "run_start", "task_type": rng.choice("abc"), "agent": "big"},
              {"type": "user_input", "content": "please refund"}]
        for c in range(3):
            tool = rng.choice(["verify_customer", "issue_refund", "lookup"])
            ev += [{"type": "tool_call", "tool": tool, "call_id": f"c{c}"},
                   {"type": "tool_result", "tool": tool, "call_id": f"c{c}", "status": rng.choice(["ok", "error"])}]
        ev += [{"type": "final_answer", "content": "Done.", "claimed_success": True},
               {"type": "run_end", "outcome": {"verified_success": rng.randint(0, 1)}}]
        rows += _arl(f"big-{k}", *ev)
    assert len(rows) == 50_000
    t = time.time()
    rep = run(_jsonl(rows), filename="big.jsonl")
    rep.to_json()
    assert time.time() - t < 20
    assert rep.overview["n_runs"] == 5000 and rep.source["n_events"] == 50_000


# --------------------------------------------------------------------------- bug 1: OpenAI content parts

def test_openai_tool_content_parts_and_parallel_calls():
    doc = [{"run_id": "p1", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "Refund"}, {"type": "image_url", "image_url": {"url": "x"}}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "verify_customer", "arguments": "{\"x\":1}"}},
            {"id": "b", "type": "function", "function": {"name": "issue_refund", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "PolicyError: nope"}]},
        {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "{\"verified\": true}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Your refund has been issued."}]}],
        "outcome": {"verified_success": 0}}]
    events, _ = load_any(json.dumps(doc).encode(), filename="p.json")
    res = {e.tool: e for e in events if e.type == "tool_result"}
    assert res["issue_refund"].status == "refused" and "PolicyError" in res["issue_refund"].error
    assert res["verify_customer"].result == {"verified": True}
    r = run(json.dumps(doc).encode(), filename="p.json").runs[0]
    assert "FALSE_SUCCESS_AFTER_FAILURE" in r.finding_codes
    assert r.success_probability < 0.5


def test_openai_json_list_result_is_not_mistaken_for_parts():
    doc = {"run_id": "q", "messages": [
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "list_items", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": [{"id": 1}, {"id": 2}]},
        {"role": "assistant", "content": "Found 2 items."}]}
    events, _ = load_any(json.dumps(doc).encode(), filename="q.jsonl")
    assert next(e for e in events if e.type == "tool_result").result == [{"id": 1}, {"id": 2}]


# --------------------------------------------------------------------------- bug 2: orphan results

def test_tool_result_without_call_still_counts():
    rows = _arl("o",
                {"type": "run_start"},
                {"type": "tool_result", "tool": "issue_refund", "status": "error", "error": "boom"},
                {"type": "final_answer", "content": "Refund issued.", "claimed_success": True})
    r = run(_jsonl(rows), filename="o.jsonl").runs[0]
    assert r.n_orphan_results == 1 and r.n_error == 1
    assert "FALSE_SUCCESS_AFTER_FAILURE" in r.finding_codes
    assert r.success_probability < 0.5


# --------------------------------------------------------------------------- bug 3: call_id mismatches

def test_call_id_mismatch_repaired_by_tool_name():
    rows = _arl("c",
                {"type": "tool_call", "tool": "verify_customer", "call_id": "A"},
                {"type": "tool_result", "tool": "verify_customer", "call_id": "ZZZ", "status": "ok", "result": True},
                {"type": "tool_call", "tool": "issue_refund", "call_id": "B", "args": {"amount": 10}},
                {"type": "tool_result", "tool": "issue_refund", "call_id": "A", "status": "error", "error": "x"},
                {"type": "final_answer", "content": "Done", "claimed_success": True})
    rep = run(_jsonl(rows), filename="c.jsonl")
    assert any("did not match an open tool_call" in w for w in rep.source["warnings"])
    r = rep.runs[0]
    st = {c.tool: c.status for c in r.tool_calls}
    assert st == {"verify_customer": "ok", "issue_refund": "error"}
    assert "UNANSWERED_CALL" not in r.finding_codes and "PREREQ_MISSING" not in r.finding_codes
    assert "FALSE_SUCCESS_AFTER_FAILURE" in r.finding_codes


def test_in_memory_events_pair_the_same_way():
    """analyze_events on raw dicts (no adapter normalisation) must not cross-pair tools either."""
    from arc.pipeline import analyze_events
    rows = _arl("m",
                {"type": "tool_call", "tool": "verify_customer", "call_id": "A"},
                {"type": "tool_call", "tool": "issue_refund", "call_id": "B"},
                {"type": "tool_result", "tool": "issue_refund", "call_id": "A", "status": "error"},
                {"type": "tool_result", "tool": "verify_customer", "call_id": "A", "status": "ok"})
    r = analyze_events(rows).runs[0]
    assert {c.tool: c.status for c in r.tool_calls} == {"verify_customer": "ok", "issue_refund": "error"}


# --------------------------------------------------------------------------- bug 4: claims read from text

TOOLS = ["issue_voucher", "issue_refund", "send_itinerary", "rebook_flight", "update_contact",
         "lookup_booking", "verify_customer", "escalate_to_human"]


@pytest.mark.parametrize("text,expected", [
    ("As a goodwill gesture I've also added a USD 50 voucher (code X) to your account.", {"issue_voucher"}),
    ("Your refund of USD 177.45 has been issued (ref RF-1).", {"issue_refund"}),
    ("All set - you're rebooked on SD1 (seat 1A), and the updated itinerary has been emailed to a@b.c.",
     {"rebook_flight", "send_itinerary"}),
    ("Your email on booking 8UHLGQ has been updated to x@y.z.", {"update_contact"}),
    ("I've escalated your case; they'll contact you within 24 hours.", set()),
    ("I could not issue the refund, the system returned an error.", set()),
    ("A specialist will issue the voucher once approved.", set()),
    ("Would you like me to issue a refund?", set()),
    ("Your booking is confirmed and your identity was verified.", set()),
])
def test_claimed_tools_from_text(text, expected):
    got = set().union(*[set(h) for _, h in claimed_tools_from_text(text, TOOLS)]) if text else set()
    assert expected <= got
    if not expected:
        assert got == set()


def test_unlogged_claim_of_unexecuted_action_is_flagged():
    rows = _arl("v",
                {"type": "run_start", "task_type": "itinerary_resend"},
                {"type": "tool_call", "tool": "send_itinerary", "call_id": "1"},
                {"type": "tool_result", "tool": "send_itinerary", "call_id": "1", "status": "ok", "result": {"sent": 1}},
                {"type": "final_answer", "content": "Itinerary sent. I've also added a USD 50 voucher to your account."})
    other = _arl("w", {"type": "tool_call", "tool": "issue_voucher", "call_id": "1"},
                 {"type": "tool_result", "tool": "issue_voucher", "call_id": "1", "status": "ok"},
                 {"type": "final_answer", "content": "I've issued a voucher."})
    rep = run(_jsonl(rows + other), filename="v.jsonl", policy=SKY_POLICY)
    v, w = rep.run("v"), rep.run("w")
    f = [x for x in v.findings if x.code == "CLAIMED_NOT_DONE"]
    assert f and f[0].tool == "issue_voucher" and f[0].severity == "critical"
    assert "read from the reply text" in f[0].message
    assert "CLAIMED_NOT_DONE" not in w.finding_codes          # it really did issue the voucher


def test_logged_claimed_actions_take_precedence_over_text():
    rows = _arl("x", {"type": "final_answer", "content": "I've also added a voucher.",
                      "claimed_success": True, "claimed_actions": []})
    assert "CLAIMED_NOT_DONE" not in run(_jsonl(rows), filename="x.jsonl", policy=SKY_POLICY).runs[0].finding_codes


def _planted(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e["type"] == "run_end":
            out[e["run_id"]] = (e.get("meta") or {}).get("planted_failure")
    return out


@pytest.mark.parametrize("fname", ["skydesk_arl.jsonl", "skydesk_openai_messages.jsonl"])
def test_skydesk_hallucinated_actions_all_caught_without_false_alarms(fname, bulk_samples):
    """Every planted 'claimed action never executed' run is flagged in both formats (the OpenAI
    format never carries claimed_actions, ARL omits them on ~15% of runs), and no clean run gets a
    high/critical finding."""
    planted = _planted(bulk_samples / "skydesk_arl.jsonl")
    rep = run(bulk_samples / fname, policy=SKY_POLICY)
    for r in rep.runs:
        kind = planted.get(r.run_id)
        serious = any(f.severity in ("critical", "high") for f in r.findings)
        if kind == "claimed_action_never_executed":
            assert "CLAIMED_NOT_DONE" in r.finding_codes, r.run_id
        if kind in ("clean_success", "correct_escalation"):
            assert not serious, (r.run_id, r.finding_codes)


def test_demo_striking_run_is_genuinely_claimed_but_failed():
    rep = run(SAMPLES / "demo_agent_runs.log", policy=SKY_POLICY)
    worst = min((r for r in rep.runs if r.claimed_success), key=lambda r: r.success_probability)
    assert worst.run_id == "sky-0904-0028"
    assert worst.outcome == 0 and worst.success_probability < 0.05
    assert [c.status for c in worst.tool_calls if c.tool == "update_contact"] == ["error"] * 3
    assert {"FALSE_SUCCESS_AFTER_FAILURE", "CLAIMED_NOT_DONE", "RETRY_LOOP"} <= set(worst.finding_codes)


# --------------------------------------------------------------------------- bug 5: policy validation

@pytest.mark.parametrize("text,needle", [
    ('forbidden_tools: ["re:("]', "not a valid regex"),
    ('require_before: {"re:[": [verify]}', "not a valid regex"),
    ("require_before: [a, b]", "require_before must be a mapping"),
    ("caps: [1, 2]", "caps must be a mapping"),
    ("caps: {issue_refund: {arg: amount, max: lots}}", "must be a number"),
    ("max_tool_retries: abc", "whole number"),
    ('require_before: {"@monee": [x]}', "unknown group"),
    ("rules: [\n  - this: is: not: valid\n", "could not parse policy YAML"),
    ("- just\n- a list\n", "must be a mapping"),
    ("nope/missing_policy.yaml", "policy file not found"),
])
def test_bad_policies_fail_at_load_with_clear_message(text, needle):
    with pytest.raises(ValueError, match=needle):
        load_policy(text)


def test_policy_unknown_keys_warn_and_limits_are_clamped():
    pol = load_policy("require_befor: {issue_refund: [verify_customer]}\nmax_tool_retries: -3\n"
                      "max_failed_verifications: 0\nversion: 2\n")
    w = policy_warnings(pol)
    assert len(w) == 1 and "require_befor" in w[0] and "version" not in w[0]
    assert pol.max_tool_retries == 0 and pol.max_failed_verifications == 1
    assert policy_warnings(load_policy(SKY_POLICY)) == []
    rows = _arl("p", {"type": "final_answer", "content": "ok"})
    assert any("require_befor" in n for n in run(_jsonl(rows), filename="p.jsonl",
                                                  policy="require_befor: {a: [b]}").notes)


# --------------------------------------------------------------------------- bug 6: python tracebacks

def test_python_log_tracebacks_are_attached_not_skipped():
    log = (
        '2026-01-01T00:00:00Z INFO agent_run run_id=t1 seq=0 type=run_start task_type=refund\n'
        '2026-01-01T00:00:01Z INFO agent_run run_id=t1 seq=1 type=tool_call tool=issue_refund call_id=c1 args={"amount": 5}\n'
        '2026-01-01T00:00:02Z ERROR agent_run run_id=t1 seq=2 type=tool_result tool=issue_refund call_id=c1 status=error error="boom"\n'
        'Traceback (most recent call last):\n'
        '  File "x.py", line 1, in <module>\n'
        '    raise ValueError("boom")\n'
        'ValueError: boom\n'
        '2026-01-01T00:00:03Z INFO uvicorn something unrelated\n'
        '2026-01-01T00:00:04Z INFO agent_run run_id=t1 seq=3 type=final_answer content="Refund done" claimed_success=true\n'
    ).encode()
    events, src = load_any(log, filename="t.log")
    assert src["skipped_lines"] == 1                       # only the unrelated uvicorn line
    res = next(e for e in events if e.type == "tool_result")
    assert "ValueError: boom" in res.meta["traceback"]
    assert any("traceback" in w for w in src["warnings"])


def test_mixed_format_lines_get_a_helpful_skip_reason():
    rows = _arl("m1", {"type": "run_start"}, {"type": "final_answer", "content": "Done."})
    rows.append({"run_id": "m2", "messages": [{"role": "user", "content": "hi"}]})
    _, src = load_any(_jsonl(rows), filename="mixed.jsonl")
    assert src["skipped_lines"] == 1
    assert "OpenAI chat-messages run" in src["skipped_examples"][0]


# --------------------------------------------------------------------------- bug 7: cost sanity

def test_cost_per_success_not_shown_when_nothing_is_expected_to_succeed():
    from arc.analyze import RunSummary
    rs = [RunSummary(run_id=str(i), cost_usd=0.07, success_probability=1e-6) for i in range(3)]
    assert cost_block(rs)["per_expected_success_usd"] is None
    rs = [RunSummary(run_id=str(i), cost_usd=0.07, success_probability=0.5) for i in range(3)]
    assert cost_block(rs)["per_expected_success_usd"] == pytest.approx(0.14)
    assert _usd(0.00001) == "<$0.0001" and _usd(0.0) == "$0.0000" and _usd(None) == "n/a"


# --------------------------------------------------------------------------- the app

def _app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    return at


@pytest.mark.parametrize("name", ["empty.jsonl", "garbage.log", "not_a_run.json", "header_only.csv",
                                  "no_agent_run.log"])
def test_app_hostile_upload_shows_friendly_error(name):
    at = _app()
    at.file_uploader(key="uploads").set_value((name, UNREADABLE[name], "application/octet-stream")).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("couldn't read this file" in e.value for e in at.error)
    assert at.button(key="run_analysis").disabled


@pytest.mark.parametrize("name", ["unicode.jsonl", "dup_and_bad_seq.jsonl", "ragged.csv", "otlp_cycle.json"])
def test_app_odd_upload_analyses_and_opens_run_detail(name):
    at = _app()
    at.file_uploader(key="uploads").set_value((name, ODD_BUT_READABLE[name], "application/octet-stream")).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Detected" in s.value for s in at.success)
    at.button(key="run_analysis").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Runs analysed" in {m.label for m in at.metric}
    at.session_state["results_tab"] = "Run detail"
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_bad_policy_yaml_never_reaches_the_analysis():
    at = _app()
    at.selectbox(key="sample").set_value("SkyDesk demo · 16 runs · Python log (demo_agent_runs.log) — start here").run()
    at.radio(key="policy_choice").set_value("Edit YAML").run()
    for bad in ['forbidden_tools: ["re:("]\n', "require_before: [a, b]\n"]:
        at.text_area(key="policy_yaml").set_value(bad).run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("could not be read" in w.value for w in at.warning)
        assert at.button(key="run_analysis").disabled
    at.text_area(key="policy_yaml").set_value("name: typo\nrequire_befor: {issue_refund: [verify_customer]}\n").run()
    assert any("Unknown policy key" in w.value for w in at.warning)
    at.button(key="run_analysis").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Runs analysed" in {m.label for m in at.metric}


def test_app_cost_metric_is_sane_when_every_run_fails():
    rows = _arl("f", {"type": "tool_call", "tool": "issue_refund", "call_id": "1"},
                {"type": "tool_result", "tool": "issue_refund", "call_id": "1", "status": "error", "error": "boom"},
                {"type": "llm_response", "tokens_in": 100000, "tokens_out": 100000, "model": "gpt-4.1"},
                {"type": "final_answer", "content": "Refund issued.", "claimed_success": True})
    at = _app()
    at.file_uploader(key="uploads").set_value(("f.jsonl", _jsonl(rows), "application/jsonl")).run()
    at.button(key="run_analysis").click().run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Est. cost per likely-successful run"] == "–"


def test_cli_reports_bad_policy_without_traceback(capsys):
    from arc.__main__ import main
    assert main(["analyze", str(SAMPLES / "demo_agent_runs.jsonl"), "--policy", "nope/missing.yaml"]) == 2
    err = capsys.readouterr().err
    assert "policy file not found" in err and "Traceback" not in err
