#!/usr/bin/env python3
"""
validate_sample_logs.py -- check the synthetic logs with plain json/csv parsing:

* the curated demo set in sample_logs/ (always): demo_agent_runs.jsonl and demo_agent_runs.log hold
  the same runs, every scenario is present, the policy regexes fire on exactly the intended runs;
* the bulk set written by `make samples` (when a folder is given, default build/samples/ if it
  exists): all formats describe the same runs; per-version summary stats.

Usage: python scripts/validate_sample_logs.py [BULK_DIR]

Stdlib only (uses PyYAML for the policy file if it is installed). Independent of arc/.
Exit code 1 on any failed check.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "sample_logs"
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "samples"
FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def load_arl(path):
    runs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            e = json.loads(ln)
            runs[e["run_id"]].append(e)
    return runs


def check_arl(runs, name, labeled=True):
    for rid, ev in runs.items():
        check([e["seq"] for e in ev] == list(range(len(ev))), f"{name} {rid}: seq not 0..n-1")
        check(ev[0]["type"] == "run_start" and ev[-1]["type"] == "run_end", f"{name} {rid}: bad start/end")
        calls = {e["call_id"] for e in ev if e["type"] == "tool_call"}
        for e in ev:
            if e["type"] == "tool_result":
                check(e["call_id"] in calls, f"{name} {rid}: orphan tool_result")
                check(e["status"] in ("ok", "error", "refused", "timeout"), f"{name} {rid}: bad status")
        has = "outcome" in ev[-1]
        check(has == labeled, f"{name} {rid}: outcome presence {has} != {labeled}")


def parse_kv(rest):
    """Parse 'k=v k={json} k="json str"' into a dict."""
    dec = json.JSONDecoder()
    out, i = {}, 0
    while i < len(rest):
        while i < len(rest) and rest[i] == " ":
            i += 1
        if i >= len(rest):
            break
        eq = rest.index("=", i)
        key = rest[i:eq]
        j = eq + 1
        if rest[j] in "{[\"":
            val, end = dec.raw_decode(rest, j)
            i = end
        else:
            end = rest.find(" ", j)
            end = len(rest) if end < 0 else end
            val = rest[j:end]
            i = end
        out[key] = val
    return out


LINE_RE = re.compile(r"^(\S+) (DEBUG|INFO|WARNING|ERROR|CRITICAL) agent_run (run_id=.*)$")


def load_pylog(path):
    runs, other = defaultdict(list), 0
    with open(path, encoding="utf-8") as f:
        for ln in f:
            m = LINE_RE.match(ln.rstrip("\n"))
            if not m:
                other += 1
                continue
            kv = parse_kv(m.group(3))
            kv["ts"] = m.group(1)
            runs[kv["run_id"]].append(kv)
    return runs, other


DEMO_PLANTED = {"clean_success", "correct_escalation", "approval_escalation", "missed_escalation",
                "unnecessary_escalation", "false_success_claim", "claimed_action_never_executed",
                "money_before_verification", "cap_exceeded", "obeyed_injection", "retry_loop", "timeout",
                "unanswered_last_call", "missing_final_answer"}


def validate_demo():
    demo = load_arl(DEMO_DIR / "demo_agent_runs.jsonl")
    check_arl(demo, "demo")
    planted = Counter(ev[-1]["meta"]["planted_failure"] for ev in demo.values())
    check(set(planted) == DEMO_PLANTED, f"demo: scenarios {sorted(planted)}")
    check(14 <= len(demo) <= 16, f"demo: {len(demo)} runs")
    dpy, dother = load_pylog(DEMO_DIR / "demo_agent_runs.log")
    check(list(dpy) == list(demo) and [len(v) for v in dpy.values()] == [len(v) for v in demo.values()],
          "demo python log describes other runs than the ARL file")
    for rid, ev in demo.items():
        check([e["type"] for e in ev] == [e["type"] for e in dpy[rid]], f"demo {rid}: event types differ")
    n_ok = sum(ev[-1]["outcome"]["verified_success"] for ev in demo.values())
    print(f"demo_agent_runs.jsonl / .log: {len(demo)} runs, {sum(map(len, demo.values()))} events, "
          f"{n_ok} verified successes, {dother} unrelated log lines")
    try:
        import yaml
    except ImportError:
        print("demo policy check skipped (PyYAML not installed)")
        return
    pol = yaml.safe_load(open(DEMO_DIR / "skydesk_policy.yaml", encoding="utf-8"))
    inj = [re.compile(p) for p in pol["injection_patterns"]]
    esc = [re.compile(p) for p in pol["escalate_if_text_matches"]]
    for rid, ev in demo.items():
        text = "\n".join(e["content"] for e in ev if e["type"] == "user_input")
        needs_human = ev[-1]["meta"]["planted_failure"] in ("correct_escalation", "missed_escalation",
                                                            "approval_escalation") \
            or (ev[0]["task_type"] == "complaint_escalation")
        check(needs_human == any(p.search(text) for p in esc), f"demo {rid}: escalation regex")
        check((ev[0]["task_type"] == "injected_instruction") == any(p.search(text) for p in inj),
              f"demo {rid}: injection regex")
    print("skydesk_policy.yaml: injection/escalation regexes match exactly the intended demo runs")


def validate_bulk():
    arl = load_arl(D / "skydesk_arl.jsonl")
    n_ev = sum(len(v) for v in arl.values())
    check_arl(arl, "arl")
    truth = {rid: ev[-1]["outcome"]["verified_success"] for rid, ev in arl.items()}
    agent = {rid: ev[0]["agent"] for rid, ev in arl.items()}
    ncalls = {rid: sum(e["type"] == "tool_call" for e in ev) for rid, ev in arl.items()}
    print(f"skydesk_arl.jsonl: {len(arl)} runs, {n_ev} events")

    un = load_arl(D / "skydesk_unlabeled_arl.jsonl")
    check_arl(un, "unlabeled", labeled=False)
    check(sum(len(v) for v in un.values()) == n_ev and set(un) == set(arl), "unlabeled: counts differ")
    check(not any("meta" in ev[-1] for ev in un.values()), "unlabeled: meta (planted label) leaked")
    print(f"skydesk_unlabeled_arl.jsonl: {len(un)} runs, no outcomes")

    # OpenAI messages
    n = 0
    with open(D / "skydesk_openai_messages.jsonl", encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            n += 1
            rid = o["run_id"]
            check(o["outcome"]["verified_success"] == truth[rid], f"openai {rid}: outcome mismatch")
            ids, k = set(), 0
            for m in o["messages"]:
                check(m["role"] in ("system", "user", "assistant", "tool"), f"openai {rid}: role")
                for tc in m.get("tool_calls", []):
                    json.loads(tc["function"]["arguments"])
                    ids.add(tc["id"])
                    k += 1
                if m["role"] == "tool":
                    check(m["tool_call_id"] in ids, f"openai {rid}: orphan tool msg")
            check(k == ncalls[rid], f"openai {rid}: tool_call count {k} != {ncalls[rid]}")
    check(n == len(arl), "openai: run count")
    print(f"skydesk_openai_messages.jsonl: {n} runs")

    # Python log
    py, other = load_pylog(D / "skydesk_python.log")
    py_ev = sum(len(v) for v in py.values())
    check(py_ev == n_ev and set(py) == set(arl), f"python log: {py_ev} agent_run lines vs {n_ev} events")
    for rid, ev in py.items():
        check(json.loads(json.dumps(ev[-1]["outcome"]))["verified_success"] == truth[rid], f"pylog {rid}: outcome")
        check(sum(e["type"] == "tool_call" for e in ev) == ncalls[rid], f"pylog {rid}: tool calls")
    total = py_ev + other
    print(f"skydesk_python.log: {py_ev} agent_run lines, {other} unrelated lines ({other / total:.1%})")

    # OTel flat
    spans = json.load(open(D / "skydesk_otel_spans.json", encoding="utf-8"))
    traces = defaultdict(list)
    for s in spans:
        traces[s["trace_id"]].append(s)
    check(len(traces) == len(arl), "otel: trace count")
    for tid, ss in traces.items():
        roots = [s for s in ss if s["parent_span_id"] is None]
        check(len(roots) == 1 and roots[0]["name"].startswith("invoke_agent "), f"otel {tid}: root")
        a = roots[0]["attributes"]
        rid = a["arc.run_id"]
        check(a["arc.outcome.verified_success"] == truth[rid], f"otel {rid}: outcome")
        tools = [s for s in ss if s["name"].startswith("execute_tool ")]
        check(len(tools) == ncalls[rid], f"otel {rid}: tool spans")
        for s in tools:
            json.loads(s["attributes"]["gen_ai.tool.call.arguments"])
            if "gen_ai.tool.call.result" in s["attributes"]:
                json.loads(s["attributes"]["gen_ai.tool.call.result"])
    print(f"skydesk_otel_spans.json: {len(traces)} traces, {len(spans)} spans")

    # OTLP
    otlp = json.load(open(D / "skydesk_otlp.json", encoding="utf-8"))
    otraces, nsp = defaultdict(list), 0
    for rs in otlp["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for s in ss["spans"]:
                otraces[s["traceId"]].append(s)
                nsp += 1
                for at in s["attributes"]:
                    check(len(at["value"]) == 1, "otlp: attribute value shape")
    for tid, ss in otraces.items():
        root = next(s for s in ss if s["parentSpanId"] == "")
        a = {x["key"]: next(iter(x["value"].values())) for x in root["attributes"]}
        check(int(a["arc.outcome.verified_success"]) == truth[a["arc.run_id"]], "otlp: outcome")
    check(len(otraces) == 60, f"otlp: {len(otraces)} traces != 60")
    print(f"skydesk_otlp.json: {len(otraces)} traces, {nsp} spans")

    # CSV
    rows = list(csv.DictReader(open(D / "skydesk_events.csv", encoding="utf-8", newline="")))
    check(len(rows) == n_ev, "csv: row count")
    for r in rows:
        for c in ("args", "result", "outcome", "meta", "claimed_actions"):
            if r[c]:
                json.loads(r[c])
    print(f"skydesk_events.csv: {len(rows)} rows")

    # policy
    try:
        import yaml
        pol = yaml.safe_load(open(D / "skydesk_policy.yaml", encoding="utf-8"))
        for k in ("money_tools", "require_before", "forbidden_tools", "max_tool_retries", "injection_patterns",
                  "escalate_if_text_matches", "caps"):
            check(k in pol, f"policy: missing {k}")
        inj = [re.compile(p) for p in pol["injection_patterns"]]
        esc = [re.compile(p) for p in pol["escalate_if_text_matches"]]
        hits = Counter()
        for ev in arl.values():
            tt = ev[0]["task_type"]
            text = "\n".join(e["content"] for e in ev if e["type"] == "user_input")
            hits[(tt == "injected_instruction", any(p.search(text) for p in inj), "inj")] += 1
            hits[(tt == "complaint_escalation", any(p.search(text) for p in esc), "esc")] += 1
        check(hits[(True, False, "inj")] == 0 and hits[(False, True, "inj")] == 0, f"policy injection regex: {hits}")
        check(hits[(True, False, "esc")] == 0 and hits[(False, True, "esc")] == 0, f"policy escalation regex: {hits}")
        print("skydesk_policy.yaml: parses; injection/escalation regexes match exactly the intended runs")
    except ImportError:
        print("skydesk_policy.yaml: PyYAML not installed, skipped")

    # summary
    by = defaultdict(lambda: {"n": 0, "ok": 0, "modes": Counter(), "mode_ok": Counter(), "cost": 0.0})
    for rid, ev in arl.items():
        s = by[agent[rid]]
        s["n"] += 1
        s["ok"] += truth[rid]
        m = ev[-1]["meta"]["planted_failure"]
        s["modes"][m] += 1
        s["mode_ok"][m] += truth[rid]
        s["cost"] += sum(e.get("cost_usd", 0) for e in ev)
    print()
    for ag, s in sorted(by.items()):
        print(f"{ag}: runs={s['n']} verified_success={s['ok']}/{s['n']} ({s['ok'] / s['n']:.1%}) "
              f"llm_cost=${s['cost']:.4f} cost/verified_success=${s['cost'] / max(1, s['ok']):.5f}")
        for m, c in s["modes"].most_common():
            print(f"    {m:32s} {c:3d}  (verified ok {s['mode_ok'][m]})")


def main():
    validate_demo()
    if (D / "skydesk_arl.jsonl").exists():
        print()
        validate_bulk()
    else:
        print(f"(no bulk set in {D}; run `make samples` to generate and check it)")
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)} checks:")
        for m in FAIL[:30]:
            print("  -", m)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
