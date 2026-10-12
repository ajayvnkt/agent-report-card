"""Format 4: OpenTelemetry GenAI spans (OTLP JSON, or a JSON list / JSONL of flat spans).

Conventions (see docs/LOG_FORMAT.md):
  trace_id = one run. Root span "invoke_agent <agent>" carries gen_ai.agent.name, arc.task_type,
  arc.user_input, arc.final_answer, arc.claimed_success, arc.outcome.verified_success.
  LLM spans: gen_ai.operation.name="chat", gen_ai.request.model, gen_ai.usage.input/output_tokens.
  Tool spans "execute_tool <tool>": gen_ai.tool.name, gen_ai.tool.call.id,
  gen_ai.tool.call.arguments / gen_ai.tool.call.result (JSON strings); failures via status ERROR /
  error.type. Spans named "arc.<event type>" carry ARL events with no OTel equivalent.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from ..schema import EVENT_TYPES, Event, group_runs, pair_tool_calls
from ._common import ParseResult, Source, dict_records, frac, iter_json_records, json_or_raw

NAME = "otel"
LABEL = "OpenTelemetry GenAI spans (OTLP JSON / flat spans)"

BASE_NS = 1_767_225_600 * 10**9  # 2026-01-01T00:00:00Z, used when events have no ts
LLM_OPS = ("chat", "text_completion", "generate_content", "completion", "embeddings")


# ----------------------------------------------------------------------------- helpers
def _is_flat_span(r: dict) -> bool:
    ctx = r.get("context") if isinstance(r.get("context"), dict) else {}
    return bool((r.get("span_id") or r.get("spanId") or ctx.get("span_id"))
                and (r.get("trace_id") or r.get("traceId") or ctx.get("trace_id")))


def sniff(records: list, head: str) -> float:
    d = dict_records(records)
    if any("resourceSpans" in r for r in d):
        return 0.98
    if not d and '"resourceSpans"' in head:
        return 0.9
    f = frac(d, _is_flat_span)
    if f >= 0.5:
        return round(0.75 + 0.2 * f, 3)
    if not d and ('"span_id"' in head or '"spanId"' in head) and ('"trace_id"' in head or '"traceId"' in head):
        return 0.7
    return 0.0


def _any_value(v):
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return v["intValue"]
    if "arrayValue" in v:
        return [_any_value(x) for x in (v["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in v:
        return {kv.get("key"): _any_value(kv.get("value")) for kv in (v["kvlistValue"] or {}).get("values", [])}
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs(a) -> dict:
    if isinstance(a, dict):
        return dict(a)
    out = {}
    if isinstance(a, list):
        for kv in a:
            if isinstance(kv, dict) and "key" in kv:
                out[kv["key"]] = _any_value(kv.get("value"))
    return out


def iso_to_ns(s) -> int | None:
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        x = float(s)
        if x > 1e17:
            return int(x)
        if x > 1e14:
            return int(x * 1e3)
        if x > 1e11:
            return int(x * 1e6)
        return int(x * 1e9)
    s = str(s).strip()
    if s.isdigit():
        return iso_to_ns(int(s))
    try:
        t = s.replace("Z", "+00:00")
        frac_ns = 0
        if "." in t:
            main, rest = t.split(".", 1)
            digits = ""
            i = 0
            while i < len(rest) and rest[i].isdigit():
                digits += rest[i]
                i += 1
            tz = rest[i:]
            frac_ns = int((digits + "000000000")[:9]) if digits else 0
            t = main + tz
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp()) * 10**9 + frac_ns
    except ValueError:
        return None


def ns_to_iso(ns: int) -> str:
    sec, rem = divmod(int(ns), 10**9)
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if rem:
        return f"{base}.{rem:09d}".rstrip("0") + "Z"
    return base + "Z"


def _status(st) -> tuple[str, str | None]:
    if not isinstance(st, dict):
        return "UNSET", None
    code = st.get("code", st.get("status_code"))
    msg = st.get("message") or st.get("description")
    if isinstance(code, int):
        return {0: "UNSET", 1: "OK", 2: "ERROR"}.get(code, "UNSET"), msg
    c = str(code or "").upper().replace("STATUS_CODE_", "")
    return (c if c in ("OK", "ERROR") else "UNSET"), msg


def _norm_span(r: dict, idx: int) -> dict:
    ctx = r.get("context") if isinstance(r.get("context"), dict) else {}
    return {
        "trace_id": str(r.get("trace_id") or r.get("traceId") or ctx.get("trace_id") or ""),
        "span_id": str(r.get("span_id") or r.get("spanId") or ctx.get("span_id") or ""),
        "parent": r.get("parent_span_id") or r.get("parentSpanId") or r.get("parent_id") or None,
        "name": str(r.get("name") or ""),
        "start": iso_to_ns(r.get("start_time") or r.get("startTimeUnixNano") or r.get("start")),
        "end": iso_to_ns(r.get("end_time") or r.get("endTimeUnixNano") or r.get("end")),
        "status": _status(r.get("status")),
        "attrs": _attrs(r.get("attributes")),
        "idx": idx,
    }


def _iter_spans(src: Source, pr: ParseResult):
    idx = 0
    for lineno, rec in iter_json_records(src, pr):
        if not isinstance(rec, dict):
            pr.skip(lineno, str(rec)[:80], "not a JSON object")
            continue
        if "resourceSpans" in rec:
            for rs in rec.get("resourceSpans") or []:
                res_attrs = _attrs((rs.get("resource") or {}).get("attributes"))
                for ss in (rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []):
                    for sp in ss.get("spans") or []:
                        n = _norm_span(sp, idx)
                        n["resource"] = res_attrs
                        idx += 1
                        yield n
        elif _is_flat_span(rec):
            yield _norm_span(rec, idx)
            idx += 1
        else:
            pr.skip(lineno, json.dumps(rec, ensure_ascii=False)[:80], "not a span")


def _ts(ns):
    return ns_to_iso(ns) if ns is not None else None


def _as_args(v):
    v = json_or_raw(v)
    if v is None:
        return None
    return v if isinstance(v, dict) else {"_value": v}


def _first(a: dict, *keys):
    for k in keys:
        if a.get(k) is not None:
            return a[k]
    return None


def _is_work_span(s: dict) -> bool:
    a, name = s["attrs"], s["name"]
    return bool(a.get("gen_ai.tool.name") or name.startswith(("execute_tool", "chat", "arc."))
                or a.get("gen_ai.operation.name") in LLM_OPS + ("execute_tool",))


def _trace_to_events(spans: list[dict], pr: ParseResult) -> list[Event]:
    root = next((s for s in spans if s["name"].startswith("invoke_agent")
                 or s["attrs"].get("gen_ai.operation.name") == "invoke_agent"), None)
    if root is None:
        root = next((s for s in spans if not s["parent"] and not _is_work_span(s)), None)
    ra = root["attrs"] if root else {}
    rid = str(ra.get("arc.run_id") or spans[0]["trace_id"])
    evs: list[Event] = []
    agent = _first(ra, "gen_ai.agent.name") or (root["name"].split(" ", 1)[1] if root and " " in root["name"] else None)
    evs.append(Event(run_id=rid, type="run_start", ts=_ts(root["start"] if root else min(
                         (x["start"] for x in spans if x["start"] is not None), default=None)), agent=agent,
                     model=_first(ra, "gen_ai.request.model"), task_type=_first(ra, "arc.task_type")))
    if ra.get("arc.user_input") is not None:
        evs.append(Event(run_id=rid, type="user_input", content=str(ra["arc.user_input"]),
                         ts=_ts(root["start"])))
    ignored = 0
    children = sorted((s for s in spans if s is not root), key=lambda s: (s["start"] or 0, s["idx"]))
    for s in children:
        a, name = s["attrs"], s["name"]
        code, smsg = s["status"]
        if name.startswith("arc.") and name[4:] in EVENT_TYPES:
            evs.append(Event(run_id=rid, type=name[4:], ts=_ts(s["start"]), content=a.get("arc.content"),
                             tool=a.get("gen_ai.tool.name"), call_id=a.get("gen_ai.tool.call.id"),
                             error=a.get("arc.error"), status=a.get("arc.status"),
                             model=a.get("gen_ai.request.model"),
                             args=_as_args(a.get("gen_ai.tool.call.arguments")),
                             result=json_or_raw(a.get("gen_ai.tool.call.result")),
                             claimed_success=a.get("arc.claimed_success")))
            continue
        op = a.get("gen_ai.operation.name")
        tool = a.get("gen_ai.tool.name") or (name.split(" ", 1)[1] if name.startswith("execute_tool ") else None)
        if tool or op == "execute_tool":
            cid = a.get("gen_ai.tool.call.id")
            evs.append(Event(run_id=rid, type="tool_call", ts=_ts(s["start"]), tool=tool, call_id=cid,
                             args=_as_args(a.get("gen_ai.tool.call.arguments")) or {}))
            if a.get("arc.unanswered"):
                continue
            # A tool span that never ended and carries no result/error never got an answer.
            if (s["end"] is None and code != "ERROR" and a.get("gen_ai.tool.call.result") is None
                    and a.get("arc.error") is None and a.get("error.type") is None):
                continue
            err = a.get("arc.error")
            if err is None and code == "ERROR" and "arc.status" not in a:
                err = smsg or a.get("error.type") or "Error"
            res = json_or_raw(a.get("gen_ai.tool.call.result"))
            evs.append(Event(run_id=rid, type="tool_result", ts=_ts(s["end"]), tool=tool, call_id=cid,
                             result=res, error=err,
                             status=a.get("arc.status") or ("error" if code == "ERROR" and not err else None),
                             latency_ms=a.get("arc.latency_ms")))
        elif op in LLM_OPS or name.startswith("chat") or "gen_ai.usage.input_tokens" in a:
            evs.append(Event(run_id=rid, type="llm_response", ts=_ts(s["end"]),
                             model=_first(a, "gen_ai.request.model", "gen_ai.response.model"),
                             content=_first(a, "arc.content", "gen_ai.completion"),
                             tokens_in=_first(a, "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"),
                             tokens_out=_first(a, "gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"),
                             cost_usd=a.get("arc.cost_usd"), latency_ms=a.get("arc.latency_ms")))
        else:
            ignored += 1
    if ignored:
        pr.warn("ignored spans that are neither tool, LLM nor arc.* spans")
    if any(ra.get(k) is not None for k in ("arc.final_answer", "arc.claimed_success", "arc.has_final_answer")):
        ca = json_or_raw(ra.get("arc.claimed_actions"))
        evs.append(Event(run_id=rid, type="final_answer", ts=_ts(root["end"]),
                         content=ra.get("arc.final_answer"), claimed_success=ra.get("arc.claimed_success"),
                         claimed_actions=ca if isinstance(ca, list) else None))
    # A failed run is marked by an ERROR status on the root (invoke_agent) span.
    if root is not None and root["status"][0] == "ERROR" and not any(e.type == "error" for e in evs):
        msg = root["status"][1] or ra.get("error.type") or "agent run failed (root span status ERROR)"
        evs.append(Event(run_id=rid, type="error", ts=_ts(root["end"]), status="error", error=str(msg)))
    vs = ra.get("arc.outcome.verified_success")
    outcome = None
    if vs is not None:
        outcome = {"verified_success": vs}
        if ra.get("arc.outcome.source") is not None:
            outcome["source"] = ra["arc.outcome.source"]
    if ra.get("arc.has_run_end") is False and outcome is None:
        return evs
    last = root["end"] if root else max((x["end"] for x in spans if x["end"] is not None), default=None)
    evs.append(Event(run_id=rid, type="run_end", ts=_ts(last), outcome=outcome))
    return evs


def parse(src: Source) -> ParseResult:
    pr = ParseResult()
    traces: dict[str, list[dict]] = {}
    for sp in _iter_spans(src, pr):
        traces.setdefault(sp["trace_id"], []).append(sp)
    for tid, spans in traces.items():
        pr.events.extend(_trace_to_events(spans, pr))
    return pr


# ----------------------------------------------------------------------------- export
def _hex(s: str, n: int) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _spans_for_run(rid: str, evs: list[Event]) -> list[dict]:
    times = {}
    for e in evs:
        ns = iso_to_ns(e.ts) if e.ts else None
        times[id(e)] = ns if ns is not None else BASE_NS + (e.seq or 0) * 10**6
    trace = _hex(rid, 32)
    start = next((e for e in evs if e.type == "run_start"), None)
    users = [e for e in evs if e.type == "user_input"]
    finals = [e for e in evs if e.type == "final_answer"]
    end = next((e for e in evs if e.type == "run_end" and e.outcome), None)
    agent = (start.agent if start else None) or next((e.agent for e in evs if e.agent), None) or "agent"
    root_attrs = {"arc.run_id": rid, "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": agent}
    if start and start.model:
        root_attrs["gen_ai.request.model"] = start.model
    if start and start.task_type:
        root_attrs["arc.task_type"] = start.task_type
    if users:
        root_attrs["arc.user_input"] = users[0].content or ""
    if finals:
        f = finals[-1]
        root_attrs["arc.has_final_answer"] = True
        if f.content is not None:
            root_attrs["arc.final_answer"] = f.content
        if f.claimed_success is not None:
            root_attrs["arc.claimed_success"] = f.claimed_success
        if f.claimed_actions is not None:
            root_attrs["arc.claimed_actions"] = json.dumps(f.claimed_actions, ensure_ascii=False)
    root_attrs["arc.has_run_end"] = any(e.type == "run_end" for e in evs)
    if end:
        if "verified_success" in end.outcome:
            root_attrs["arc.outcome.verified_success"] = end.outcome["verified_success"]
        if "source" in end.outcome:
            root_attrs["arc.outcome.source"] = end.outcome["source"]
    root_id = _hex(rid + "#root", 16)
    all_t = list(times.values()) or [BASE_NS]
    root = {"trace_id": trace, "span_id": root_id, "parent_span_id": None, "name": f"invoke_agent {agent}",
            "start": min(all_t), "end": max(all_t), "status": ("OK", None), "attributes": root_attrs}
    spans = [root]
    results = {id(r): c for c, r in pair_tool_calls(evs) if r is not None}
    result_of = {id(c): r for c, r in pair_tool_calls(evs)}
    skip = {id(start)} if start else set()
    if users:
        skip.add(id(users[0]))
    if finals:
        skip.add(id(finals[-1]))
    for e in evs:
        if id(e) in skip or e.type == "run_end" or (e.type == "tool_result" and id(e) in results):
            continue
        sid = _hex(f"{rid}#{e.seq}", 16)
        t = times[id(e)]
        if e.type == "tool_call":
            r = result_of.get(id(e))
            a = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": e.tool or ""}
            if e.call_id:
                a["gen_ai.tool.call.id"] = e.call_id
            a["gen_ai.tool.call.arguments"] = json.dumps(e.args or {}, ensure_ascii=False)
            status = ("OK", None)
            t_end = t
            if r is None:
                a["arc.unanswered"] = True
            else:
                t_end = times[id(r)]
                if r.result is not None:
                    a["gen_ai.tool.call.result"] = json.dumps(r.result, ensure_ascii=False, default=str)
                if r.status:
                    a["arc.status"] = r.status
                if r.error:
                    a["arc.error"] = r.error
                    a["error.type"] = r.error.split(":", 1)[0].strip()[:60] if ":" in r.error else "ToolError"
                if r.latency_ms is not None:
                    a["arc.latency_ms"] = r.latency_ms
                if r.status and r.status != "ok":
                    status = ("ERROR", r.error)
            spans.append({"trace_id": trace, "span_id": sid, "parent_span_id": root_id,
                          "name": f"execute_tool {e.tool}", "start": t, "end": t_end, "status": status,
                          "attributes": a})
        elif e.type == "llm_response":
            a = {"gen_ai.operation.name": "chat"}
            m = e.model or (start.model if start else None)
            if m:
                a["gen_ai.request.model"] = m
            if e.tokens_in is not None:
                a["gen_ai.usage.input_tokens"] = e.tokens_in
            if e.tokens_out is not None:
                a["gen_ai.usage.output_tokens"] = e.tokens_out
            if e.content is not None:
                a["arc.content"] = e.content
            if e.cost_usd is not None:
                a["arc.cost_usd"] = e.cost_usd
            if e.latency_ms is not None:
                a["arc.latency_ms"] = e.latency_ms
            spans.append({"trace_id": trace, "span_id": sid, "parent_span_id": root_id,
                          "name": f"chat {m or ''}".strip(), "start": t, "end": t, "status": ("OK", None),
                          "attributes": a})
        else:
            a = {}
            for k, v in (("arc.content", e.content), ("arc.error", e.error), ("arc.status", e.status),
                         ("gen_ai.tool.name", e.tool), ("gen_ai.tool.call.id", e.call_id),
                         ("gen_ai.request.model", e.model), ("arc.claimed_success", e.claimed_success)):
                if v is not None:
                    a[k] = v
            if e.args is not None:
                a["gen_ai.tool.call.arguments"] = json.dumps(e.args, ensure_ascii=False)
            if e.result is not None:
                a["gen_ai.tool.call.result"] = json.dumps(e.result, ensure_ascii=False, default=str)
            st = ("ERROR", e.error) if e.type == "error" else ("OK", None)
            spans.append({"trace_id": trace, "span_id": sid, "parent_span_id": root_id,
                          "name": f"arc.{e.type}", "start": t, "end": t, "status": st, "attributes": a})
    return spans


def dump(events: list[Event], otlp: bool = False) -> str:
    spans = []
    for rid, evs in group_runs(events).items():
        spans.extend(_spans_for_run(rid, evs))
    if not otlp:
        flat = []
        for s in spans:
            st = {"code": s["status"][0]}
            if s["status"][1]:
                st["message"] = s["status"][1]
            flat.append({"trace_id": s["trace_id"], "span_id": s["span_id"], "parent_span_id": s["parent_span_id"],
                         "name": s["name"], "start_time": ns_to_iso(s["start"]), "end_time": ns_to_iso(s["end"]),
                         "status": st, "attributes": s["attributes"]})
        return json.dumps(flat, ensure_ascii=False, indent=1, default=str) + "\n"

    def val(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        return {"stringValue": str(v)}

    otlp_spans = []
    for s in spans:
        sp = {"traceId": s["trace_id"], "spanId": s["span_id"], "name": s["name"], "kind": 1,
              "startTimeUnixNano": str(s["start"]), "endTimeUnixNano": str(s["end"]),
              "attributes": [{"key": k, "value": val(v)} for k, v in s["attributes"].items()],
              "status": {"code": 2 if s["status"][0] == "ERROR" else 1}}
        if s["parent_span_id"]:
            sp["parentSpanId"] = s["parent_span_id"]
        if s["status"][1]:
            sp["status"]["message"] = s["status"][1]
        otlp_spans.append(sp)
    doc = {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "agent"}}]},
                              "scopeSpans": [{"scope": {"name": "agent-report-card"}, "spans": otlp_spans}]}]}
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
