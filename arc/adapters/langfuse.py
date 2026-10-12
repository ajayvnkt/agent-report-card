"""Format 6: Langfuse export (traces with their observations).

Accepted shapes (what the Langfuse UI / public API give you):
  * a JSON list of traces, ``{"data": [traces], "meta": {...}}``, or JSONL with one trace per line,
    where a trace is ``{"id","name","timestamp","input","output","metadata","tags","sessionId",
    "userId","scores":[...], "observations":[...]}``;
  * observations exported on their own (list / ``{"data": [...]}`` / JSONL), grouped by ``traceId``.
    Both can be mixed in one file: loose observations are attached to their trace.

Mapping (trace = one run, ordered by observation ``startTime``):
  trace.id -> run_id; trace.name (or metadata.agent) -> agent; metadata.task_type or a
  ``task_type:<x>`` tag -> task_type; trace.input -> user_input; trace.output -> final_answer
  (claimed_success from metadata.claimed_success, else inferred from the text by arc/claims.py);
  a score named verified_success / success / ... (0/1) or metadata.outcome -> run_end.outcome.
  GENERATION -> llm_request (startTime) + llm_response (endTime; model, usage tokens, cost), and any
  ``tool_calls`` in its output -> tool_call (results from later ``role:"tool"`` messages).
  TOOL observations, or SPANs that are tools (metadata.tool, a ``tool:``/``execute_tool`` name, or the
  name of a tool the model asked for) -> tool_call (input = args) + tool_result (output = result;
  level ERROR / statusMessage -> error, then status error/refused/timeout by the shared inference).
  EVENT observations named like an ARL event type (escalation, error, user_input, ...) -> that event.
  A successful call of an escalation tool (escalate_*, handoff, transfer_to_human) -> escalation,
  unless the trace already logs escalation events. Other spans (AGENT, CHAIN, ...) are containers
  and are ignored. Without trace.output, the last plain-text generation is taken as the final answer.

The writer (``dump``) puts what plain Langfuse fields cannot hold (call ids, explicit statuses, event
order, meta) under ``metadata.arc`` so that ARL -> Langfuse -> ARL is lossless. Ordinary Langfuse
exports never need it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace

from ..schema import EVENT_TYPES, FIELD_NAMES, Event, _to_bool, group_runs, infer_tool_status, is_error_text, \
    normalize_type, pair_tool_calls
from ._common import ParseResult, Source, dict_records, iter_json_records, json_or_raw
from .otel import iso_to_ns

NAME = "langfuse"
LABEL = "Langfuse export (traces + observations)"

OBS_TYPES = {"GENERATION", "SPAN", "EVENT", "TOOL", "AGENT", "CHAIN", "RETRIEVER", "EVALUATOR", "EMBEDDING",
             "GUARDRAIL"}
ESCALATION_TOOL_RE = re.compile(r"escalat|hand_?off|handover|transfer_to_(human|agent|specialist|operator)|"
                                r"human_?(agent|handoff|review)", re.I)
OUTCOME_SCORES = ("verified_success", "success", "task_success", "outcome", "resolved", "correct",
                  "task_completed", "goal_completed")
TOOL_PREFIXES = ("tool:", "tool.", "execute_tool ", "tool_call:")
_EVENT_KEYS = [f for f in FIELD_NAMES if f not in ("run_id", "seq", "type")]


# ----------------------------------------------------------------------------- record shapes
def _upper(v) -> str:
    return str(v or "").strip().upper()


def _is_obs(r) -> bool:
    return (isinstance(r, dict) and bool(r.get("traceId")) and _upper(r.get("type")) in OBS_TYPES
            and "spanId" not in r and "span_id" not in r)


def _is_trace(r) -> bool:
    if not isinstance(r, dict) or "traceId" in r or "type" in r or "messages" in r:
        return False
    if "spanId" in r or "span_id" in r or "resourceSpans" in r:
        return False
    if isinstance(r.get("observations"), list) and ("id" in r or "timestamp" in r):
        return True
    return ("id" in r and "timestamp" in r
            and any(k in r for k in ("input", "output", "sessionId", "userId", "tags", "scores", "metadata")))


def _expand(rec):
    """A record, or the items of a ``{"data": [...]}`` page."""
    if isinstance(rec, dict) and isinstance(rec.get("data"), list) and not _is_trace(rec):
        yield from rec["data"]
    else:
        yield rec


def sniff(records: list, head: str) -> float:
    d = [r for rec in dict_records(records) for r in _expand(rec) if isinstance(r, dict)]
    if not d:
        if '"observations"' in head and '"timestamp"' in head and '"spanId"' not in head:
            return 0.8
        if '"traceId"' in head and '"startTime"' in head and '"spanId"' not in head \
                and re.search(r'"type"\s*:\s*"(GENERATION|SPAN|EVENT|TOOL)"', head):
            return 0.75
        return 0.0
    n_tr = sum(1 for r in d if _is_trace(r))
    n_ob = sum(1 for r in d if _is_obs(r))
    f = (n_tr + n_ob) / len(d)
    if f < 0.5:
        return 0.0
    rich = any(isinstance(r.get("observations"), list) and any(_is_obs(o) for o in r["observations"])
               for r in d if _is_trace(r))
    return round(min(0.99, 0.72 + 0.22 * f + (0.05 if rich else 0.0)), 3)


# ----------------------------------------------------------------------------- value helpers
def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _message_text(content) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        parts = [p.get("text") or "" if isinstance(p, dict) else str(p) for p in content]
        return "".join(str(p) for p in parts)
    return json.dumps(content, ensure_ascii=False)


def _text(v, role: str | None = None) -> str | None:
    """Readable text from a Langfuse input/output value (string, message, message list, dict)."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, list):
        msgs = [m for m in v if isinstance(m, dict) and "role" in m]
        if msgs:
            pick = [m for m in msgs if role is None or m.get("role") == role] or msgs
            return _message_text(pick[-1].get("content"))
    if isinstance(v, dict):
        if isinstance(v.get("messages"), list):
            return _text(v["messages"], role)
        for k in ("content", "text", "output", "answer", "response", "input", "query", "question", "message",
                  "prompt"):
            if isinstance(v.get(k), str):
                return v[k]
    return json.dumps(v, ensure_ascii=False)


def _as_args(v) -> dict:
    v = json_or_raw(v)
    if v is None:
        return {}
    return v if isinstance(v, dict) else {"_value": v}


def _first(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _task_type(md: dict, tags) -> str | None:
    tt = _first(md, "task_type", "taskType", "task")
    if tt is not None:
        return str(tt)
    tags = [str(t) for t in tags or [] if t is not None]
    for t in tags:
        for p in ("task_type:", "task_type=", "task:"):
            if t.lower().startswith(p):
                return t[len(p):].strip() or None
    return tags[0] if len(tags) == 1 else None


def _outcome(trace: dict, md: dict) -> dict | None:
    for s in trace.get("scores") or []:
        if not isinstance(s, dict) or str(s.get("name", "")).lower() not in OUTCOME_SCORES:
            continue
        v = s.get("value") if s.get("value") is not None else s.get("stringValue")
        b = _to_bool(v)
        if b is not None:
            return {"verified_success": int(b), "source": s.get("comment") or f"langfuse score {s.get('name')}"}
    oc = md.get("outcome")
    if isinstance(oc, dict) and "verified_success" in oc:
        return dict(oc)
    for v in (oc, md.get("verified_success")):
        b = _to_bool(v) if not isinstance(v, dict) else None
        if b is not None:
            return {"verified_success": int(b)}
    return None


def _gen_output(out) -> tuple[str | None, list[dict]]:
    """(text, tool_calls) of a generation's output."""
    msg = out
    if isinstance(out, list):
        msgs = [m for m in out if isinstance(m, dict) and "role" in m]
        msg = msgs[-1] if msgs else None
        if msg is None:
            return _text(out), []
    if isinstance(msg, dict):
        if isinstance(msg.get("choices"), list) and msg["choices"]:
            msg = _dict(msg["choices"][0]).get("message") or msg
        calls = msg.get("tool_calls") or ([{"function": msg["function_call"]}] if msg.get("function_call") else [])
        calls = [c for c in calls if isinstance(c, dict)]
        text = msg.get("content")
        if calls or "role" in msg or "content" in msg:
            return _message_text(text), calls
        return _text(msg), []
    return _text(out), []


def _ms(start, end) -> float | None:
    a, b = iso_to_ns(start), iso_to_ns(end)
    if a is None or b is None or b < a:
        return None
    v = (b - a) / 1e6
    return int(v) if v == int(v) else round(v, 3)


# ----------------------------------------------------------------------------- native mapping
def _tool_name(o: dict, requested: set[str]) -> str | None:
    """Tool name when this observation is a tool execution, else None."""
    t, name, md = _upper(o.get("type")), str(o.get("name") or ""), _dict(o.get("metadata"))
    if t == "EVENT" and normalize_type(name) in EVENT_TYPES:
        return None
    if isinstance(md.get("tool"), str) and md["tool"]:
        return md["tool"]
    for p in TOOL_PREFIXES:
        if name.lower().startswith(p):
            return name[len(p):].strip() or None
    if t == "TOOL" or md.get("tool") is True or str(md.get("type", "")).lower() == "tool":
        return name or None
    if t in ("SPAN", "EVENT") and name in requested:
        return name
    return None


def _obs_events(o: dict, rid: str, requested: set[str], covered_ids: set[str], covered_names: set[str],
                pr: ParseResult | None) -> list[Event]:
    """ARL events for one observation, from the plain Langfuse fields only."""
    t, name, md = _upper(o.get("type")), str(o.get("name") or ""), _dict(o.get("metadata"))
    level, smsg = _upper(o.get("level")) or "DEFAULT", o.get("statusMessage")
    start, end = o.get("startTime"), o.get("endTime")
    out, inp = o.get("output"), o.get("input")
    etype = normalize_type(name) if t in ("EVENT", "SPAN") else None
    if etype in EVENT_TYPES and t == "EVENT":
        content = _text(out) if out is not None else _text(inp)
        err = smsg if level in ("ERROR", "WARNING") and smsg else None
        if etype == "error" and err is None:
            err = content
        return [Event(run_id=rid, type=etype, ts=start, content=content, error=err,
                      model=o.get("model"))]
    if t == "GENERATION":
        usage = {**_dict(o.get("usage")), **_dict(o.get("usageDetails"))}
        cost = _num(_first(o, "calculatedTotalCost", "totalCost"))
        if cost is None:
            cost = _num(_dict(o.get("costDetails")).get("total"))
        model = o.get("model") or md.get("model")
        evs = [Event(run_id=rid, type="llm_request", ts=start, model=model)]
        text, calls = _gen_output(out)
        if end is not None or out is not None or usage:
            evs.append(Event(run_id=rid, type="llm_response", ts=end or start, model=model, content=text,
                             tokens_in=_num(_first(usage, "input", "promptTokens", "prompt_tokens", "input_tokens")),
                             tokens_out=_num(_first(usage, "output", "completionTokens", "completion_tokens",
                                                    "output_tokens")),
                             cost_usd=cost, latency_ms=_ms(start, end)))
        for c in calls:
            fn = _dict(c.get("function"))
            tname = fn.get("name") or c.get("name")
            cid = c.get("id")
            if (cid and cid in covered_ids) or (tname and tname in covered_names):
                continue  # executed as its own TOOL observation
            evs.append(Event(run_id=rid, type="tool_call", ts=end or start, tool=tname, call_id=cid,
                             args=_as_args(fn.get("arguments", c.get("arguments")))))
        for m in inp if isinstance(inp, list) else _dict(inp).get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id") \
                    and m["tool_call_id"] not in covered_ids:
                evs.insert(0, Event(run_id=rid, type="tool_result", ts=start, tool=m.get("name"),
                                    call_id=m["tool_call_id"], result=json_or_raw(_message_text(m.get("content")))))
        return evs
    tool = _tool_name(o, requested)
    if tool:
        cid = _first(md, "call_id", "tool_call_id") or o.get("id")
        cid = str(cid) if cid is not None else None
        call = Event(run_id=rid, type="tool_call", ts=start, tool=tool, call_id=cid, args=_as_args(inp))
        if out is None and end is None and level != "ERROR" and not smsg:
            return [call]  # never answered
        err, res = None, out
        if level == "ERROR":
            err = smsg or (out if isinstance(out, str) else None) or "Error"
            if err is out:
                res = None
        elif smsg and (level == "WARNING" or is_error_text(smsg)):
            err = smsg
        status = md.get("status") if isinstance(md.get("status"), str) else None
        return [call, Event(run_id=rid, type="tool_result", ts=end or start, tool=tool, call_id=cid, result=res,
                            error=err, status=status, latency_ms=_ms(start, end))]
    if pr is not None:
        pr.warn(f"ignored {t or 'untyped'} observations that are neither generations, tools nor ARL events")
    return []


def _trace_events(trace: dict, rid: str) -> dict[str, Event]:
    """The trace-level events (run_start, user_input, final_answer, run_end) from plain fields."""
    md = _dict(trace.get("metadata"))
    meta = {k2: trace[k] for k, k2 in (("sessionId", "session_id"), ("userId", "user_id")) if trace.get(k)}
    out = {"run_start": Event(run_id=rid, type="run_start", ts=trace.get("timestamp"),
                              agent=md.get("agent") or trace.get("name") or None, model=md.get("model"),
                              task_type=_task_type(md, trace.get("tags")), meta=meta or None)}
    if trace.get("input") is not None:
        out["user_input"] = Event(run_id=rid, type="user_input", ts=trace.get("timestamp"),
                                  content=_text(trace["input"], "user"))
    if trace.get("output") is not None:
        ca = md.get("claimed_actions")
        out["final_answer"] = Event(run_id=rid, type="final_answer", content=_text(trace["output"], "assistant"),
                                    claimed_success=_to_bool(md.get("claimed_success")),
                                    claimed_actions=ca if isinstance(ca, list) else None)
    out["run_end"] = Event(run_id=rid, type="run_end", outcome=_outcome(trace, md))
    return out


def _apply(ev: Event, x: dict | None) -> Event:
    for k, v in (x or {}).items():
        if k in _EVENT_KEYS:
            setattr(ev, k, v)
    return ev


# ----------------------------------------------------------------------------- parse
def _context(obs: list[dict]) -> tuple[set, set, set]:
    requested, covered_ids, covered_names = set(), set(), set()
    for o in obs:
        if _upper(o.get("type")) == "GENERATION":
            for c in _gen_output(o.get("output"))[1]:
                n = _dict(c.get("function")).get("name") or c.get("name")
                if n:
                    requested.add(n)
    for o in obs:
        if _upper(o.get("type")) != "GENERATION":
            tool = _tool_name(o, requested)
            if tool:
                covered_names.add(tool)
                md = _dict(o.get("metadata"))
                for cid in (md.get("call_id"), md.get("tool_call_id"), o.get("id")):
                    if cid:
                        covered_ids.add(str(cid))
    return requested, covered_ids, covered_names


def _trace_to_events(trace: dict, obs: list[dict], pr: ParseResult) -> list[Event]:
    rid = str(trace.get("id"))
    md = _dict(trace.get("metadata"))
    arc = md.get("arc") if isinstance(md.get("arc"), dict) else None
    times = [iso_to_ns(o.get("startTime")) for o in obs]
    last = None
    keyed = []
    for i, (o, t) in enumerate(zip(obs, times)):  # observations without startTime keep their place
        last = t if t is not None else last
        keyed.append(((last if last is not None else -1), i, o))
    keyed.sort(key=lambda k: (k[0], k[1]))
    obs = [o for _, _, o in keyed]
    requested, covered_ids, covered_names = _context(obs)

    top = _trace_events(trace, rid)
    if arc is not None:  # written by dump(): exact order and fields
        items: list[tuple[float, Event]] = []
        for role, ev in top.items():
            if role in (arc.get("drop") or []):
                continue
            items.append((_seq(arc, role), _apply(ev, _dict(arc.get("x")).get(role))))
        for o in obs:
            oa = _dict(_dict(o.get("metadata")).get("arc"))
            if isinstance(oa.get("events"), list):
                evs = [Event.from_dict({**d, "run_id": rid}) for d in oa["events"] if isinstance(d, dict)]
            else:
                evs = _obs_events(o, rid, requested, covered_ids, covered_names, pr)
                for ev, x in zip(evs, oa.get("x") or []):
                    _apply(ev, x)
            seqs = oa.get("seq") or []
            for j, ev in enumerate(evs):
                items.append((seqs[j] if j < len(seqs) else float("inf"), ev))
        items.sort(key=lambda t: t[0])
        return [ev for _, ev in items]

    body: list[Event] = []
    for o in obs:
        body.extend(_obs_events(o, rid, requested, covered_ids, covered_names, pr))
    # Escalation tools count as a hand-off unless the trace logs escalation events itself.
    if not any(e.type == "escalation" for e in body):
        out: list[Event] = []
        args_of: dict = {}
        for e in body:
            out.append(e)
            if e.type == "tool_call":
                args_of[e.call_id] = e.args or {}
            elif e.type == "tool_result" and e.tool and ESCALATION_TOOL_RE.search(e.tool) and not e.error:
                why = _first(args_of.get(e.call_id, {}), "summary", "reason", "message")
                if why is None and isinstance(e.result, dict):
                    why = _first(e.result, "summary", "reason", "queue", "ticket_id")
                out.append(Event(run_id=rid, type="escalation", ts=e.ts, content=None if why is None else str(why)))
        body = out
    if top["run_start"].model is None:
        top["run_start"].model = next((e.model for e in body if e.model), None)
    t_end = max((t for t in (o.get("endTime") or o.get("startTime") for o in obs) if t),
                key=lambda t: iso_to_ns(t) or 0, default=None)
    evs = [top["run_start"]] + ([top["user_input"]] if "user_input" in top else []) + body
    if "final_answer" in top:
        top["final_answer"].ts = t_end
        evs.append(top["final_answer"])
    else:  # no trace.output: the last plain-text generation after all tool work is the answer
        for i in range(len(evs) - 1, -1, -1):
            e = evs[i]
            if e.type in ("tool_call", "tool_result"):
                break
            if e.type == "llm_response" and e.content and e.content.strip():
                evs.insert(i + 1, Event(run_id=rid, type="final_answer", ts=e.ts, content=e.content))
                break
    top["run_end"].ts = t_end
    evs.append(top["run_end"])
    return evs


def _seq(arc: dict, role: str) -> float:
    v = _dict(arc.get("seq")).get(role)
    return v if isinstance(v, (int, float)) else float("inf")


def parse(src: Source) -> ParseResult:
    pr = ParseResult()
    traces: dict[str, dict] = {}
    obs_by_trace: dict[str, dict[str, dict]] = {}
    listed: dict[str, list[str]] = {}

    def add_obs(tid: str, o: dict) -> None:
        bucket = obs_by_trace.setdefault(tid, {})
        bucket[str(o.get("id") or f"_{len(bucket)}")] = o

    for lineno, rec in iter_json_records(src, pr):
        for r in _expand(rec):
            if _is_trace(r):
                tid = str(r.get("id") or f"{src.stem}-{len(traces) + 1}")
                traces.setdefault(tid, {}).update({**r, "id": tid})
                obs_by_trace.setdefault(tid, {})
                for o in r.get("observations") or []:
                    if isinstance(o, dict):
                        add_obs(tid, o)
                    elif isinstance(o, str):
                        listed.setdefault(tid, []).append(o)
            elif _is_obs(r):
                tid = str(r["traceId"])
                add_obs(tid, r)
            else:
                pr.skip(lineno, json.dumps(r, ensure_ascii=False, default=str)[:80],
                        "not a Langfuse trace or observation")
    if any(oid not in obs_by_trace.get(tid, {}) for tid, ids in listed.items() for oid in ids):
        pr.warn("some traces list observation ids only -- export them with their observations "
                "(or add the observations file) to see tool calls")
    for tid in list(obs_by_trace):
        if tid not in traces:
            traces[tid] = {"id": tid}
    for tid, trace in traces.items():
        pr.events.extend(_trace_to_events(trace, list(obs_by_trace.get(tid, {}).values()), pr))
    return pr


# ----------------------------------------------------------------------------- export
def _hex(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def _diff(orig: Event, native: Event) -> dict:
    x = {k: getattr(orig, k) for k in _EVENT_KEYS if not _same(getattr(orig, k), getattr(native, k))}
    if "status" in x and orig.type == "tool_result":  # leave out a status the loader infers anyway
        probe = replace(native, **x)
        probe.status = None
        infer_tool_status(probe)
        if probe.status == orig.status:
            del x["status"]
    return x


def _attach(o: dict, originals: list[Event], rid: str) -> dict:
    """Store in metadata.arc what the plain fields of `o` do not reproduce exactly."""
    native = _obs_events(o, rid, set(), set(), set(), None)
    arc: dict = {"seq": [e.seq for e in originals]}
    if [e.type for e in native] != [e.type for e in originals]:
        arc["events"] = [{k: v for k, v in e.to_dict().items() if k not in ("run_id", "seq")} for e in originals]
    else:
        xs = [_diff(a, b) for a, b in zip(originals, native)]
        if any(xs):
            arc["x"] = xs
    o.setdefault("metadata", {})["arc"] = arc
    return o


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _trace_for_run(rid: str, evs: list[Event]) -> dict:
    starts = [e for e in evs if e.type == "run_start"]
    users = [e for e in evs if e.type == "user_input"]
    finals = [e for e in evs if e.type == "final_answer"]
    ends = [e for e in evs if e.type == "run_end"]
    start, user = (starts[0] if starts else None), (users[0] if users else None)
    final, end = (finals[-1] if finals else None), (ends[-1] if ends else None)
    top_ids = {id(x) for x in (start, user, final, end) if x is not None}
    pairs = pair_tool_calls(evs)
    result_of = {id(c): r for c, r in pairs}
    answered = {id(r) for _, r in pairs if r is not None}
    # llm_request directly followed (among LLM events) by its llm_response -> one GENERATION
    gen_resp: dict[int, Event] = {}
    pending = None
    for e in evs:
        if e.type == "llm_request":
            pending = e
        elif e.type == "llm_response":
            if pending is not None:
                gen_resp[id(pending)] = e
            pending = None
    merged = {id(r) for r in gen_resp.values()}

    observations: list[dict] = []
    for e in evs:
        if id(e) in top_ids or id(e) in answered or id(e) in merged:
            continue
        oid = _hex(f"{rid}#{e.seq}")
        base = {"id": oid, "traceId": rid}
        if e.type == "tool_call":
            r = result_of.get(id(e))
            md: dict = {}
            if e.call_id:
                md["call_id"] = e.call_id
            o = {**base, "type": "TOOL", "name": e.tool, "startTime": e.ts, "input": e.args or {},
                 "metadata": md}
            orig = [e]
            if r is not None:
                o.update({"endTime": r.ts or e.ts, "output": r.result})
                if r.status and r.status != "ok":
                    o["level"] = "ERROR"
                    o["statusMessage"] = r.error or r.status
                    md["status"] = r.status
                orig.append(r)
            observations.append(_attach(_clean(o), orig, rid))
        elif e.type in ("llm_request", "llm_response"):
            resp = gen_resp.get(id(e)) if e.type == "llm_request" else e
            req = e if e.type == "llm_request" else None
            o = {**base, "type": "GENERATION", "name": "chat", "startTime": (req or resp).ts,
                 "model": (resp.model if resp else None) or (req.model if req else None)}
            if resp is not None:
                o["endTime"] = resp.ts
                o["output"] = resp.content
                usage = _clean({"input": resp.tokens_in, "output": resp.tokens_out,
                                "total": (resp.tokens_in or 0) + (resp.tokens_out or 0)
                                if resp.tokens_in is not None or resp.tokens_out is not None else None})
                if usage:
                    o["usageDetails"] = usage
                if resp.cost_usd is not None:
                    o["calculatedTotalCost"] = resp.cost_usd
            orig = [x for x in (req, resp) if x is not None]
            observations.append(_attach(_clean(o), orig, rid))
        else:
            o = {**base, "type": "EVENT", "name": e.type, "startTime": e.ts}
            if e.type == "user_input":
                o["input"] = e.content
            else:
                o["output"] = e.content if e.content is not None else (e.result if e.type == "tool_result" else None)
            if e.error:
                o["level"] = "ERROR"
                o["statusMessage"] = e.error
            observations.append(_attach(_clean(o), [e], rid))

    agent = start.agent if start else None
    task_type = start.task_type if start else None
    md: dict = _clean({"task_type": task_type, "model": start.model if start else None})
    if final is not None:
        if final.claimed_success is not None:
            md["claimed_success"] = final.claimed_success
        if final.claimed_actions is not None:
            md["claimed_actions"] = final.claimed_actions
    trace = _clean({"id": rid, "name": agent, "timestamp": start.ts if start else None,
                    "input": user.content if user else None, "output": final.content if final else None,
                    "tags": [f"task_type:{task_type}"] if task_type else None, "metadata": md})
    if user is not None and user.content is None:
        trace["input"] = ""  # keep the user turn even when it had no text
    if final is not None and final.content is None:
        trace["output"] = ""
    if end is not None and end.outcome and "verified_success" in end.outcome:
        score = {"name": "verified_success", "value": end.outcome["verified_success"], "dataType": "BOOLEAN"}
        if end.outcome.get("source"):
            score["comment"] = end.outcome["source"]
        trace["scores"] = [score]
    # what the trace fields do not reproduce exactly
    native = _trace_events(trace, rid)
    arc: dict = {"seq": {}, "x": {}, "drop": []}
    for role, orig in (("run_start", start), ("user_input", user), ("final_answer", final), ("run_end", end)):
        if orig is None:
            if role in native:
                arc["drop"].append(role)
            continue
        arc["seq"][role] = orig.seq
        x = _diff(orig, native[role])
        if x:
            arc["x"][role] = x
    trace["metadata"]["arc"] = {k: v for k, v in arc.items() if v}
    trace["observations"] = observations
    return trace


def dump(events: list[Event]) -> str:
    traces = [_trace_for_run(rid, evs) for rid, evs in group_runs(events).items()]
    return json.dumps(traces, ensure_ascii=False, indent=1, default=str) + "\n"
