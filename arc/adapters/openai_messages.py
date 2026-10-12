"""Format 2: OpenAI chat messages.

Accepted shapes:
  * JSONL, one run per line: {"run_id","agent","model","task_type","messages":[...],"outcome":{...}}
  * a JSON array of such run objects, or one such object
  * a bare messages list (= one run), as JSON array or one message per JSONL line

Mapping: system -> run_start.meta.system_prompt; user -> user_input; assistant text with tool_calls
-> llm_response + tool_call per call; tool -> tool_result (by tool_call_id); the LAST assistant
text without tool_calls -> final_answer (earlier ones -> llm_response); outcome -> run_end.
"""
from __future__ import annotations

import json

from ..schema import Event
from ._common import ParseResult, Source, dict_records, dump_value_as_text, frac, iter_json_records, json_or_raw

NAME = "openai"
LABEL = "OpenAI chat messages (messages + tool_calls)"
ROLES = ("system", "developer", "user", "assistant", "tool", "function")


def _is_run(r: dict) -> bool:
    return isinstance(r.get("messages"), list)


def _is_message(r: dict) -> bool:
    return r.get("role") in ROLES and ("content" in r or "tool_calls" in r)


def sniff(records: list, head: str) -> float:
    d = dict_records(records)
    if not d:
        return 0.0
    f_run = frac(d, _is_run)
    if f_run >= 0.5:
        return round(0.7 + 0.25 * f_run, 3)
    f_msg = frac(d, _is_message)
    if f_msg >= 0.8:
        return 0.85
    return 0.0


def _text(content) -> str | None:
    """OpenAI content may be a string or a list of parts."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(str(x) for x in parts)
    return str(content)


def _is_parts(content) -> bool:
    """A list of OpenAI content parts ({"type": "text", "text": ...}) rather than a JSON list result."""
    return (isinstance(content, list) and bool(content)
            and all(isinstance(p, dict) and "type" in p and ("text" in p or p.get("type") != "text") for p in content)
            and any(isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text") for p in content))


def _args(raw, pr: ParseResult):
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {"_value": v}
    except ValueError:
        pr.warn("tool_call arguments are not valid JSON; kept as {'_raw': ...}")
        return {"_raw": raw}


def run_to_events(obj: dict, run_id: str, pr: ParseResult) -> list[Event]:
    msgs = [m for m in obj.get("messages") or [] if isinstance(m, dict)]
    rid = str(obj.get("run_id") or obj.get("id") or obj.get("conversation_id") or run_id)
    evs: list[Event] = []
    start_meta = {}
    sys_texts = [_text(m.get("content")) or "" for m in msgs if m.get("role") in ("system", "developer")]
    if sys_texts:
        start_meta["system_prompt"] = "\n\n".join(sys_texts)
    evs.append(Event(run_id=rid, type="run_start", agent=obj.get("agent"), model=obj.get("model"),
                     task_type=obj.get("task_type"), meta=start_meta or None))
    # index of last assistant message that is plain text (=> final answer)
    last_final = None
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and not m.get("tool_calls") and not m.get("function_call"):
            if (_text(m.get("content")) or "").strip():
                last_final = i
    if last_final is not None and any(m.get("role") in ("tool", "function") or m.get("tool_calls")
                                      for m in msgs[last_final + 1:]):
        last_final = None  # agent kept working after that text
    call_names: dict[str, str] = {}
    model = obj.get("model")
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role in ("system", "developer"):
            continue
        if role == "user":
            evs.append(Event(run_id=rid, type="user_input", content=_text(m.get("content"))))
        elif role == "assistant":
            text = _text(m.get("content"))
            calls = list(m.get("tool_calls") or [])
            if m.get("function_call"):
                calls.append({"id": None, "function": m["function_call"]})
            usage = m.get("usage") or {}
            if i == last_final:
                evs.append(Event(run_id=rid, type="final_answer", content=text,
                                 claimed_success=m.get("claimed_success")))
                continue
            if (text and text.strip()) or usage:
                evs.append(Event(run_id=rid, type="llm_response", content=text or None,
                                 model=m.get("model") if m.get("model") != model else None,
                                 tokens_in=usage.get("prompt_tokens") or usage.get("input_tokens"),
                                 tokens_out=usage.get("completion_tokens") or usage.get("output_tokens")))
            for c in calls:
                if not isinstance(c, dict):
                    continue
                fn = c.get("function") or {}
                name = fn.get("name") or c.get("name")
                cid = c.get("id")
                if cid and name:
                    call_names[cid] = name
                evs.append(Event(run_id=rid, type="tool_call", tool=name, call_id=cid,
                                 args=_args(fn.get("arguments", c.get("arguments")), pr)))
        elif role in ("tool", "function"):
            cid = m.get("tool_call_id")
            name = m.get("name") or call_names.get(cid)
            content = m.get("content")
            if _is_parts(content):      # [{"type":"text","text":"..."}] -> the text, then JSON-decoded
                content = _text(content)
            value = json_or_raw(_text(content)) if not isinstance(content, (dict, list)) else content
            err = None
            if m.get("is_error"):
                text = _text(content) if not isinstance(content, (dict, list)) else json.dumps(content)
                err, value = (text or "error"), None
            evs.append(Event(run_id=rid, type="tool_result", tool=name, call_id=cid, result=value, error=err,
                             status=m.get("status")))
        else:
            pr.warn(f"ignored message with role {role!r}")
    outcome = obj.get("outcome")
    if isinstance(outcome, dict) and "verified_success" in outcome:
        evs.append(Event(run_id=rid, type="run_end", outcome=outcome))
    else:
        evs.append(Event(run_id=rid, type="run_end"))
    return evs


def parse(src: Source) -> ParseResult:
    pr = ParseResult()
    loose_msgs: list[dict] = []
    n = 0
    for lineno, rec in iter_json_records(src, pr, keep_lists=True):
        if isinstance(rec, dict) and _is_run(rec):
            n += 1
            pr.events.extend(run_to_events(rec, f"{src.stem}-{n}", pr))
        elif isinstance(rec, dict) and _is_message(rec):
            loose_msgs.append(rec)
        elif isinstance(rec, list) and rec and all(isinstance(x, dict) and _is_message(x) for x in rec):
            n += 1
            pr.events.extend(run_to_events({"messages": rec}, f"{src.stem}-{n}", pr))
        else:
            pr.skip(lineno, json.dumps(rec, ensure_ascii=False)[:80], "not a run object or message")
    if loose_msgs:
        n += 1
        pr.events.extend(run_to_events({"messages": loose_msgs}, src.stem if n == 1 else f"{src.stem}-{n}", pr))
    return pr


# ----------------------------------------------------------------------------- export
def dump(events: list[Event]) -> str:
    from ..schema import group_runs
    lines = []
    for rid, evs in group_runs(events).items():
        start = next((e for e in evs if e.type == "run_start"), None)
        obj: dict = {"run_id": rid}
        for k in ("agent", "model", "task_type"):
            v = getattr(start, k, None) if start else None
            v = v or next((getattr(e, k) for e in evs if getattr(e, k)), None)
            if v is not None:
                obj[k] = v
        msgs: list[dict] = []
        sp = (start.meta or {}).get("system_prompt") if start else None
        if sp:
            msgs.append({"role": "system", "content": sp})
        open_assistant = None  # assistant msg that may still receive tool_calls
        for e in evs:
            if e.type == "user_input":
                msgs.append({"role": "user", "content": e.content or ""})
                open_assistant = None
            elif e.type == "llm_response" and e.content:
                open_assistant = {"role": "assistant", "content": e.content}
                msgs.append(open_assistant)
            elif e.type == "tool_call":
                if open_assistant is None:
                    open_assistant = {"role": "assistant", "content": None}
                    msgs.append(open_assistant)
                open_assistant.setdefault("tool_calls", []).append({
                    "id": e.call_id or f"call_{e.seq}", "type": "function",
                    "function": {"name": e.tool, "arguments": json.dumps(e.args or {}, ensure_ascii=False)}})
            elif e.type == "tool_result":
                content = e.error if e.error and e.result is None else dump_value_as_text(e.result)
                msg = {"role": "tool", "tool_call_id": e.call_id or "", "content": content}
                if e.tool:
                    msg["name"] = e.tool
                if e.status and e.status != "ok":  # extension keys, ignored by OpenAI itself
                    msg["status"] = e.status
                if e.error and e.result is None:
                    msg["is_error"] = True
                msgs.append(msg)
                open_assistant = None
            elif e.type == "final_answer":
                msgs.append({"role": "assistant", "content": e.content or ""})
                open_assistant = None
        obj["messages"] = msgs
        end = next((e for e in evs if e.type == "run_end" and e.outcome), None)
        if end:
            obj["outcome"] = end.outcome
        lines.append(json.dumps(obj, ensure_ascii=False, default=str))
    return "\n".join(lines) + ("\n" if lines else "")
