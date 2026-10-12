"""Agent Run Log (ARL) v1 -- the canonical event model.

See docs/LOG_FORMAT.md. Every adapter produces a list of `Event`; `finalize_events` then applies
the same normalisation to all of them (seq order, status inference, call pairing, claimed_success).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

EVENT_TYPES = (
    "run_start", "user_input", "llm_request", "llm_response", "tool_call", "tool_result",
    "final_answer", "escalation", "error", "run_end",
)
STATUSES = ("ok", "error", "refused", "timeout")

# Loose spellings seen in the wild -> canonical type.
TYPE_ALIASES = {
    "start": "run_start", "begin": "run_start", "session_start": "run_start",
    "end": "run_end", "finish": "run_end", "session_end": "run_end",
    "user": "user_input", "user_message": "user_input", "input": "user_input",
    "llm_call": "llm_request", "prompt": "llm_request",
    "assistant": "llm_response", "llm_output": "llm_response", "completion": "llm_response",
    "tool_use": "tool_call", "function_call": "tool_call", "action": "tool_call",
    "tool_response": "tool_result", "tool_output": "tool_result", "observation": "tool_result",
    "function_result": "tool_result",
    "final": "final_answer", "answer": "final_answer", "final_response": "final_answer",
    "escalate": "escalation", "handoff": "escalation", "exception": "error",
}
STATUS_ALIASES = {
    "ok": "ok", "success": "ok", "succeeded": "ok", "completed": "ok", "done": "ok", "pass": "ok",
    "true": "ok", "200": "ok", "unset": "ok", "status_code_ok": "ok", "status_code_unset": "ok",
    "error": "error", "err": "error", "failed": "error", "failure": "error", "fail": "error",
    "exception": "error", "false": "error", "status_code_error": "error",
    "refused": "refused", "denied": "refused", "rejected": "refused", "blocked": "refused",
    "forbidden": "refused", "policy": "refused", "not_allowed": "refused",
    "timeout": "timeout", "timed_out": "timeout", "time_out": "timeout",
}

INT_FIELDS = ("seq", "tokens_in", "tokens_out")
FLOAT_FIELDS = ("latency_ms", "cost_usd")
JSON_FIELDS = ("args", "result", "outcome", "meta", "claimed_actions")  # structured values

# A string result/error is a failure when it starts with an error class (+colon), e.g.
# "PolicyError: ...", "NotFound: ...", "Error: ...", "Denied", "Traceback (most recent call last)".
ERROR_TEXT_RE = re.compile(
    r"^\s*(?:(?=[A-Z])[A-Za-z_.]*(?:Error|Exception|NotFound|Denied|Refused|Forbidden|Timeout|"
    r"Unauthorized|Unauthorised)(?:\s*:|\s*$)|Traceback \(most recent call last\))"
)
REFUSAL_TEXT_RE = re.compile(
    r"polic|denied|deny|forbidden|refus|not ?allowed|unauthori[sz]ed|permission|blocked|rejected|"
    r"not permitted|guardrail", re.I)
TIMEOUT_TEXT_RE = re.compile(r"time ?out|timed out|deadline exceeded", re.I)
VERIFY_TOOL_RE = re.compile(r"verif|validat|check|authenticat|auth\b|confirm|kyc", re.I)

@dataclass
class Event:
    run_id: str | None
    type: str
    seq: int | None = None
    ts: str | None = None
    agent: str | None = None
    model: str | None = None
    task_type: str | None = None
    content: str | None = None
    tool: str | None = None
    call_id: str | None = None
    args: dict | None = None
    result: Any = None
    status: str | None = None
    error: str | None = None
    claimed_actions: list | None = None
    claimed_success: bool | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    outcome: dict | None = None
    meta: dict | None = None

    # ---------------------------------------------------------------- dict conversion
    def to_dict(self) -> dict:
        """ARL dict with None fields dropped (key order = spec order)."""
        out = {}
        for f in FIELD_NAMES:
            v = getattr(self, f)
            if v is not None:
                out[f] = v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        """Build an Event from an ARL-like dict. Unknown keys are preserved in `meta`.
        Raises ValueError when the type is missing/unknown."""
        if not isinstance(d, dict):
            raise ValueError(f"event must be a JSON object, got {type(d).__name__}")
        raw_type = d.get("type") if d.get("type") is not None else d.get("event")
        etype = normalize_type(raw_type)
        if etype is None:
            raise ValueError(f"unknown event type {raw_type!r} (expected one of {', '.join(EVENT_TYPES)})")
        kw: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in d.items():
            if k in ("type", "event"):
                continue
            if k == "run":  # common alias
                k = "run_id"
            if k in FIELD_SET:
                kw[k] = v
            else:
                extra[k] = v
        kw = coerce_fields(kw)
        if extra:
            meta = dict(kw.get("meta") or {})
            meta.update(extra)
            kw["meta"] = meta
        return cls(type=etype, **{"run_id": kw.pop("run_id", None), **kw})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# spec order for output
FIELD_NAMES = ["run_id", "seq", "ts", "type", "agent", "model", "task_type", "content", "tool",
               "call_id", "args", "result", "status", "error", "claimed_actions", "claimed_success",
               "tokens_in", "tokens_out", "latency_ms", "cost_usd", "outcome", "meta"]
FIELD_SET = set(FIELD_NAMES)


# ------------------------------------------------------------------------------- coercion
def normalize_type(t) -> str | None:
    if not isinstance(t, str):
        return None
    k = t.strip().lower().replace("-", "_").replace(" ", "_")
    if k in EVENT_TYPES:
        return k
    return TYPE_ALIASES.get(k)


def normalize_status(s) -> str | None:
    if s is None or s == "":
        return None
    if isinstance(s, bool):
        return "ok" if s else "error"
    k = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    return STATUS_ALIASES.get(k)


def _to_bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    if s in ("", "none", "null", "nan"):
        return None
    return None


def coerce_fields(kw: dict) -> dict:
    """Best-effort type coercion so CSV / text sources produce the same values as JSON ones."""
    out = dict(kw)
    for k in INT_FIELDS:
        v = out.get(k)
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, int):
            continue
        try:
            out[k] = int(float(v))
        except (TypeError, ValueError):
            out[k] = None
    for k in FLOAT_FIELDS:
        v = out.get(k)
        if v is None or isinstance(v, (int, float)) and not isinstance(v, bool):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = None
    if "claimed_success" in out:
        out["claimed_success"] = _to_bool(out["claimed_success"])
    for k in ("run_id", "call_id"):
        v = out.get(k)
        if v is not None and not isinstance(v, str):
            out[k] = str(v)
    for k in ("args", "outcome", "meta"):
        v = out.get(k)
        if isinstance(v, str):
            try:
                v2 = json.loads(v)
                out[k] = v2 if isinstance(v2, dict) else {"_raw": v}
            except ValueError:
                out[k] = {"_raw": v}
        elif v is not None and not isinstance(v, dict):
            out[k] = {"_raw": v}
    ca = out.get("claimed_actions")
    if isinstance(ca, str):
        try:
            ca2 = json.loads(ca)
            out["claimed_actions"] = ca2 if isinstance(ca2, list) else [ca]
        except ValueError:
            out["claimed_actions"] = [x.strip() for x in ca.split("|") if x.strip()]
    oc = out.get("outcome")
    if isinstance(oc, dict) and "verified_success" in oc:
        vs = oc["verified_success"]
        b = _to_bool(vs)
        if b is not None:
            oc = dict(oc)
            oc["verified_success"] = int(b)
            out["outcome"] = oc
    return out


# ------------------------------------------------------------------------------- inference
def classify_error_text(text: str) -> str:
    if TIMEOUT_TEXT_RE.search(text or ""):
        return "timeout"
    if REFUSAL_TEXT_RE.search(text or ""):
        return "refused"
    return "error"


def is_error_text(value) -> bool:
    return isinstance(value, str) and bool(ERROR_TEXT_RE.match(value))


def infer_tool_status(ev: Event) -> None:
    """Normalise/infer `status` (and move error text out of `result`) for a tool_result in place."""
    explicit = normalize_status(ev.status)
    if isinstance(ev.result, str) and is_error_text(ev.result) and ev.error is None:
        ev.error, ev.result = ev.result.strip(), None
    elif isinstance(ev.result, dict) and ev.error is None:
        err = ev.result.get("error") or ev.result.get("exception")
        if err and isinstance(err, (str, dict)):
            ev.error = err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
    if explicit is not None:
        ev.status = explicit
        return
    if ev.error:
        ev.status = classify_error_text(ev.error)
    elif ev.result is False and ev.tool and VERIFY_TOOL_RE.search(ev.tool):
        ev.status = "refused"  # verification said "no"
    else:
        ev.status = "ok"


def infer_claimed_success(text: str | None) -> bool | None:
    """True/False from the final-answer text, None when the text gives no signal.
    Same heuristic as the analyzer (see arc/claims.py)."""
    from .claims import claimed_success_or_none
    return claimed_success_or_none(text)


# ------------------------------------------------------------------------------- validation
def validate_event(ev: Event) -> list[str]:
    """Problems with one event (empty list = valid)."""
    p = []
    if not ev.run_id:
        p.append("missing run_id")
    if ev.type not in EVENT_TYPES:
        p.append(f"unknown type {ev.type!r}")
    if ev.seq is not None and (not isinstance(ev.seq, int) or ev.seq < 0):
        p.append(f"bad seq {ev.seq!r}")
    if ev.type in ("tool_call", "tool_result") and not ev.tool:
        p.append(f"{ev.type} without tool name")
    if ev.status is not None and ev.status not in STATUSES:
        p.append(f"bad status {ev.status!r}")
    if ev.args is not None and not isinstance(ev.args, dict):
        p.append("args must be an object")
    if ev.outcome is not None:
        if ev.type != "run_end":
            p.append("outcome only allowed on run_end")
        vs = ev.outcome.get("verified_success") if isinstance(ev.outcome, dict) else None
        if vs not in (None, 0, 1):
            p.append(f"outcome.verified_success must be 0/1, got {vs!r}")
    if ev.claimed_actions is not None and not isinstance(ev.claimed_actions, list):
        p.append("claimed_actions must be a list")
    return p


# ------------------------------------------------------------------------------- run helpers
def group_runs(events: Iterable[Event]) -> dict[str, list[Event]]:
    """run_id -> events (in seq order), runs in order of first appearance."""
    runs: dict[str, list[Event]] = {}
    for e in events:
        runs.setdefault(e.run_id or "", []).append(e)
    for evs in runs.values():
        evs.sort(key=lambda e: (e.seq if e.seq is not None else 10**12))
    return runs


def pair_tool_calls(events: list[Event], stats: dict | None = None) -> list[tuple[Event, Event | None]]:
    """Pair every tool_call of ONE run with its tool_result (None if unanswered).

    Match by call_id when it points at an unanswered call of the same tool; otherwise (no call_id,
    an unknown call_id, or a call_id whose call is for a different tool / already answered) use the
    latest unanswered call of the same tool. ``stats["call_id_mismatch"]`` counts the fallbacks
    that happened despite a call_id."""
    pairs: list[list] = []
    by_id: dict[str, list] = {}
    for e in events:
        if e.type == "tool_call":
            p = [e, None]
            pairs.append(p)
            if e.call_id:
                by_id[e.call_id] = p
        elif e.type == "tool_result":
            p = by_id.get(e.call_id) if e.call_id else None
            if p is not None and (p[1] is not None or (e.tool and p[0].tool and e.tool != p[0].tool)):
                p = None
            if p is None:
                p = next((q for q in reversed(pairs) if q[1] is None and q[0].tool == e.tool), None)
                if p is not None and e.call_id and stats is not None:
                    stats["call_id_mismatch"] = stats.get("call_id_mismatch", 0) + 1
            if p is not None:
                p[1] = e
    return [(c, r) for c, r in pairs]



def finalize_events(events: list[Event], default_run_id: str, warnings: list[str] | None = None) -> list[Event]:
    """Normalise a freshly parsed event list (all adapters go through this).

    - missing run_id -> default_run_id
    - order by seq within each run (missing seq -> file order); seq renumbered 0..n-1
    - tool_result: status inferred / normalised; error text moved out of result
    - tool_call/tool_result paired; missing call_ids filled so pairs are explicit
    - final_answer.claimed_success inferred from text when missing
    """
    warnings = warnings if warnings is not None else []
    for e in events:
        if not e.run_id:
            e.run_id = default_run_id
    out: list[Event] = []
    runs: dict[str, list[tuple[int, Event]]] = {}
    for i, e in enumerate(events):
        runs.setdefault(e.run_id, []).append((i, e))
    renumbered = 0
    mismatched = 0
    for rid, items in runs.items():
        if all(e.seq is not None for _, e in items):
            items.sort(key=lambda t: (t[1].seq, t[0]))
        elif any(e.seq is not None for _, e in items):
            renumbered += 1  # mixed: keep file order
        evs = [e for _, e in items]
        for n, e in enumerate(evs):
            e.seq = n
        mismatched += _normalise_run(evs)
        out.extend(evs)
    if mismatched:
        warnings.append(f"{mismatched} tool_result(s) had a call_id that did not match an open tool_call of the "
                        "same tool; they were paired with the latest unanswered call of that tool instead")
    if renumbered:
        warnings.append(f"{renumbered} run(s) had seq on only some events; file order was used")
    return out


def _normalise_run(evs: list[Event]) -> int:
    """Normalise one run in place. Returns how many results were re-paired despite a call_id."""
    for e in evs:
        if e.type == "tool_result":
            infer_tool_status(e)
        elif e.type == "error":
            e.status = normalize_status(e.status) or ("timeout" if TIMEOUT_TEXT_RE.search(e.error or e.content or "") else "error")
        elif e.status is not None:
            e.status = normalize_status(e.status)
    auto = 0
    stats: dict = {}
    for call, res in pair_tool_calls(evs, stats):
        if res is None:
            continue
        if not call.call_id:
            call.call_id = res.call_id or f"auto-{call.seq}"
            auto += 1
        if not res.call_id:
            res.call_id = call.call_id
        elif res.call_id != call.call_id:   # re-paired by tool name: make the link explicit for the analyzer
            res.meta = {**(res.meta or {}), "original_call_id": res.call_id}
            res.call_id = call.call_id
        if not res.tool:
            res.tool = call.tool
    for e in evs:
        if e.type == "final_answer" and e.claimed_success is None:
            guess = infer_claimed_success(e.content)
            if guess is not None:
                e.claimed_success = guess
                e.meta = {**(e.meta or {}), "claimed_success_inferred": True}
    return stats.get("call_id_mismatch", 0)
