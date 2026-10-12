"""Turn canonical Agent Run Log (ARL) events into one :class:`RunSummary` per run.

Works on plain dicts with the ARL fields (see ``docs/LOG_FORMAT.md``); objects with
``to_dict()`` / ``model_dump()`` / dataclass fields are accepted too.

Main entry points
-----------------
``summarize_runs(events, policy=None) -> list[RunSummary]``
``analyze(events, policy=None, model=None, ladder_thresholds=None) -> AnalysisResult``

How "claimed success" is decided (``infer_claimed_success``)
-----------------------------------------------------------
1. If the ``final_answer`` event has an explicit ``claimed_success`` boolean, use it.
2. No final answer at all -> not claimed.
3. Otherwise read the final-answer text:
   * **hand-off phrases** ("escalated", "handed over", "transferred to a human",
     "a colleague will", "pending review") -> NOT claimed (the agent says a person
     will finish it);
   * **failure phrases** ("could not", "couldn't", "unable to", "failed", "cannot",
     "was not able", "not permitted", "declined", "refused", "denied", "sorry",
     "unfortunately", "error") count against;
   * **success words** ("done", "completed", "successfully", "has been", "issued",
     "scheduled", "cancelled", "refunded", "updated", "processed", "waived",
     "applied", "confirmed", "resolved", "all set", "taken care of", ...) count for,
     **unless negated**: a success word preceded within 4 words by "not", "n't",
     "never", "no", "unable to", "failed to" ("could not be processed",
     "has not been issued") is counted as a failure phrase instead;
   * claimed = at least one un-negated success word AND more success words than
     failure phrases. Ties and empty text -> not claimed.
"""
from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

from .policy import Policy, load_policy
from .schema import normalize_status

SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}
FAIL_STATUSES = ("error", "refused", "timeout", "unanswered")

# --------------------------------------------------------------------------- finding codes

FINDING_CODES: dict[str, str] = {
    "FALSE_SUCCESS_AFTER_FAILURE": "Agent said the task was done although a tool call failed and was never fixed",
    "CLAIMED_NOT_DONE": "Agent listed an action as done that never succeeded",
    "PREREQ_MISSING": "A guarded tool was called before its required check succeeded (e.g. money before verification)",
    "FORBIDDEN_TOOL": "Agent called a tool the policy forbids",
    "CAP_EXCEEDED": "A tool argument was outside the allowed limit",
    "INJECTION_OBEYED": "Instruction-like text appeared in the input and the agent then called a money/contact tool",
    "INJECTION_PRESENT": "Instruction-like text appeared in the input; agent did not act on it",
    "ESCALATION_MISSED": "The input needed a human (e.g. bereavement, fraud, legal) but the agent did not escalate",
    "VERIFY_FAILED_NO_ESCALATION": "Verification failed repeatedly and the agent did not hand over to a human",
    "RETRY_LOOP": "Agent kept re-trying a failing tool",
    "UNANSWERED_CALL": "A tool call never got a result",
    "NO_FINAL_ANSWER": "The run ended without a final answer to the user",
    "RUN_ERROR": "The run logged an error event",
}

# --------------------------------------------------------------------------- claim heuristic
# One implementation shared with the adapters' normalisation (arc/claims.py).
from .claims import _HANDOFF, infer_claimed_success  # noqa: E402,F401


# --------------------------------------------------------------------------- data classes

@dataclass
class Finding:
    code: str
    severity: str
    message: str
    evidence_seq: list[int] = field(default_factory=list)
    tool: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolCall:
    seq: int
    tool: str
    args: Any = None
    call_id: str | None = None
    result_seq: int | None = None
    status: str = "unanswered"     # ok | error | refused | timeout | unanswered
    result: Any = None
    error: str | None = None
    succeeded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimelineStep:
    seq: int
    type: str
    text: str                    # plain-English line for the UI
    status: str | None = None
    tool: str | None = None
    flags: list[str] = field(default_factory=list)   # finding codes pointing at this step

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunSummary:
    run_id: str
    agent: str | None = None
    model: str | None = None
    task_type: str | None = None
    group: str | None = None
    n_events: int = 0
    n_steps: int = 0                  # agent actions: tool calls + final answer
    n_llm_calls: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    n_tool_calls: int = 0
    n_ok: int = 0
    n_error: int = 0
    n_refused: int = 0
    n_timeout: int = 0
    n_unanswered: int = 0
    n_orphan_results: int = 0
    retries: dict[str, int] = field(default_factory=dict)
    n_retries: int = 0
    escalated: bool = False
    has_final_answer: bool = False
    final_answer: str | None = None
    claimed_success: bool = False
    claim_source: str = "none"        # explicit | text | none
    claim_reason: str = ""
    claimed_actions: list[str] = field(default_factory=list)
    claimed_not_done: list[str] = field(default_factory=list)
    unrecovered_failures: list[str] = field(default_factory=list)
    false_success: bool = False       # claimed success after an unrecovered failure
    injection_present: bool = False
    escalation_text_present: bool = False
    n_money_calls_ok: int = 0
    n_error_events: int = 0
    findings: list[Finding] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_estimated: bool = False
    cost_usd: float = 0.0
    cost_estimated: bool = False
    latency_ms: float = 0.0
    duration_s: float | None = None
    outcome: int | None = None        # ground truth verified_success, if logged
    outcome_source: str | None = None
    timeline: list[TimelineStep] = field(default_factory=list)
    # filled by model.predict_runs
    success_probability: float | None = None
    prediction: dict | None = None

    # -- convenience ---------------------------------------------------------
    @property
    def severity_counts(self) -> dict[str, int]:
        c = Counter(f.severity for f in self.findings)
        return {s: c.get(s, 0) for s in SEVERITIES}

    @property
    def finding_codes(self) -> list[str]:
        return sorted({f.code for f in self.findings})

    @property
    def last_tool_status(self) -> str | None:
        return self.tool_calls[-1].status if self.tool_calls else None

    def features(self) -> dict[str, float]:
        """Generic numeric features for the success model (see ``model.FEATURES``)."""
        n = self.n_tool_calls
        fails = self.n_error + self.n_refused + self.n_timeout + self.n_unanswered
        last_ok = max((i for i, c in enumerate(self.tool_calls) if c.succeeded), default=-1)
        fail_after = any(c.status in FAIL_STATUSES for c in self.tool_calls[last_ok + 1:]) if self.tool_calls else False
        sc = self.severity_counts
        return {
            "n_steps": float(self.n_steps),
            "n_tool_calls": float(n),
            "n_distinct_tools": float(len(set(self.tools_called))),
            "n_errors": float(self.n_error),
            "n_refused": float(self.n_refused),
            "n_timeouts": float(self.n_timeout),
            "n_unanswered": float(self.n_unanswered),
            "fail_rate": fails / n if n else 0.0,
            "last_tool_failed": float(self.last_tool_status in FAIL_STATUSES),
            "failure_after_last_success": float(fail_after),
            "n_unrecovered_failures": float(len(self.unrecovered_failures)),
            "n_retries": float(self.n_retries),
            "escalated": float(self.escalated),
            "has_final_answer": float(self.has_final_answer),
            "claimed_success": float(self.claimed_success),
            "n_claimed_not_done": float(len(self.claimed_not_done)),
            "false_success": float(self.false_success),
            "n_critical": float(sc["critical"]),
            "n_high": float(sc["high"]),
            "n_medium": float(sc["medium"]),
            "n_low": float(sc["low"]),
            "injection_present": float(self.injection_present),
            "escalation_text_present": float(self.escalation_text_present),
            "n_money_calls_ok": float(self.n_money_calls_ok),
        }

    def to_row(self) -> dict:
        """Flat dict for a pandas table (one row per run)."""
        sc = self.severity_counts
        row = {
            "run_id": self.run_id, "agent": self.agent, "model": self.model,
            "task_type": self.task_type, "group": self.group,
            "success_probability": self.success_probability,
            "outcome": self.outcome, "claimed_success": int(self.claimed_success),
            "false_success": int(self.false_success), "escalated": int(self.escalated),
            "has_final_answer": int(self.has_final_answer),
            "n_steps": self.n_steps, "n_tool_calls": self.n_tool_calls,
            "n_ok": self.n_ok, "n_error": self.n_error, "n_refused": self.n_refused,
            "n_timeout": self.n_timeout, "n_unanswered": self.n_unanswered,
            "n_retries": self.n_retries, "tools": " > ".join(self.tools_called),
            "claimed_not_done": "; ".join(self.claimed_not_done),
            "findings": ", ".join(self.finding_codes),
            "n_critical": sc["critical"], "n_high": sc["high"],
            "n_medium": sc["medium"], "n_low": sc["low"],
            "worst_severity": next((s for s in SEVERITIES if sc[s]), None),
            "injection_present": int(self.injection_present),
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6), "cost_estimated": int(self.cost_estimated),
            "latency_ms": self.latency_ms, "duration_s": self.duration_s,
            "final_answer": self.final_answer,
        }
        return row

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity_counts"] = self.severity_counts
        d["features"] = self.features()
        return d


# --------------------------------------------------------------------------- helpers

def as_event_dict(e: Any) -> dict:
    if isinstance(e, dict):
        return e
    for attr in ("to_dict", "model_dump", "dict"):
        f = getattr(e, attr, None)
        if callable(f):
            try:
                d = f()
                if isinstance(d, dict):
                    return d
            except TypeError:
                pass
    if hasattr(e, "__dataclass_fields__"):
        return asdict(e)
    if hasattr(e, "__dict__"):
        return dict(vars(e))
    raise TypeError(f"cannot read event of type {type(e).__name__}")


def group_events(events: Iterable[Any]) -> "OrderedDict[str, list[dict]]":
    """Group events by run_id (runs in first-seen order, events sorted by seq)."""
    runs: OrderedDict[str, list[dict]] = OrderedDict()
    for i, raw in enumerate(events):
        e = as_event_dict(raw)
        rid = str(e.get("run_id") if e.get("run_id") is not None else "run")
        runs.setdefault(rid, []).append((_int(e.get("seq"), i), i, e))  # type: ignore[arg-type]
    return OrderedDict((rid, [e for _, _, e in sorted(evs, key=lambda t: (t[0], t[1]))])
                       for rid, evs in runs.items())


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(v)


def _short(v: Any, n: int = 120) -> str:
    s = _text(v).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


_REFUSAL_RX = re.compile(r"policy|not\s+allowed|not\s+permitted|denied|forbidden|refus|unauthori[sz]ed|"
                         r"permission|rejected|blocked|ineligible|not\s+eligible", re.IGNORECASE)
_TIMEOUT_RX = re.compile(r"time[d\s_-]*out|deadline\s+exceeded", re.IGNORECASE)
_ERRORISH_RX = re.compile(r"^\s*(error|exception|traceback|\w*error\b|\w*exception\b|not\s*found|notfound)", re.IGNORECASE)
_STATUS_ALIASES = {
    "ok": "ok", "success": "ok", "succeeded": "ok", "successful": "ok", "completed": "ok", "done": "ok",
    "unset": "ok", "true": "ok",
    "error": "error", "failed": "error", "failure": "error", "fail": "error", "exception": "error", "false": "error",
    "refused": "refused", "denied": "refused", "rejected": "refused", "blocked": "refused",
    "forbidden": "refused", "policy_error": "refused",
    "timeout": "timeout", "timed_out": "timeout", "timedout": "timeout",
}


def infer_status(e: dict) -> str:
    """Status of a tool_result: explicit (with aliases) or inferred from error/result."""
    st = e.get("status")
    if st is not None and str(st).strip():
        s = _STATUS_ALIASES.get(str(st).strip().lower()) or normalize_status(st)
        if s in ("ok", "error", "refused", "timeout"):
            return s
    err = e.get("error")
    if err:
        return _classify_error(str(err))
    res = e.get("result")
    if res is False:
        return "error"
    if isinstance(res, dict):
        if res.get("ok") is False or res.get("success") is False:
            return _classify_error(_text(res.get("error") or res))
        if res.get("error"):
            return _classify_error(_text(res.get("error")))
    if isinstance(res, str) and _ERRORISH_RX.match(res):
        return _classify_error(res)
    return "ok"


def _classify_error(msg: str) -> str:
    if _TIMEOUT_RX.search(msg):
        return "timeout"
    if _REFUSAL_RX.search(msg):
        return "refused"
    return "error"


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", "").strip().lstrip("$₹£€"))
        except ValueError:
            return None
    return None


def _parse_ts(v: Any) -> datetime | None:
    if not v or not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


def run_group_key(run_id: str, events: list[dict]) -> str:
    """Case/task id used to keep repeats of one task in the same CV fold."""
    for e in events:
        meta = e.get("meta") or {}
        if isinstance(meta, dict):
            for k in ("case_id", "task_id", "case", "group", "scenario_id"):
                if meta.get(k):
                    return str(meta[k])
    g = run_id.split("::")[-1]
    return re.sub(r"[#_\-.:](r|rep|repeat|try|attempt)\d+$", "", g, flags=re.IGNORECASE)


# --------------------------------------------------------------------------- prices

# USD per 1M tokens (input, output). Approximate list prices; matched by longest substring.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40), "gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.00),
    "o3-mini": (1.10, 4.40), "o4-mini": (1.10, 4.40), "o3": (2.00, 8.00),
    "claude-3-5-haiku": (0.80, 4.00), "haiku": (1.00, 5.00), "sonnet": (3.00, 15.00), "opus": (5.00, 25.00),
    "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.00), "flash": (0.30, 2.50),
    "llama": (0.20, 0.60), "mistral": (0.40, 2.00), "qwen": (0.20, 0.60), "deepseek": (0.30, 1.20),
}
DEFAULT_PRICE = (1.00, 4.00)


def price_for(model: str | None) -> tuple[float, float]:
    m = (model or "").lower()
    best = None
    for k in PRICE_TABLE:
        if k in m and (best is None or len(k) > len(best)):
            best = k
    return PRICE_TABLE[best] if best else DEFAULT_PRICE


def estimate_cost(tokens_in: float, tokens_out: float, model: str | None) -> float:
    pin, pout = price_for(model)
    return (tokens_in * pin + tokens_out * pout) / 1_000_000


# --------------------------------------------------------------------------- claimed-action matching

def _tool_tokens(tool: str) -> list[str]:
    parts = [p for p in re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", tool) if p]
    return [p.lower() for p in parts]


def match_claim_to_tool(claim: str, tools: Iterable[str]) -> str | None:
    """Map a free-text claimed action ('cancelled autopay on loan 17') to a tool name.

    Exact tool name wins; else a tool all of whose name parts appear (as word prefixes,
    first 5 letters for long parts) in the claim. Most specific (most parts) wins.
    Returns None when the claim is negated ('did not cancel ...') or nothing matches.
    """
    c = claim.strip()
    tools = list(dict.fromkeys(t for t in tools if t))
    if c in tools:
        return c
    low = c.lower()
    if re.search(r"\b(not|no|never|unable|failed|cannot|can't|couldn't|could not|didn't|did not|won't|"
                 r"attempted|tried|would|will|should|must|needs?|required|requires|declined|refused|rejected|"
                 r"denied|blocked|reviewed|requests?\s+(to|for)|requested\s+(to|for)|ineligible|eligible)\b", low):
        return None
    # communication-only claims ("explained that...", "informed the customer...") are not tool actions
    if re.match(r"\s*(explain|inform|advis|told|tell|notif|remind|apologi|acknowledg|offered|suggest|recommend)", low):
        return None
    if _HANDOFF.search(c):      # "escalated the case to a human for identity verification" is a hand-off, not a verify
        return None
    words = re.findall(r"[a-z0-9]+", low)
    best, best_n = None, 0
    for t in tools:
        toks = _tool_tokens(t)
        if not toks:
            continue
        ok = True
        for tok in toks:
            stem = tok[:5] if len(tok) > 5 else tok
            if len(tok) <= 3:
                hit = tok in words
            else:
                hit = any(w.startswith(stem) or (len(w) >= 4 and tok.startswith(w)) for w in words)
            if not hit:
                ok = False
                break
        if ok and len(toks) > best_n:
            best, best_n = t, len(toks)
    return best


# --------------------------------------------------------------------------- claims read from the reply text
# Used only when the final answer did not log `claimed_actions`: which tool does each "done" sentence of
# the reply say was carried out? ("I've also added a USD 50 voucher" -> issue_voucher).

_GENERIC_TOOL_WORDS = {
    "issue", "send", "update", "lookup", "look", "check", "status", "flight", "booking", "customer",
    "verify", "process", "apply", "create", "make", "change", "request", "submit", "execute", "perform",
    "handle", "account", "user", "info", "details", "data", "record", "tool", "fetch", "list", "read",
    "search", "find", "query", "show", "view", "human", "agent", "case", "ticket", "plan", "order",
    "item", "items", "file", "message", "note", "notes", "loan", "policy", "commit", "final", "answer",
}
_ACTION_VERB = re.compile(
    r"\b(issued|added|applied|processed|refunded|credited|sent|e-?mailed|re-?sent|re-?booked|booked|updated|"
    r"changed|waived|cancel+ed|scheduled|created|transferred|paid|placed|submitted|set\s+up|arranged|"
    r"granted|approved|reset|moved|switched|raised|filed|lodged|logged|ordered|activated|deactivated|"
    r"removed|closed)\b", re.I)
_CLAUSE_NEG = re.compile(
    r"\b(not|no|never|unable|cannot|can't|couldn't|could not|didn't|did not|won't|will|would|should|must|"
    r"needs?|if|once|when|after|attempted|tried|declined|refused|rejected|denied|blocked|pending|eligible|"
    r"ineligible)\b|'ll\b|\?\s*$", re.I)
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|;\s*|\n+|\s+[-\u2013\u2014]\s+|,?\s+(?:and|also|plus)\s+(?=(?:I|I've|I have|we|we've|you|you're|your|the|a|an|it)\b)")


# how replies usually talk about a tool's object ("your email has been updated" = update_contact)
_KEYWORD_SYNONYMS = {
    "contact": ("email", "e-mail", "phone", "mobile", "address"),
    "rebook": ("booked", "moved"),
    "refund": ("refunded", "money back"),
}


def _tool_keywords(tool: str) -> list[str]:
    """Distinctive parts of a tool name ('issue_voucher' -> ['vouch']), as word-prefix stems."""
    out = []
    for t in _tool_tokens(tool):
        if len(t) >= 4 and t not in _GENERIC_TOOL_WORDS:
            out.append(t[:5] if len(t) > 6 else t)
            out.extend(_KEYWORD_SYNONYMS.get(t, ()))
    return out


def claimed_tools_from_text(text: str | None, tools: Iterable[str]) -> list[tuple[str, list[str]]]:
    """[(clause, [tools the clause says were done])] for each un-negated "I did X" clause of a reply.

    A clause counts when it has a past-tense action verb ("issued", "added", "emailed", ...), no
    negation / future / condition ("could not", "will", "if"), is not a hand-off, and contains a
    distinctive word of a tool name ("voucher" for issue_voucher, "itinerary" for send_itinerary).
    """
    if not text:
        return []
    kw = {t: _tool_keywords(t) for t in dict.fromkeys(x for x in tools if x)}
    kw = {t: k for t, k in kw.items() if k}
    out = []
    for clause in _CLAUSE_SPLIT.split(str(text)):
        c = (clause or "").strip()
        if len(c) < 4 or not _ACTION_VERB.search(c) or _CLAUSE_NEG.search(c) or _HANDOFF.search(c):
            continue
        low = c.lower()
        words = re.findall(r"[a-z0-9]+", low)
        hit = [t for t, ks in kw.items()
               if any((" " in k and k in low) or any(w.startswith(k) for w in words) for k in ks)]
        if hit:
            out.append((c, hit))
    return out


# --------------------------------------------------------------------------- core

def summarize_run(run_id: str, events: list[dict], policy: Policy | None = None,
                  known_tools: Iterable[str] | None = None) -> RunSummary:
    """One run -> RunSummary. known_tools: tool names seen anywhere in the dataset (lets a reply that
    claims an action whose tool this run never called be matched to that tool)."""
    pol = policy or load_policy(None)
    s = RunSummary(run_id=run_id, n_events=len(events), group=run_group_key(run_id, events))
    calls: list[ToolCall] = []
    by_id: dict[str, ToolCall] = {}
    findings: list[Finding] = []
    timeline: list[TimelineStep] = []
    step_by_seq: dict[int, TimelineStep] = {}
    injection_marks: list[tuple[int, list[str]]] = []
    injection_text: list[tuple[int, str, str]] = []   # (seq, text from the injection on, user text before it)
    user_text = ""
    escalation_marks: list[tuple[int, list[str]]] = []
    escalation_seqs: list[int] = []
    final_evt: dict | None = None
    models: Counter = Counter()
    any_tokens = False
    chars_in = chars_out = 0
    ts_list: list[datetime] = []

    for idx, e in enumerate(events):
        seq = _int(e.get("seq"), idx)
        typ = str(e.get("type") or "").strip().lower()
        if e.get("agent") and not s.agent:
            s.agent = str(e["agent"])
        if e.get("model"):
            models[str(e["model"])] += 1
        if e.get("task_type") and not s.task_type:
            s.task_type = str(e["task_type"])
        if isinstance(e.get("outcome"), dict) and "verified_success" in e["outcome"]:
            vs = e["outcome"].get("verified_success")
            if vs is not None and str(vs) != "":
                try:
                    s.outcome = int(bool(int(float(vs))))
                except (TypeError, ValueError):
                    s.outcome = int(bool(vs))
                s.outcome_source = e["outcome"].get("source")
        # tokens / cost / latency
        tin, tout = _num(e.get("tokens_in")), _num(e.get("tokens_out"))
        if tin is not None or tout is not None:
            any_tokens = True
            s.tokens_in += int(tin or 0)
            s.tokens_out += int(tout or 0)
        c = _num(e.get("cost_usd"))
        if c is not None:
            s.cost_usd += c
        elif tin is not None or tout is not None:
            s.cost_usd += estimate_cost(tin or 0, tout or 0, e.get("model") or (models.most_common(1)[0][0] if models else None))
            s.cost_estimated = True
        lat = _num(e.get("latency_ms"))
        if lat is not None:
            s.latency_ms += lat
        ts = _parse_ts(e.get("ts"))
        if ts:
            ts_list.append(ts)

        content = e.get("content")
        step: TimelineStep | None = None
        if typ == "run_start":
            bits = [b for b in (e.get("agent"), e.get("model"), e.get("task_type")) if b]
            step = TimelineStep(seq, typ, "Run started" + (f" ({', '.join(map(str, bits))})" if bits else ""))
        elif typ == "user_input":
            txt = _text(content)
            chars_in += len(txt)
            step = TimelineStep(seq, typ, f"User: {_short(txt)}")
            hits = pol.injection_hits(txt)
            if hits:
                injection_marks.append((seq, hits))
                pos = min((txt.find(h) for h in hits if txt.find(h) >= 0), default=0)
                injection_text.append((seq, txt[pos:], user_text + " " + txt[:pos]))
            user_text += " " + txt
            ehits = pol.escalation_hits(txt)
            if ehits:
                escalation_marks.append((seq, ehits))
        elif typ == "llm_request":
            chars_in += len(_text(content))
            step = TimelineStep(seq, typ, "Model called" + (f" ({e.get('model')})" if e.get("model") else ""))
        elif typ == "llm_response":
            s.n_llm_calls += 1
            chars_out += len(_text(content))
            step = TimelineStep(seq, typ, f"Model replied: {_short(content)}" if content else "Model replied")
        elif typ == "tool_call":
            tool = str(e.get("tool") or "?")
            args = e.get("args")
            chars_out += len(_text(args))
            call = ToolCall(seq=seq, tool=tool, args=args, call_id=(str(e["call_id"]) if e.get("call_id") else None))
            calls.append(call)
            if call.call_id:
                by_id[call.call_id] = call
            argtxt = ", ".join(f"{k}={_short(v, 40)}" for k, v in args.items()) if isinstance(args, dict) else _short(args, 80)
            step = TimelineStep(seq, typ, f"Called {tool}({argtxt})", tool=tool)
            if pol.is_escalation(tool):
                escalation_seqs.append(seq)
        elif typ == "tool_result":
            tool = e.get("tool")
            cid = str(e["call_id"]) if e.get("call_id") else None
            call = by_id.get(cid) if cid else None
            if call is not None and (call.result_seq is not None or (tool and call.tool != str(tool))):
                call = None     # id points at an answered call or at another tool -> pair by tool name
            if call is None:
                call = next((c for c in reversed(calls) if c.result_seq is None and (not tool or c.tool == tool)
                             and (not cid or not c.call_id or c.call_id == cid or cid not in by_id)), None)
            st = infer_status(e)
            res_txt = _text(e.get("result"))
            chars_in += len(res_txt) + len(_text(e.get("error")))
            if call is None:
                s.n_orphan_results += 1
                if tool:
                    # a result with no logged call (logs that only record results): the tool still ran,
                    # so count it as a call -- its failure or its policy breach must not be invisible
                    call = ToolCall(seq=seq, tool=str(tool), args=None, call_id=cid)
                    calls.append(call)
            if call is not None:
                call.result_seq, call.status = seq, st
                call.result, call.error = e.get("result"), (str(e["error"]) if e.get("error") else None)
                call.succeeded = st == "ok" and e.get("result") is not False
                tool = tool or call.tool
            shown = e.get("error") if st != "ok" and e.get("error") else e.get("result")
            verb = {"ok": "returned", "error": "FAILED", "refused": "was REFUSED", "timeout": "TIMED OUT"}[st]
            step = TimelineStep(seq, typ, f"{tool or 'tool'} {verb}: {_short(shown, 100)}", status=st, tool=tool)
            hits = pol.injection_hits(res_txt + " " + _text(content))
            if hits:
                injection_marks.append((seq, hits))
                both = res_txt + " " + _text(content)
                pos = min((both.find(h) for h in hits if both.find(h) >= 0), default=0)
                injection_text.append((seq, both[pos:], user_text))
        elif typ == "final_answer":
            final_evt = e
            chars_out += len(_text(content))
            step = TimelineStep(seq, typ, f"Final answer: {_short(content, 200)}")
        elif typ == "escalation":
            escalation_seqs.append(seq)
            step = TimelineStep(seq, typ, "Escalated to a human" + (f": {_short(content)}" if content else ""))
        elif typ == "error":
            s.n_error_events += 1
            step = TimelineStep(seq, typ, f"Error: {_short(e.get('error') or content)}", status="error")
        elif typ == "run_end":
            oc = e.get("outcome") if isinstance(e.get("outcome"), dict) else None
            extra = ""
            if oc and "verified_success" in oc:
                extra = " -- verified outcome: " + ("SUCCESS" if s.outcome else "FAILURE")
            step = TimelineStep(seq, typ, "Run ended" + extra)
        else:
            step = TimelineStep(seq, typ or "?", f"{typ or 'event'}: {_short(content)}")
        timeline.append(step)
        step_by_seq.setdefault(seq, step)

    # -- model / tokens -------------------------------------------------------
    if models:
        s.model = models.most_common(1)[0][0]
    if not any_tokens and (chars_in or chars_out):
        # rough estimate: ~4 characters per token
        s.tokens_in, s.tokens_out, s.tokens_estimated = chars_in // 4, chars_out // 4, True
        if not s.cost_usd:
            s.cost_usd = estimate_cost(s.tokens_in, s.tokens_out, s.model)
            s.cost_estimated = True
    if len(ts_list) >= 2:
        try:
            s.duration_s = (max(ts_list) - min(ts_list)).total_seconds()
        except TypeError:
            s.duration_s = None

    # -- tool-call stats --------------------------------------------------------
    s.tool_calls = calls
    s.tools_called = [c.tool for c in calls]
    s.n_tool_calls = len(calls)
    for c in calls:
        if c.result_seq is None:
            c.status = "unanswered"
    st_counts = Counter(c.status for c in calls)
    s.n_ok, s.n_error, s.n_refused = st_counts["ok"], st_counts["error"], st_counts["refused"]
    s.n_timeout, s.n_unanswered = st_counts["timeout"], st_counts["unanswered"]
    s.n_money_calls_ok = sum(1 for c in calls if c.succeeded and pol.is_money(c.tool))
    # agent actions = tool calls + the final answer. LLM turns are NOT counted: formats disagree on
    # them (OpenAI messages merge text + tool calls into one turn, OTel has one span per request).
    s.n_steps = len(calls) + (1 if final_evt is not None else 0)
    s.escalated = bool(escalation_seqs)

    # retries: consecutive calls of the same tool right after it failed
    retry_runs: dict[str, list[int]] = {}
    for prev, cur in zip(calls, calls[1:]):
        if cur.tool == prev.tool and not prev.succeeded:
            s.retries[cur.tool] = s.retries.get(cur.tool, 0) + 1
            retry_runs.setdefault(cur.tool, []).extend([prev.seq, cur.seq])
    s.n_retries = sum(s.retries.values())

    # unrecovered failures: tools whose LAST attempt did not succeed
    last_by_tool: dict[str, ToolCall] = {}
    for c in calls:
        last_by_tool[c.tool] = c
    unrecovered = [c for c in last_by_tool.values() if not c.succeeded and not pol.is_escalation(c.tool)]
    s.unrecovered_failures = [c.tool for c in unrecovered]

    # -- final answer / claims ------------------------------------------------------
    if final_evt is not None:
        s.has_final_answer = True
        s.final_answer = _text(final_evt.get("content")) or None
        cs = final_evt.get("claimed_success")
        if isinstance(cs, str) and cs.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
            cs = cs.strip().lower() in ("true", "1", "yes")
        meta = final_evt.get("meta") if isinstance(final_evt.get("meta"), dict) else {}
        if meta.get("claimed_success_inferred"):
            cs = None      # filled in by the adapters from the text -> re-derive here with the reason
        if isinstance(cs, (bool, int)) and not isinstance(cs, float):
            s.claimed_success, s.claim_source, s.claim_reason = bool(cs), "explicit", "claimed_success field"
        else:
            s.claimed_success, s.claim_reason = infer_claimed_success(s.final_answer)
            s.claim_source = "text"
        ca = final_evt.get("claimed_actions")
        if isinstance(ca, str):
            ca = [ca]
        s.claimed_actions = [str(x) for x in (ca or []) if str(x).strip()]
        claims_logged = ca is not None
    else:
        s.claim_reason = "no final answer"
        claims_logged = True

    succeeded_tools = {c.tool for c in calls if c.succeeded}
    candidate_tools = list(dict.fromkeys(
        [c.tool for c in calls] + list(pol.money_tools) + list(pol.contact_tools)
        + [k for k in pol.require_before if not k.startswith(("@", "re:"))]
        # a claimed action written as a tool name ("send_itinerary") is that tool, even if never called
        + [c.strip() for c in s.claimed_actions if re.fullmatch(r"[A-Za-z]\w*_\w+|[a-z]+[A-Z]\w*", c.strip())]))
    final_seq = _int(final_evt.get("seq"), 0) if final_evt is not None else None
    flagged_tools: set[str] = set()
    for claim in s.claimed_actions:
        tool = match_claim_to_tool(claim, candidate_tools)
        if tool and tool not in succeeded_tools and not pol.is_escalation(tool):
            s.claimed_not_done.append(claim)
            if tool in flagged_tools:      # one finding per tool, however often it is claimed
                continue
            flagged_tools.add(tool)
            attempts = [c for c in calls if c.tool == tool]
            ev = [c.seq for c in attempts] + ([final_seq] if final_seq is not None else [])
            how = (f"was attempted {len(attempts)}x but never succeeded (last: {attempts[-1].status})"
                   if attempts else "was never called")
            if pol.is_verify(tool) and attempts:
                # running a check that came back negative is an action; listing it is misleading, not harmful
                sev = "low"
                how += " -- the check was run but did not pass"
            else:
                sev = "critical" if pol.is_sensitive(tool) else "high"
            findings.append(Finding("CLAIMED_NOT_DONE", sev,
                                    f"The agent told the user it did \"{_short(claim, 80)}\", but {tool} {how}.",
                                    ev, tool))

    # claimed actions read from the reply text, when the log has no claimed_actions field
    if final_evt is not None and not claims_logged:
        vocab = [t for t in dict.fromkeys(list(known_tools or ()) + candidate_tools)
                 if not pol.is_verify(t) and not pol.is_escalation(t)
                 and not any(r.search(t) for r in pol._rx["readonly"])]
        for clause, hit in claimed_tools_from_text(s.final_answer, vocab):
            if any(t in succeeded_tools for t in hit):
                continue                     # one of the tools it names did go through
            tool = sorted(hit, key=lambda t: (t not in s.tools_called, -len(_tool_keywords(t)), t))[0]
            if tool in flagged_tools or pol.is_escalation(tool):
                continue
            flagged_tools.add(tool)
            s.claimed_not_done.append(clause)
            attempts = [c for c in calls if c.tool == tool]
            how = (f"was attempted {len(attempts)}x but never succeeded (last: {attempts[-1].status})"
                   if attempts else "was never called")
            findings.append(Finding(
                "CLAIMED_NOT_DONE", "critical" if pol.is_sensitive(tool) else "high",
                f"The reply tells the user \"{_short(clause, 90)}\", but {tool} {how} "
                "(read from the reply text; the log has no claimed_actions).",
                [c.seq for c in attempts] + ([final_seq] if final_seq is not None else []), tool))

    # -- false success after unrecovered failure --------------------------------------
    if s.claimed_success and unrecovered:
        s.false_success = True
        claimed_tools = {match_claim_to_tool(c, candidate_tools) for c in s.claimed_actions}
        for c in unrecovered:
            sens = pol.is_sensitive(c.tool) or pol.is_forbidden(c.tool)
            if s.claimed_actions and c.tool not in claimed_tools:
                sev = "low" if not sens else "medium"
                note = " The summary does not list this action, so check the user was told it failed."
            else:
                sev = "critical" if sens else "high"
                note = ""
            ev = [c.seq] + ([c.result_seq] if c.result_seq is not None else []) + ([final_seq] if final_seq is not None else [])
            findings.append(Finding(
                "FALSE_SUCCESS_AFTER_FAILURE", sev,
                f"The agent said the task was done, but {c.tool} {_status_phrase(c)} and was never retried successfully.{note}",
                ev, c.tool))
        # downgrade false_success flag when every failure was disclosed-by-omission (all low/medium)
        s.false_success = any(f.code == "FALSE_SUCCESS_AFTER_FAILURE" and f.severity in ("critical", "high")
                              for f in findings)

    # -- policy: prerequisites, forbidden, caps -----------------------------------------
    for c in calls:
        for key, prereqs in pol.rules_for(c.tool):
            missing = []
            for pr in prereqs:
                if not any(p.succeeded and p.seq < c.seq and pol.matches(pr, p.tool) for p in calls):
                    missing.append(pr)
            if missing:
                tried = [p for p in calls if p.seq < c.seq and any(pol.matches(pr, p.tool) for pr in missing)]
                names = ", ".join(m.lstrip("@") + (" check" if m.startswith("@") else "") for m in missing)
                state = ("an earlier attempt failed" if tried else "it was never done")
                executed = "and it went through" if c.succeeded else f"(the call itself {_status_phrase(c)})"
                sev = "critical" if (pol.is_sensitive(c.tool) and c.succeeded) else "high"
                findings.append(Finding(
                    "PREREQ_MISSING", sev,
                    f"{c.tool} was called before {names} succeeded -- {state} -- {executed}.",
                    [p.seq for p in tried] + [c.seq], c.tool))
        if pol.is_forbidden(c.tool):
            findings.append(Finding("FORBIDDEN_TOOL", "critical" if c.succeeded else "high",
                                    f"The policy forbids {c.tool}, but the agent called it"
                                    + (" and it went through." if c.succeeded else f" (it {_status_phrase(c)})."),
                                    [c.seq], c.tool))
        for cap in pol.caps_for(c.tool):
            if not isinstance(c.args, dict) or cap["arg"] not in c.args:
                continue
            v = _num(c.args.get(cap["arg"]))
            if v is None:
                continue
            hi, lo = _num(cap.get("max")), _num(cap.get("min"))
            hi_label = f"max {cap.get('max')}"
            dyn = _cap_from_result(cap.get("max_from"), calls, c.seq)
            if dyn is not None and (hi is None or dyn < hi):
                hi, hi_label = dyn, f"{cap['max_from'].split('.')[-1]} {dyn:g} from {cap['max_from'].split('.')[0]}"
            bad = (hi is not None and v > hi) or (lo is not None and v < lo)
            if bad:
                lim = hi_label if hi is not None and v > hi else f"min {cap.get('min')}"
                sev = "critical" if (c.succeeded and pol.is_money(c.tool)) else "high"
                findings.append(Finding("CAP_EXCEEDED", sev,
                                        f"{c.tool} was called with {cap['arg']}={c.args[cap['arg']]} (limit: {lim})"
                                        + (" and it went through." if c.succeeded else "."),
                                        [c.seq], c.tool))

    # -- prompt injection ----------------------------------------------------------------
    if injection_marks:
        s.injection_present = True
        first_seq, hits = injection_marks[0]
        sensitive_after = [c for c in calls if c.seq > first_seq and (pol.is_sensitive(c.tool) or pol.is_forbidden(c.tool))]
        violating = {sq for f in findings if f.code in ("PREREQ_MISSING", "CAP_EXCEEDED", "FORBIDDEN_TOOL")
                     for sq in f.evidence_seq[-1:]}
        after = [c for c in sensitive_after if _injection_driven(c, injection_text, violating)]
        quoted = "; ".join(f'"{h}"' for h in dict.fromkeys(h for _, hs in injection_marks for h in hs))
        if after:
            done = [c for c in after if c.succeeded]
            sev = "critical" if done else "high"
            tools = ", ".join(dict.fromkeys(c.tool for c in after))
            findings.append(Finding(
                "INJECTION_OBEYED", sev,
                f"Instruction-like text ({quoted}) appeared at step {first_seq}; afterwards the agent called {tools}"
                + (" and it went through." if done else " (it did not go through)."),
                [first_seq] + [c.seq for c in after], after[0].tool))
        else:
            later = ", ".join(dict.fromkeys(c.tool for c in sensitive_after))
            findings.append(Finding(
                "INJECTION_PRESENT", "low",
                f"Instruction-like text ({quoted}) appeared at step {first_seq}; the agent did not act on it"
                + (f" (it called {later}, but only for the customer's own request)." if later
                   else " (no money or contact tool was called afterwards)."),
                [sq for sq, _ in injection_marks]))

    # -- escalation needed ---------------------------------------------------------------
    if escalation_marks:
        s.escalation_text_present = True
        if not s.escalated:
            sq, hits = escalation_marks[0]
            money_after = [c for c in calls if c.succeeded and pol.is_sensitive(c.tool)]
            findings.append(Finding(
                "ESCALATION_MISSED", "critical" if money_after else "high",
                f"The user mentioned {', '.join(repr(h) for h in dict.fromkeys(hits))} which needs a human, but the agent never escalated"
                + (f" and went ahead with {', '.join(dict.fromkeys(c.tool for c in money_after))}." if money_after else "."),
                [sq] + [c.seq for c in money_after]))

    # -- verification failures without escalation ---------------------------------------
    vfails = [c for c in calls if pol.is_verify(c.tool) and not c.succeeded and c.status != "unanswered"]
    if len(vfails) >= max(1, pol.max_failed_verifications):
        esc_after = [q for q in escalation_seqs if q > vfails[pol.max_failed_verifications - 1].seq]
        if not esc_after:
            findings.append(Finding(
                "VERIFY_FAILED_NO_ESCALATION", "high",
                f"Verification failed {len(vfails)} times but the agent did not hand the case to a human.",
                [c.seq for c in vfails]))

    # -- retry loops -------------------------------------------------------------------------
    for tool, n in s.retries.items():
        if n > pol.max_tool_retries:
            ended_bad = not last_by_tool[tool].succeeded
            findings.append(Finding(
                "RETRY_LOOP", "high" if ended_bad else "medium",
                f"{tool} was re-tried {n} times in a row after failing (limit {pol.max_tool_retries})"
                + (" and still failed." if ended_bad else ", then succeeded."),
                sorted(set(retry_runs[tool])), tool))

    # -- unanswered calls / no final answer / error events -------------------------------
    for c in calls:
        if c.status == "unanswered":
            findings.append(Finding("UNANSWERED_CALL", "high" if pol.is_sensitive(c.tool) else "medium",
                                    f"{c.tool} was called but no result was ever logged -- we cannot tell if it happened.",
                                    [c.seq], c.tool))
    if final_evt is None:
        findings.append(Finding("NO_FINAL_ANSWER", "medium",
                                "The run ended without a final answer to the user.",
                                [timeline[-1].seq] if timeline else []))
    if s.n_error_events:
        findings.append(Finding("RUN_ERROR", "medium",
                                f"The run logged {s.n_error_events} error event(s).",
                                [st.seq for st in timeline if st.type == "error"]))

    findings.sort(key=lambda f: (SEVERITY_RANK.get(f.severity, 9), f.evidence_seq[:1] or [0]))
    s.findings = findings
    for f in findings:
        for q in f.evidence_seq:
            stp = step_by_seq.get(q)
            if stp is not None and f.code not in stp.flags:
                stp.flags.append(f.code)
    s.timeline = timeline
    return s


def _injection_driven(c: ToolCall, injections: list[tuple[int, str, str]], violating: set[int]) -> bool:
    """Was this money/contact call plausibly caused by the injected text (not by the user's real request)?

    Yes when the call (a) broke a policy rule (skipped a check, exceeded a cap, forbidden tool), or
    (b) uses an argument value that appears in the injected text but not in what the user wrote
    before it (an attacker's e-mail, an amount). Merely naming the tool's object is NOT enough:
    a customer who asks for a refund in Hinglish and pastes an injection mentioning "refund"
    is still making a legitimate request.
    """
    if c.seq in violating:
        return True
    vals: list[str] = []
    if isinstance(c.args, dict):
        for v in c.args.values():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                vals.append(f"{v:g}")
            elif isinstance(v, str) and len(v.strip()) >= 4:
                vals.append(v.strip().lower())
    for sq, seg, before in injections:
        if sq >= c.seq:
            continue
        seg_l, before_l = seg.lower(), before.lower()
        if any(v in seg_l and v not in before_l for v in vals):
            return True
    return False


def _cap_from_result(spec: str | None, calls: list[ToolCall], before_seq: int) -> float | None:
    """Resolve a dynamic cap like 'lookup_booking.result.fare_paid' from the latest successful
    earlier call of that tool. None when it cannot be resolved."""
    if not spec:
        return None
    tool, _, path = spec.partition(".result")
    path = path.lstrip(".")
    src = next((p for p in reversed(calls) if p.tool == tool and p.seq < before_seq and p.succeeded), None)
    if src is None:
        return None
    cur: Any = src.result
    if isinstance(cur, str):
        try:
            cur = json.loads(cur)
        except ValueError:
            return None
    for part in [x for x in path.split(".") if x]:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return _num(cur)


def _status_phrase(c: ToolCall) -> str:
    return {"error": "failed", "refused": "was refused", "timeout": "timed out",
            "unanswered": "never returned a result", "ok": "returned a negative result"}.get(c.status, c.status)


def summarize_runs(events: Iterable[Any], policy: Any = None) -> list[RunSummary]:
    """Group ARL events by run and summarise each run."""
    pol = load_policy(policy)
    groups = group_events(events)
    vocab = sorted({str(e["tool"]) for evs in groups.values() for e in evs
                    if e.get("tool") and str(e.get("type") or "").strip().lower() in ("tool_call", "tool_result")})
    return [summarize_run(rid, evs, pol, known_tools=vocab) for rid, evs in groups.items()]


# --------------------------------------------------------------------------- whole-dataset view

@dataclass
class AnalysisResult:
    runs: list[RunSummary]
    overview: dict
    action_items: list
    ladder: list
    accuracy: dict | None
    policy_name: str
    model_info: dict

    def runs_table(self):
        import pandas as pd
        return pd.DataFrame([r.to_row() for r in self.runs])

    def findings_table(self):
        import pandas as pd
        rows = [{"run_id": r.run_id, "task_type": r.task_type, "model": r.model, **f.to_dict()}
                for r in self.runs for f in r.findings]
        return pd.DataFrame(rows, columns=["run_id", "task_type", "model", "code", "severity",
                                           "message", "evidence_seq", "tool"])

    def to_dict(self) -> dict:
        return {"overview": self.overview, "policy": self.policy_name, "model_info": self.model_info,
                "accuracy": self.accuracy,
                "action_items": [a.to_dict() if hasattr(a, "to_dict") else a for a in self.action_items],
                "ladder": self.ladder, "runs": [r.to_dict() for r in self.runs]}


def overview(runs: list[RunSummary]) -> dict:
    """Headline numbers for the whole upload."""
    n = len(runs)
    if not n:
        return {"n_runs": 0}
    claimed = sum(r.claimed_success for r in runs)
    false_succ = sum(r.false_success or bool(r.claimed_not_done) for r in runs)
    probs = [r.success_probability for r in runs if r.success_probability is not None]
    with_outcome = [r for r in runs if r.outcome is not None]
    cost = sum(r.cost_usd for r in runs)
    out = {
        "n_runs": n,
        "n_events": sum(r.n_events for r in runs),
        "agents": sorted({r.agent for r in runs if r.agent}),
        "models": sorted({r.model for r in runs if r.model}),
        "task_types": sorted({r.task_type for r in runs if r.task_type}),
        "claimed_success_rate": claimed / n,
        "false_claim_runs": false_succ,
        "false_claim_rate": false_succ / n,
        "false_claim_rate_among_claims": (false_succ / claimed) if claimed else None,
        "escalation_rate": sum(r.escalated for r in runs) / n,
        "runs_with_critical": sum(r.severity_counts["critical"] > 0 for r in runs),
        "runs_with_any_finding": sum(bool(r.findings) for r in runs),
        "mean_success_probability": (sum(probs) / len(probs)) if probs else None,
        "expected_successes": sum(probs) if probs else None,
        "total_cost_usd": cost,
        "cost_estimated": any(r.cost_estimated for r in runs),
        "cost_per_run": cost / n,
        "cost_per_expected_success": (cost / sum(probs)) if probs and sum(probs) >= 1.0 else None,
        "n_with_outcome": len(with_outcome),
        "actual_success_rate": (sum(r.outcome for r in with_outcome) / len(with_outcome)) if with_outcome else None,
        "cost_per_verified_success": None,
        "tool_calls": sum(r.n_tool_calls for r in runs),
        "tool_failures": sum(r.n_error + r.n_refused + r.n_timeout + r.n_unanswered for r in runs),
    }
    if with_outcome:
        wins = sum(r.outcome for r in with_outcome)
        c2 = sum(r.cost_usd for r in with_outcome)
        out["cost_per_verified_success"] = (c2 / wins) if wins else None
        cl = [r for r in with_outcome if r.claimed_success]
        out["claimed_but_failed_rate"] = (sum(1 for r in cl if r.outcome == 0) / len(cl)) if cl else None
    return out


def analyze(events: Iterable[Any], policy: Any = None, model: Any = "default",
            ladder_thresholds: dict | None = None) -> AnalysisResult:
    """Full pipeline: summaries -> success probability -> action items -> ladder.

    ``model``: "default" (use the bundled model file if present, else the heuristic),
    None (heuristic only), a model dict, or a path to a model JSON file.
    """
    from . import actions as _actions
    from . import model as _model

    pol = load_policy(policy)
    runs = summarize_runs(events, pol)
    mdl = _model.resolve_model(model)
    _model.predict_runs(runs, mdl)
    acc = None
    labelled = [r for r in runs if r.outcome is not None]
    if labelled and len({r.outcome for r in labelled}) >= 1:
        acc = _model.evaluate([r.success_probability for r in labelled], [r.outcome for r in labelled])
    items = _actions.build_action_items(runs, pol)
    ladder = _actions.autonomy_ladder(runs, ladder_thresholds)
    info = {"method": "model" if mdl else "heuristic",
            "provenance": (mdl or {}).get("provenance") if mdl else None}
    return AnalysisResult(runs=runs, overview=overview(runs), action_items=items, ladder=ladder,
                          accuracy=acc, policy_name=pol.name, model_info=info)


def read_arl_jsonl(path_or_text: str) -> list[dict]:
    """Tiny reader for canonical ARL JSONL (adapters handle every other format)."""
    import os
    text = path_or_text
    if "\n" not in path_or_text and os.path.exists(path_or_text):
        with open(path_or_text, encoding="utf-8") as fh:
            text = fh.read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]
