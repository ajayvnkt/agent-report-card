"""ARLLogger -- a tiny, dependency-free helper that writes Agent Run Log (ARL) JSONL.

Copy this single file into your agent project (it only uses the standard library):

    from arl_logger import ARLLogger

    log = ARLLogger("runs.jsonl", run_id="r-001", agent="refund-agent", model="gpt-5-mini",
                    task_type="refund")
    log.user_input("Refund my cancelled flight AB123")

    @log.tool
    def issue_refund(booking_ref: str, amount: float) -> dict:
        ...                                # call + result (or exception) are logged automatically

    issue_refund("XYZ9", 420)
    log.final_answer("Your refund has been issued.", claimed_success=True,
                     claimed_actions=["issue_refund"])
    log.end(outcome={"verified_success": 1, "source": "backend check"})

Upload runs.jsonl to Agent Report Card as-is. Spec: docs/LOG_FORMAT.md.
"""
from __future__ import annotations

import functools
import inspect
import itertools
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

__all__ = ["ARLLogger"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [_jsonable(x) for x in v]
        return repr(v)


class ARLLogger:
    """Append ARL events for ONE run to a JSONL file (several runs may share a file).

    Every method writes one line immediately (flushed), so a crash still leaves a usable log.
    Thread-safe. Use as a context manager to guarantee a run_end event:

        with ARLLogger("runs.jsonl", run_id="r-002", agent="a") as log: ...
    """

    def __init__(self, path: str | os.PathLike, run_id: str | None = None, agent: str | None = None,
                 model: str | None = None, task_type: str | None = None, **meta: Any):
        self.path = os.fspath(path)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.agent, self.model, self.task_type = agent, model, task_type
        self._seq = itertools.count()
        self._calls = itertools.count(1)
        self._lock = threading.Lock()
        self._ended = False
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._write("run_start", agent=agent, model=model, task_type=task_type, meta=meta or None)

    # ------------------------------------------------------------------ core
    def _write(self, type_: str, **fields: Any) -> dict:
        with self._lock:
            ev = {"run_id": self.run_id, "seq": next(self._seq), "ts": _now(), "type": type_}
            for k, v in fields.items():
                if v is not None:
                    ev[k] = _jsonable(v)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                fh.flush()
            return ev

    # ------------------------------------------------------------------ events
    def user_input(self, content: str, **meta: Any) -> dict:
        return self._write("user_input", content=content, meta=meta or None)

    def llm_request(self, content: str | None = None, model: str | None = None, **meta: Any) -> dict:
        return self._write("llm_request", content=content, model=model or self.model, meta=meta or None)

    def llm_response(self, content: str | None = None, model: str | None = None, tokens_in: int | None = None,
                     tokens_out: int | None = None, latency_ms: float | None = None,
                     cost_usd: float | None = None, **meta: Any) -> dict:
        return self._write("llm_response", content=content, model=model or self.model, tokens_in=tokens_in,
                           tokens_out=tokens_out, latency_ms=latency_ms, cost_usd=cost_usd, meta=meta or None)

    def tool_call(self, tool: str, args: dict | None = None, call_id: str | None = None, **meta: Any) -> str:
        """Log a tool call; returns its call_id (pass it to tool_result)."""
        cid = call_id or f"c{next(self._calls)}"
        self._write("tool_call", tool=tool, call_id=cid, args=args or {}, meta=meta or None)
        return cid

    def tool_result(self, tool: str, result: Any = None, call_id: str | None = None, status: str | None = None,
                    error: str | None = None, latency_ms: float | None = None, **meta: Any) -> dict:
        """status: ok | error | refused | timeout (inferred by the analyser when omitted)."""
        return self._write("tool_result", tool=tool, call_id=call_id, result=result, status=status,
                           error=error, latency_ms=latency_ms, meta=meta or None)

    def final_answer(self, content: str, claimed_success: bool | None = None,
                     claimed_actions: list[str] | None = None, **meta: Any) -> dict:
        return self._write("final_answer", content=content, claimed_success=claimed_success,
                           claimed_actions=claimed_actions, meta=meta or None)

    def escalation(self, content: str | None = None, **meta: Any) -> dict:
        return self._write("escalation", content=content, meta=meta or None)

    def error(self, error: str | BaseException, **meta: Any) -> dict:
        msg = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        return self._write("error", error=msg, status="error", meta=meta or None)

    def end(self, outcome: dict | None = None, **meta: Any) -> dict | None:
        """Close the run. outcome = {"verified_success": 0|1, "source": "..."} if you have ground truth."""
        if self._ended:
            return None
        self._ended = True
        return self._write("run_end", outcome=outcome, meta=meta or None)

    # ------------------------------------------------------------------ decorator
    def tool(self, fn: Callable | None = None, *, name: str | None = None) -> Callable:
        """Decorator: log tool_call (bound arguments), then tool_result (return value, latency) --
        or a status="error" tool_result with "ExcType: message" when the tool raises (re-raised)."""
        def deco(f: Callable) -> Callable:
            tname = name or f.__name__
            sig = inspect.signature(f)

            def _args(a, kw):
                try:
                    b = sig.bind_partial(*a, **kw)
                    return {k: v for k, v in b.arguments.items() if k not in ("self", "cls")}
                except TypeError:
                    return {"args": list(a), **kw}

            if inspect.iscoroutinefunction(f):
                @functools.wraps(f)
                async def awrapper(*a, **kw):
                    cid = self.tool_call(tname, _args(a, kw))
                    t0 = time.perf_counter()
                    try:
                        out = await f(*a, **kw)
                    except Exception as exc:
                        self.tool_result(tname, call_id=cid, error=f"{type(exc).__name__}: {exc}",
                                         latency_ms=round((time.perf_counter() - t0) * 1000, 3))
                        raise
                    self.tool_result(tname, out, call_id=cid, latency_ms=round((time.perf_counter() - t0) * 1000, 3))
                    return out
                return awrapper

            @functools.wraps(f)
            def wrapper(*a, **kw):
                cid = self.tool_call(tname, _args(a, kw))
                t0 = time.perf_counter()
                try:
                    out = f(*a, **kw)
                except Exception as exc:
                    self.tool_result(tname, call_id=cid, error=f"{type(exc).__name__}: {exc}",
                                     latency_ms=round((time.perf_counter() - t0) * 1000, 3))
                    raise
                self.tool_result(tname, out, call_id=cid, latency_ms=round((time.perf_counter() - t0) * 1000, 3))
                return out
            return wrapper

        return deco(fn) if fn is not None else deco

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> "ARLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.error(exc)
        self.end()
        return False
