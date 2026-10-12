# Agent Run Log (ARL) v1 — the canonical format

Every input format is converted to this before analysis. Store your agent's requests and
responses this way and the tool needs no adapter at all.

**File:** JSON Lines (`.jsonl`), one **event** per line, events of a run in order.

## Event fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `run_id` | string | yes | One agent task/session. All events of a run share it. |
| `seq` | int | yes | Order within the run, starting at 0. |
| `ts` | string (ISO-8601) | no | Event time. |
| `type` | enum | yes | `run_start`, `user_input`, `llm_request`, `llm_response`, `tool_call`, `tool_result`, `final_answer`, `escalation`, `error`, `run_end` |
| `agent` | string | no | Agent name, e.g. `refund-agent`. |
| `model` | string | no | LLM used for this step, e.g. `gpt-5-mini`. |
| `task_type` | string | no | Category of the task (enables per-task Autonomy Ladder). Usually on `run_start`. |
| `content` | string | no | Text: user message, model text, final answer summary. |
| `tool` | string | for tool events | Tool name. |
| `call_id` | string | no | Links a `tool_result` to its `tool_call`. If absent, results pair with the latest unanswered call of the same tool. |
| `args` | object | no | Tool arguments. |
| `result` | any | no | Tool return value. |
| `status` | enum | no | For `tool_result`/`error`: `ok`, `error`, `refused`, `timeout`. Inferred if missing. |
| `error` | string | no | Error / refusal message. |
| `claimed_actions` | list[string] | no | On `final_answer`: actions the agent says it did. |
| `claimed_success` | bool | no | On `final_answer`: does the agent present the task as done? A hand-off ("I've escalated your case") is **not** done → `false`. Inferred from text if missing (rules in `arc/claims.py`). |
| `tokens_in`, `tokens_out` | int | no | Usage for `llm_response`. |
| `latency_ms` | number | no | Step latency. |
| `cost_usd` | number | no | Step cost; estimated from tokens & price table if missing. |
| `outcome` | object | no | On `run_end` only: `{"verified_success": 0/1, "source": "..."}` — ground truth if you have it. Enables accuracy reporting; never used to compute a run's probability. |
| `meta` | object | no | Anything else; preserved, ignored by analysis. |

## Minimal example (one run)

```jsonl
{"run_id":"r-001","seq":0,"type":"run_start","agent":"refund-agent","model":"gpt-5-mini","task_type":"refund"}
{"run_id":"r-001","seq":1,"type":"user_input","content":"Refund my cancelled flight AB123, booking ref XYZ9."}
{"run_id":"r-001","seq":2,"type":"tool_call","tool":"verify_customer","call_id":"c1","args":{"booking_ref":"XYZ9"}}
{"run_id":"r-001","seq":3,"type":"tool_result","tool":"verify_customer","call_id":"c1","status":"ok","result":true}
{"run_id":"r-001","seq":4,"type":"tool_call","tool":"issue_refund","call_id":"c2","args":{"booking_ref":"XYZ9","amount":420}}
{"run_id":"r-001","seq":5,"type":"tool_result","tool":"issue_refund","call_id":"c2","status":"error","error":"PolicyError: amount exceeds fare paid"}
{"run_id":"r-001","seq":6,"type":"final_answer","content":"Your refund of $420 has been issued.","claimed_success":true,"claimed_actions":["issue_refund"]}
{"run_id":"r-001","seq":7,"type":"run_end","outcome":{"verified_success":0,"source":"backend check"}}
```

This run is the classic failure the tool exists to catch: the refund **failed** but the agent **claimed success**.

## Other formats accepted (auto-detected)

| Format | Typical source | How it maps |
|---|---|---|
| ARL JSONL (this spec) | your own logger (`arl_logger.py`) | native |
| OpenAI chat messages (`messages` with `tool_calls` + `role:"tool"`) JSON / JSONL | OpenAI / most agent frameworks | assistant `tool_calls` → `tool_call`; `tool` messages → `tool_result`; last assistant text → `final_answer` |
| OpenTelemetry GenAI spans (OTLP JSON, or a list / JSONL of flat spans) | tracing backends | `gen_ai.*` / `tool.*` attributes → events, grouped by trace_id |
| Python `logging` text lines | app logs | `key=value` pairs per line: `run=… event=tool_call tool=… args={…}` (see examples) |
| CSV with ARL column names | spreadsheets | one row per event |
| Langfuse export (traces + observations; JSON list, `{"data":[…]}` or JSONL) | Langfuse UI / public API | trace = run; `GENERATION` → model turns; `TOOL` observations → `tool_call` + `tool_result`; `trace.output` → `final_answer`; a `verified_success` score → outcome (details below) |

Example files: [`examples/`](../examples/) has the same 3 runs in every format (`example_arl.jsonl`,
`example_openai_messages.jsonl`, `example_python.log`, `example_otel_spans.json`, `example_otlp.json`,
`example_events.csv`, `example_langfuse.json`). The demo set `sample_logs/demo_agent_runs.log` (Python logging text,
with 3 unrelated server lines mixed in) and `sample_logs/demo_agent_runs.jsonl` (ARL) hold the same 16 runs, one per
scenario. `make samples` writes 300 runs in every format to `build/samples/`.

## Langfuse exports

Upload what Langfuse gives you, as it is: the traces from the UI export or from `GET /api/public/traces`
(a JSON list, a `{"data": [...], "meta": {...}}` page, or JSONL with one trace per line). Observations exported
on their own (each with a `traceId`) are grouped into their trace; a file may hold both.

| Langfuse | ARL |
|---|---|
| trace `id` | `run_id` (one trace = one run) |
| trace `name` (or `metadata.agent`) | `run_start.agent` |
| `metadata.task_type`, or a tag `task_type:<x>` (or the only tag) | `run_start.task_type` |
| `sessionId`, `userId` | `run_start.meta.session_id` / `user_id` |
| trace `input` (string, or the last user message) | `user_input` |
| `GENERATION` | `llm_request` at `startTime` + `llm_response` at `endTime` with `model`, `usage`/`usageDetails` `input`/`output` → `tokens_in`/`tokens_out`, `calculatedTotalCost` / `costDetails.total` → `cost_usd`. `tool_calls` in its output → `tool_call` (when no `TOOL` observation runs that call; the result is then read from a later `role:"tool"` message). |
| `TOOL` observation, or a `SPAN` that is a tool (`metadata.tool`, a `tool:<name>` / `execute_tool <name>` name, or the name of a tool the model asked for) | `tool_call` (`input` → `args`, `metadata.tool_call_id` or the observation id → `call_id`) + `tool_result` (`output` → `result`). `level: "ERROR"` (or `WARNING` with a `statusMessage`) → `error` = `statusMessage`; the status (`error` / `refused` / `timeout`) is then inferred from that text like every other format. A tool with no `endTime` and no `output` was never answered. |
| successful call of an escalation tool (`escalate_*`, `handoff`, `transfer_to_human`) | also an `escalation` event, unless the trace logs `escalation` events itself |
| `EVENT` named like an ARL type (`escalation`, `error`, `user_input`, …) | that event; `output` (else `input`) → `content`, `statusMessage` → `error` |
| trace `output` (else the last plain-text generation after the last tool) | `final_answer`; `metadata.claimed_success` / `claimed_actions` when present, otherwise claimed success is inferred from the text (`arc/claims.py`) |
| score named `verified_success` (or `success`, `task_success`, `resolved`, …) with value 0/1, or `metadata.outcome` | `run_end.outcome.verified_success` |
| other observations (`AGENT`, `CHAIN`, plain `SPAN`s) | containers, ignored (counted in the notes) |

Events are ordered by `startTime`. Minimal example (one run: the refund was refused, the agent said it was done):

```json
[{"id": "r-001", "name": "refund-agent", "timestamp": "2026-09-01T10:00:00Z",
  "tags": ["task_type:refund"],
  "input": "Refund my cancelled flight AB123, booking XYZ9",
  "output": "Your refund of $420 has been issued.",
  "scores": [{"name": "verified_success", "value": 0}],
  "observations": [
    {"id": "o1", "traceId": "r-001", "type": "TOOL", "name": "verify_customer",
     "startTime": "2026-09-01T10:00:01Z", "endTime": "2026-09-01T10:00:02Z",
     "input": {"booking_ref": "XYZ9"}, "output": true},
    {"id": "o2", "traceId": "r-001", "type": "TOOL", "name": "issue_refund",
     "startTime": "2026-09-01T10:00:03Z", "endTime": "2026-09-01T10:00:04Z",
     "input": {"booking_ref": "XYZ9", "amount": 420}, "level": "ERROR",
     "statusMessage": "PolicyError: amount exceeds fare paid"}]}]
```

`python -m arc convert <file> --to langfuse` writes this shape. It also stores what plain Langfuse fields cannot hold
(call ids, explicit statuses, event order, `meta`) under `metadata.arc`, so ARL → Langfuse → ARL is lossless. Your
own exports do not need `metadata.arc`.

## Policy file

The rules every run is checked against live in a separate YAML file. See
[POLICY_FORMAT.md](POLICY_FORMAT.md) for every key, an annotated example, a 5-line minimal template,
how patterns work and what each validation message means.
