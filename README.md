# 🧾 Agent Report Card

> **Upload your agent's logs. Find out which runs really worked, why the others failed, and what to fix.**

Agent Report Card is a reusable tool for **any tool-using AI agent**. You give it your agent's logs in any of six
common formats, and it works with Langfuse exports as they are. It works out the format, converts everything to one canonical **Agent Run Log (ARL)**, and then for
every run it shows you:

- **what happened**: a step-by-step timeline of user input, model turns, tool calls and results, and the final answer;
- **a success probability**: how likely the run really worked. **No ground truth is needed**; the score is read from
  the log alone;
- **findings**: "said done after a tool failed", "claimed an action that never ran", "moved money before
  verification", "obeyed a prompt injection", "retry loop", "tool call with no result", "no final answer", and more;
- **action items**: the problems ranked P0 to P3, each with a concrete fix and the run ids that prove it.

If your logs contain ground-truth outcomes, it also reports **how accurate its own probabilities were**. If they contain
a `task_type`, it gives each task type an **Autonomy Ladder** rung: 🟢 Autonomous, 🟠 Human approves or 🔴 Suggest only.

Built as the Week 1 project ("Data analysis with vibe coding") for *Mastering Agentic AI* at The Gen Academy, with
Claude Code as the AI coding assistant.

### Week 1 submission (Path C: build your own vibe-coded app)

| Deliverable (per the Week 1 handout) | Where it is |
|---|---|
| 1 · Project documentation: overview, datasets, prompts, iterations, learnings, vibe-coding screenshots | [`docs/PROJECT_DOC.md`](docs/PROJECT_DOC.md) (also submitted as a Google Doc) |
| 2 · Video demo (5 minutes or less) | [▶ Demo video, 4:58](https://github.com/ajayvnkt/agent-report-card/releases/download/v1.0/agent_report_card_demo.mp4) (with [subtitles](https://github.com/ajayvnkt/agent-report-card/releases/download/v1.0/agent_report_card_demo.srt)) |
| 3 · Code base | this repository |

![Step 1 upload, Step 2 policy, Step 3 run](docs/images/01_steps.png)

| Summary | Action items |
|---|---|
| ![Summary tab](docs/images/03_summary.png) | ![Action items tab](docs/images/04_action_items.png) |
| **Run detail** | **Policy format guide** |
| ![Run detail tab](docs/images/05_run_detail.png) | ![What should a policy look like?](docs/images/02_policy_format.png) |

---

## Install

Python 3.10+.

```bash
git clone https://github.com/ajayvnkt/agent-report-card.git
cd agent-report-card
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # streamlit, pandas, plotly, pyyaml, numpy
make run                                 # or: streamlit run app.py  → http://localhost:9003
```

`make test` runs the test suite (needs `pip install -r requirements-dev.txt`). `make help` lists every command.

## Quickstart (60 seconds)

1. **Step 1 · Upload your agent logs.** Click **Upload** and choose `sample_logs/demo_agent_runs.log`.
   This is an ordinary Python log file: 16 runs of an airline support agent, each a different story (clean
   successes, correct hand-offs to a human, and every failure the tool looks for), mixed in with 3 unrelated server lines.
   The app answers **"✅ Detected: Python logging text (agent_run key=value lines) · 16 runs · 336 events · 3 lines skipped"**.
   (Or skip the file dialog: open **"…or try a sample"** and pick
   **"SkyDesk demo · 16 runs · Python log (demo_agent_runs.log) — start here"**.)
2. **Step 2 · Policy (optional).** Choose **"SkyDesk policy"**. Picking a sample selects its policy for you.
3. **Step 3 · Run the analysis.** Click **"▶ Run analysis"**. The progress box ends with
   **"Analysis complete — 16 runs, 11 action items"**.
4. The Summary shows predicted success **31%** (actual 31%), **7** runs that claimed success but likely failed,
   **3** critical policy violations and **$0.0123** per likely-successful run. Start with run **`sky-0904-0028`**:
   `update_contact` failed three times, and the agent still told the customer the phone number was updated.
5. Read the tabs: **Summary**, **Action items**, **Runs**, **Run detail**, **Autonomy ladder**, **Model accuracy**.

Not sure what your logs should look like? Open **"📄 What format should my logs be in?"** under Step 1. It has a
minimal example, every ARL field, one example line per accepted format, and download buttons for an example file in
each format and for `arl_logger.py`.

Not sure what your policy should look like? Open **"📐 What should a policy look like?"** under Step 2. It shows the
SkyDesk policy line by line, a table of every key, a 5-line minimal template, and download buttons for
`skydesk_policy.yaml`, `shopdesk_policy.yaml` and `policy_template.yaml`.

### Sample inputs

| Folder | What is in it |
|---|---|
| [`examples/`](examples/) | **Start here.** The same 3 runs in every format (`example_arl.jsonl`, `example_openai_messages.jsonl`, `example_python.log`, `example_otel_spans.json`, `example_otlp.json`, `example_events.csv`, `example_langfuse.json`), the policies `skydesk_policy.yaml`, `shopdesk_policy.yaml`, `policy_template.yaml`, and `arl_logger.py`. Compare your logs with the matching file. |
| [`sample_logs/`](sample_logs/) | **The demo set:** `demo_agent_runs.log` (upload this) and `demo_agent_runs.jsonl` (same 16 runs in ARL), one run per scenario with ground truth, plus `skydesk_policy.yaml` and `shopdesk_policy.yaml`. The run-by-run guide is in [`sample_logs/README.md`](sample_logs/README.md). |
| `build/samples/` (not committed) | `make samples` generates 300 SkyDesk runs in every format, an unlabeled copy and the 600 ShopDesk training runs here (deterministic, about a second). |

The 16 demo runs:

| Run | Scenario | Ground truth | Predicted | What the report shows |
|---|---|---|---|---|
| `sky-0901-0003` | No final answer (bereavement case escalated, customer never told) | failure | 0% | No final answer |
| `sky-0901-0004` | Clean success (voucher) | success | 97% | — |
| `sky-0902-0009` | Prompt injection obeyed (attacker e-mail set as contact) | failure | 0% | Obeyed a prompt injection |
| `sky-0902-0012` | Money moved before verification | failure | 0% | Skipped a required check |
| `sky-0902-0014` | Tool timed out, agent said so honestly | failure | 0% | — (scored low) |
| `sky-0902-0015` | Said "done" after a failed tool (rebooking) | failure | 0% | Said "done" after a failed tool, claimed an action that never happened |
| `sky-0903-0019` | Unnecessary escalation | failure | 2% | — (scored low) |
| `sky-0903-0021` | Claimed an action that never ran | failure | 0% | Claimed an action that never happened |
| `sky-0903-0023` | Clean success (rebooking) | success | 97% | — |
| `sky-0903-0024` | **Human in the loop:** bereavement case escalated to a human | success | 98% | — |
| `sky-0904-0028` | Retry loop, then "your phone has been updated" (`update_contact` failed 3x) | failure | 0% | Retry loop, said "done" after a failed tool, claimed an action that never happened |
| `sky-0904-0031` | Log ends on an unanswered `issue_refund` call | failure | 0% | Unanswered call, no final answer |
| `sky-0905-0055` | Refund over the cap | failure | 0% | Went over a limit, said "done" after a failed tool, claimed an action that never happened |
| `sky-0905-0301` | **Human in the loop:** fare refunded, hotel claim above the cap sent to a supervisor for approval | success | 96% | — |
| `sky-0906-0059` | Prompt injection resisted | success | 99% | Prompt injection resisted (keep testing) |
| `sky-0906-0063` | Legal threat that should have gone to a human; agent issued a voucher itself | failure | 3% | Should have gone to a human |

Run the tests (336 tests, about 25 seconds): `make test`.

`.streamlit/config.toml` pins port 9003 and headless mode. The app follows the viewer's light or dark preference.

## Supported log formats (auto-detected)

| # | Format | Typical source | What one record looks like | Example file |
|---|---|---|---|---|
| 1 | **ARL JSONL** (canonical) | your own logger (`arc/arl_logger.py`) | one event per line: `{"run_id","seq","type", ...}` | `sample_logs/demo_agent_runs.jsonl`, `examples/example_arl.jsonl` |
| 2 | **OpenAI chat messages** | OpenAI SDK, most agent frameworks | one run per line: `{"run_id","agent","model","task_type","messages":[...],"outcome":{...}}`. A JSON list of runs, or a bare `messages` list (one run), also works. | `examples/example_openai_messages.jsonl` |
| 3 | **Python `logging` text** | app logs | `2026-09-01T04:46:09Z INFO agent_run run_id=r-1 seq=4 type=tool_call tool=issue_refund args={"amount":420}` | `sample_logs/demo_agent_runs.log`, `examples/example_python.log` |
| 4 | **OpenTelemetry** GenAI spans | tracing backends (OTLP JSON, or a list/JSONL of flat spans) | trace = run; `invoke_agent <agent>` root span; `chat` spans; `execute_tool <tool>` spans with `gen_ai.tool.*` attributes | `examples/example_otel_spans.json`, `examples/example_otlp.json` |
| 5 | **CSV** with ARL column names | spreadsheets | one row per event; `args` / `result` as JSON strings | `examples/example_events.csv` |
| 6 | **Langfuse export** | Langfuse UI or public API (JSON list, `{"data":[...]}` page, or JSONL; observations alone also work) | trace = run; `input` / `output`; `GENERATION` and `TOOL` observations; a `verified_success` score = outcome | `examples/example_langfuse.json` |

How unusual input is handled:

- Unrelated lines in a Python log (web-server access lines, scheduler noise) are **skipped and counted**. Traceback
  lines after an `agent_run` line are attached to that event.
- A tool result with no matching call still counts. Results whose call id does not match are re-paired by tool name,
  with a warning saying how many.
- Empty, binary or unrelated files give a clear **"We couldn't read this file."** message, never a crash.
- If you upload several files, they are merged. Clashing run ids get the file name in front, and the app says so.

The full specification, with a field table and a complete example run, is in [`docs/LOG_FORMAT.md`](docs/LOG_FORMAT.md).
`python -m arc formats` prints the list above.

### The canonical format in one example

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

The refund **failed**, but the agent **said it was issued**. Catching this is the tool's main job.
The minimum you need is `run_id`, `seq` and `type`, plus `tool`, `args`, `result` and `status` for tool events.
Add `task_type` for the Autonomy Ladder, tokens or `cost_usd` for cost, and `outcome` for accuracy reporting.

## Log your own agent with `arc/arl_logger.py`

`arc/arl_logger.py` is a single file that uses only the standard library. Copy it into your agent project:

```python
from arl_logger import ARLLogger

log = ARLLogger("runs.jsonl", run_id="r-001", agent="refund-agent", model="gpt-5-mini",
                task_type="refund")
log.user_input("Refund my cancelled flight AB123")

@log.tool                       # each call and its result (or exception) is logged automatically
def issue_refund(booking_ref: str, amount: float) -> dict:
    ...

issue_refund("XYZ9", 420)
log.final_answer("Your refund has been issued.", claimed_success=True,
                 claimed_actions=["issue_refund"])
log.end(outcome={"verified_success": 1, "source": "backend check"})   # outcome is optional
```

Every event is written and flushed straight away, so a crash still leaves a usable log. It is thread-safe, and
`with ARLLogger(...) as log:` guarantees a `run_end` event. Upload `runs.jsonl` as it is. The app also offers
this file under **"⬇ arl_logger.py"** in the format box.

## Command line

```bash
# analyse a file; prints a Markdown report (or write .json / .md with --out)
.venv/bin/python -m arc analyze sample_logs/demo_agent_runs.log --policy sample_logs/skydesk_policy.yaml --out report.md

# force a format, or use the rule-of-thumb score instead of the trained model
.venv/bin/python -m arc analyze my_logs.txt --format pylog --no-model

# convert any supported format to another (for example to canonical ARL)
.venv/bin/python -m arc convert sample_logs/demo_agent_runs.log --to arl --out demo.jsonl

# list the accepted formats
.venv/bin/python -m arc formats

# train your own success model on labelled runs
.venv/bin/python -m arc train my_labelled_logs.jsonl --policy my_policy.yaml --out arc/models/success_model.json
```

`--format` takes `arl|openai|pylog|otel|csv|langfuse`. `--to` takes `arl|openai|pylog|otel|otlp|csv|langfuse`. A bad policy
file prints a clean error and exits with code 2.

## Policies: the rules every run is checked against

A policy is a short YAML file: which tools move money or change contact details, which tools must succeed before
others (`require_before`, e.g. `issue_refund: [verify_customer]`), forbidden tools, a retry limit, numeric caps on
arguments, prompt-injection patterns, and text that must be escalated to a human (bereavement, fraud, legal,
expense claims that need a supervisor's approval).
Every key is optional; a key you leave out keeps the generic default.

A minimal policy:

```yaml
name: my-support-agent
money_tools: [issue_refund]
require_before: {issue_refund: [verify_customer]}
forbidden_tools: [delete_account]
caps: {issue_refund: {arg: amount, max: 500}}
```

- [`docs/POLICY_FORMAT.md`](docs/POLICY_FORMAT.md): every key with its type, meaning and default, a fully annotated
  example, how `re:` selectors, `@groups` and regex patterns work, every validation message, and how each finding
  maps to a severity.
- `arc/policies/default.yaml`: the generic default (money and contact changes need verification first; at most
  2 retries).
- `sample_logs/skydesk_policy.yaml`: the SkyDesk airline agent's rules (used by the demo).
- `sample_logs/shopdesk_policy.yaml`: the ShopDesk online-store agent's rules (used for training).

In the app, choose **"Default generic policy"**, **"SkyDesk policy"**, **"Upload YAML"** or **"Edit YAML"**.
Misspelt keys (such as `require_befor`) show a warning instead of being silently ignored, and an invalid pattern
or cap is caught at Step 2 with **"This policy could not be read: …"**, not mid-analysis.

## How the success probability works

For each run the analyzer turns the log into 24 signals: unrecovered tool failures, retries, unanswered calls,
whether there is a final answer, whether the agent claimed success, claimed actions that never ran,
critical / high / medium / low findings, injection text present, escalation, run length, and so on. A small model
turns those signals into a probability.

- **Model:** L2-regularised logistic regression with Platt calibration (`arc/models/success_model.json`).
  **Sign constraints** keep explanations honest: failure signals may only lower the score. A feature that would get
  the wrong sign is dropped (`n_refused`, `fail_rate` and `n_retries` were dropped), so a run with many failed calls
  is never told "this raises the chance of success".
- **Every score is explained.** Run detail lists the factors that pushed it up or down under **"Why this score"**.
  The strongest signal by far is the number of critical findings.
- **Training and selection, on our own synthetic data only:** the model is trained on **600 ShopDesk runs**, an
  online-store support agent (generated on demand into `build/samples/training/`). Ten candidates (two feature sets × five L2 strengths) were
  compared with order-grouped 5-fold cross-validation on ShopDesk alone. The winner was then scored **once** on the
  **300 SkyDesk runs**, an airline agent with different tools, policy and models that it never saw.
- **Fallback:** `--no-model` (or `model=None` in code) uses a transparent rule-of-thumb score instead.

### Model card summary (real metrics)

AUC: 0.5 is chance, 1.0 is perfect ranking. Brier: lower is better.

| Evaluation | n | AUC | Brier | Baseline Brier | Accuracy at 0.5 |
|---|---|---|---|---|---|
| ShopDesk, order-grouped 5-fold CV (training agent) | 600 | **0.985** | 0.018 | 0.248 | 98.0% |
| SkyDesk, all runs (never seen in training) | 300 | **0.991** | 0.008 | 0.246 | 99.3% |
| SkyDesk-v1 / SkyDesk-v2 | 150 / 150 | 0.987 / 0.998 | 0.009 / 0.007 | — | 99.3% / 99.3% |
| Rule-of-thumb fallback on SkyDesk, all runs | 300 | 0.968 | 0.104 | 0.246 | — |
| SkyDesk demo set (in the video) | 16 | 1.000 | 0.000 | 0.215 | 100%, but only 16 runs |

On all 300 SkyDesk runs the model makes 2 mistakes (two failures it called successes); mean predicted success is
55.6% against an actual 56.3%.

**Read this honestly:** both agents come from the same generator, with the same family of planted failures and the
same log conventions, so the SkyDesk test is much easier than a real new agent would be. These numbers show the
model learned this generator's failure signals; they are **not** an estimate of real-world accuracy. **Expect real
agents to score clearly lower**, and expect over-confident probabilities (it often says 1% or 98%). The full card,
with every candidate, the selection protocol and the limitations, is in
[`arc/models/MODEL_CARD.md`](arc/models/MODEL_CARD.md). To use it seriously, retrain on your own labelled runs
(`python -m arc train`).

## What the action items look like

From the 16-run demo file with the SkyDesk policy (real output of `python -m arc analyze`):

> **1. [P0] Agent reports actions that never happened**
> *Why:* 4 of 16 runs (25%) listed actions as done that never succeeded (issue_refund, issue_voucher,
> rebook_flight, send_itinerary, update_contact).
> *Fix:* Build the user-facing summary from the audit log, not from the model's memory: list only tool calls whose
> result was ok, and fail the run if claimed_actions contains anything else.
> *Impact:* 4 of 4 affected runs with a verified outcome actually failed. They average 0% predicted success,
> against 41% for the rest.
> *Evidence runs:* sky-0902-0015, sky-0903-0021, sky-0904-0028, sky-0905-0055

> **2. [P0] Agent says "done" after a tool call failed**: 3 of 16 runs. *Fix:* re-read every tool result before
> the final answer and block `claimed_success` if a money or contact call failed; say what failed and offer a hand-off.

> **5. [P0] Sensitive cases not handed to a human**: a customer threatened legal action and the agent issued a
> voucher itself instead of escalating (`sky-0906-0063`). *Fix:* check the message against the escalation list
> before any tool runs.

> **6. [P1] Retry loop on update_contact**: the same call failed with "UpstreamError: 503 Service Unavailable" and
> was retried more than the policy allows. *Fix:* cap retries at 1, then escalate.

The two human-in-the-loop runs (`sky-0903-0024`, `sky-0905-0301`) get no finding and score 96-98%: handing a
case to a person when the policy says so counts as success. On the 300-run SkyDesk set (`make samples`) the top items are: actions reported that never happened (47 runs), "done" after a tool
failed (28), refunds above the cap (13) and obeyed prompt injections (11). In every one of those groups, all affected
runs really failed.

## Data: all of it ours, all of it synthetic

Every log in this project was generated by `scripts/generate_sample_logs.py` (standard library only, deterministic).
There are no real customers, bookings, orders, cards or e-mail addresses.

| Dataset | Runs | Role |
|---|---|---|
| **ShopDesk**, a made-up online store's support agent (`shopdesk-v1` on gpt-4o-mini, `shopdesk-v2` on gemini-2.5-flash), seed 11 | 600 (275 distinct orders) | **Training and model selection.** Generated on demand into `build/samples/training/` (`make samples` or `make train`) |
| **SkyDesk**, a made-up airline's support agent (`skydesk-v1` on gpt-4.1-mini, `skydesk-v2` on gpt-5-mini), seed 7 | 300, generated on demand into `build/samples/` | **Testing.** Every upload format, plus an unlabeled copy for the no-ground-truth case; the test suite generates it into a temp folder |
| **SkyDesk demo set**, committed in `sample_logs/` | 16 (15 picked from the 300, one per scenario, plus one human-approval run) | **Demos.** `demo_agent_runs.log` and `demo_agent_runs.jsonl` |

Failure modes (false "done", money before verification, refunds over the cap, obeyed injections, retry loops,
missing replies, missed escalations…) were planted on purpose at different rates in the two agents. Ground truth
comes from a simulated backend plus an audit of the reply, not from the planted label, and includes noise the log
never shows (about 4% of refunds reversed, some e-mails bouncing). See [`sample_logs/README.md`](sample_logs/README.md).

## Project structure

```
agent-report-card/
├── app.py                     # Streamlit app: Step 1 upload → Step 2 policy → Step 3 run → 6 result tabs
├── arc/                       # the reusable engine (no Streamlit inside)
│   ├── adapters/              # arl, openai_messages, pylog, otel, csv_arl, langfuse → ARL events
│   ├── detect.py              # format auto-detection
│   ├── schema.py              # ARL validation, call pairing
│   ├── analyze.py             # per-run findings and features
│   ├── claims.py              # reads "done" / claimed actions from reply text when not logged
│   ├── model.py               # success model (train + predict + explanations)
│   ├── actions.py             # aggregated, prioritised action items
│   ├── policy.py              # policy loading and validation
│   ├── pipeline.py            # file → report (JSON / Markdown)
│   ├── arl_logger.py          # drop-in logger for your own agent
│   ├── models/                # success_model.json, MODEL_CARD.md, training_report.json
│   └── policies/              # default.yaml
├── examples/                  # small sample inputs: 3 runs in every format + policy files + arl_logger.py
├── sample_logs/               # the 16-run demo set (demo_agent_runs.log / .jsonl) + skydesk_policy.yaml + shopdesk_policy.yaml
├── build/samples/             # (git-ignored) bulk sets from `make samples`: 300 SkyDesk runs in every format, ShopDesk training runs
├── scripts/                   # generate_sample_logs.py, train_success_model.py, validate_sample_logs.py, make_examples.py
├── tests/                     # 336 tests (adapters, analyzer, pipeline, app, robustness, training); conftest.py generates the bulk sets
├── docs/                      # LOG_FORMAT.md, POLICY_FORMAT.md, PROJECT_DOC.md, DEMO_SCRIPT.md, images/
├── video/                     # narration.json, make_video_v2.py (narrated demo), screenshots.py
├── requirements.txt / requirements-dev.txt / Makefile
└── ci/github-actions-tests.yml  # CI config: copy to .github/workflows/ to run the tests on every push
```

## Limitations

- **The score reads the log, not the backend.** A refund that returns `ok` and is reversed later looks like a
  success. Treat the probability as "how healthy does this run look", not as an audit.
- **It depends on your policy.** A wrong policy gives wrong findings and probabilities, because critical findings
  are the model's strongest signal. Write one for your agent with [`docs/POLICY_FORMAT.md`](docs/POLICY_FORMAT.md).
- **Synthetic, narrow training data.** 600 ShopDesk runs from one generator, two models. Calibration was fitted on
  a 54% base rate, so on very different agents the probabilities shift even when the ranking holds. Retrain on your
  own labelled runs before trusting the numbers.
- **English-only text rules.** When a log does not record `claimed_success` or `claimed_actions` (as in every
  OpenAI-messages log), they are read from the reply with English keywords and a small synonym table
  (contact/email/phone, rebook/booked, refund). Other domains may need more entries.
- **Some failures get no high-severity finding:** time-outs, runs with no final answer and unnecessary escalations.
  They are still mostly scored below 0.5. A "handed to a human for no reason" run (e.g. `sky-0903-0019`) gets no
  finding at all; only the score (2%) flags it, because the model learned that hand-offs on routine requests
  usually fail. A hand-off the policy asks for (a sensitive topic or an expense claim) is scored as success.
- **Mixed-format files** are read in their main format only; lines in the other format are skipped with an
  explanation. An unterminated quoted value in a Python log line keeps its stray quote, with a warning.
- **Uploader extensions:** Streamlit only accepts `.jsonl`, `.json`, `.log`, `.txt` and `.csv`. Rename other
  extensions (such as `.ndjson`) before uploading, or use the CLI.
- **Costs are estimates** when the log has no `cost_usd` (tokens × a price table, or characters ÷ 4 when there are no token counts).
  "Est. cost per likely-successful run" shows "–" when fewer than one success is expected.
- **Dark mode has not been checked by eye.** The app uses Streamlit's own theming, so it should work.

## How it was built

Vibe coded with Claude Code: an idea, short prompts, steering when it went the wrong way, and multi-agent build and
test rounds. The prompts, iterations and bugs fixed are written up in [`docs/PROJECT_DOC.md`](docs/PROJECT_DOC.md#5-how-it-was-vibe-coded-with-claude-code).

| The idea | Steering it |
|---|---|
| ![Idea prompt](docs/images/process_1_idea.png) | ![Redirect to Streamlit](docs/images/process_3_redirect_to_streamlit.png) |

## License

MIT, see [`LICENSE`](LICENSE). All sample data is synthetic and generated by this repository.
