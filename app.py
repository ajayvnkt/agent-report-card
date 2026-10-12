"""Agent Report Card v2 -- Streamlit UI.

A guided demo on top of ``arc.pipeline``: upload logs -> (optional) policy -> Run analysis -> results.
All analysis happens in the ``arc`` package; this file only loads inputs and draws the results.

Run:  .venv/bin/streamlit run app.py        (port 9003, set in .streamlit/config.toml)
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from arc.actions import DEFAULT_THRESHOLDS, autonomy_ladder
from arc.analyze import FINDING_CODES, SEVERITY_RANK
from arc.detect import UnrecognizedFormatError, load_any
from arc.export import export
from arc.pipeline import analyze_events
from arc.policy import load_policy, policy_warnings

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "sample_logs"
EXAMPLES_DIR = ROOT / "examples"
DEMO_LOG = SAMPLES_DIR / "demo_agent_runs.log"
DEMO_ARL = SAMPLES_DIR / "demo_agent_runs.jsonl"
SKYDESK_POLICY = SAMPLES_DIR / "skydesk_policy.yaml"
LOG_FORMAT_MD = ROOT / "docs" / "LOG_FORMAT.md"
MODEL_CARD_MD = ROOT / "arc" / "models" / "MODEL_CARD.md"
POLICY_FORMAT_MD = ROOT / "docs" / "POLICY_FORMAT.md"
SHOPDESK_POLICY = SAMPLES_DIR / "shopdesk_policy.yaml"
ARL_LOGGER_PY = ROOT / "arc" / "arl_logger.py"

# --------------------------------------------------------------------------- labels (also used by tests / demo recording)

APP_TITLE = "Agent Report Card"
PROMISE = ("Upload your agent's logs. Find out which runs really worked, why the others failed, "
           "and what to fix.")

STEP1 = "Step 1 · Upload your agent logs"
STEP2 = "Step 2 · Policy (optional)"
STEP3 = "Step 3 · Run the analysis"
UPLOAD_LABEL = "Drag your log file(s) here, or click **Upload**"
SAMPLE_LABEL = "…or try a sample"
SAMPLE_NONE = "— pick a sample log —"
FORMAT_HELP_LABEL = "📄 What format should my logs be in?"
POLICY_HELP_LABEL = "📐 What should a policy look like?"
RUN_BUTTON = "▶ Run analysis"

POLICY_LABEL = "Which rules should every run be checked against?"
POL_DEFAULT = "Default generic policy"
POL_SKYDESK = "SkyDesk policy"
POL_UPLOAD = "Upload YAML"
POL_EDIT = "Edit YAML"
POLICY_OPTIONS = [POL_DEFAULT, POL_SKYDESK, POL_UPLOAD, POL_EDIT]

TAB_SUMMARY = "Summary"
TAB_ACTIONS = "Action items"
TAB_RUNS = "Runs"
TAB_DETAIL = "Run detail"
TAB_LADDER = "Autonomy ladder"
TAB_ACCURACY = "Model accuracy"
TABS = [TAB_SUMMARY, TAB_ACTIONS, TAB_RUNS, TAB_DETAIL, TAB_LADDER, TAB_ACCURACY]

# friendly sample name -> (path, suggested policy)
SAMPLES: dict[str, tuple[Path, str]] = {
    "SkyDesk demo · 16 runs · Python log (demo_agent_runs.log) — start here": (DEMO_LOG, POL_SKYDESK),
    "SkyDesk demo · 16 runs · ARL JSONL (demo_agent_runs.jsonl)": (DEMO_ARL, POL_SKYDESK),
    "Example · 3 runs · ARL JSONL": (EXAMPLES_DIR / "example_arl.jsonl", POL_SKYDESK),
    "Example · 3 runs · Python log": (EXAMPLES_DIR / "example_python.log", POL_SKYDESK),
    "Example · 3 runs · OpenAI chat messages":(EXAMPLES_DIR / "example_openai_messages.jsonl", POL_SKYDESK),
    "Example · 3 runs · OpenTelemetry spans": (EXAMPLES_DIR / "example_otel_spans.json", POL_SKYDESK),
    "Example · 3 runs · OpenTelemetry OTLP JSON": (EXAMPLES_DIR / "example_otlp.json", POL_SKYDESK),
    "Example · 3 runs · CSV": (EXAMPLES_DIR / "example_events.csv", POL_SKYDESK),
    "Example · 3 runs · Langfuse export": (EXAMPLES_DIR / "example_langfuse.json", POL_SKYDESK),
}

# verdict bands on P(success)
LIKELY_SUCCESS, UNCERTAIN, LIKELY_FAILURE = "Likely success", "Uncertain", "Likely failure"
BAND_HI, BAND_LO = 0.7, 0.4
VERDICT_COLORS = {LIKELY_SUCCESS: "#2e9e5b", UNCERTAIN: "#e0a100", LIKELY_FAILURE: "#d64545"}
VERDICT_BADGE = {LIKELY_SUCCESS: "green", UNCERTAIN: "orange", LIKELY_FAILURE: "red"}
SEVERITY_BADGE = {"critical": "red", "high": "orange", "medium": "blue", "low": "gray"}
PRIORITY_BADGE = {"P0": "red", "P1": "orange", "P2": "blue", "P3": "gray"}
RUNG_BADGE = {"Autonomous": "green", "Human approves": "orange", "Suggest only": "red"}
RUNG_ICON = {"Autonomous": "🟢", "Human approves": "🟠", "Suggest only": "🔴"}

FINDING_SHORT = {
    "FALSE_SUCCESS_AFTER_FAILURE": "Said “done” after a failed tool",
    "CLAIMED_NOT_DONE": "Claimed an action that never happened",
    "PREREQ_MISSING": "Skipped a required check",
    "FORBIDDEN_TOOL": "Called a forbidden tool",
    "CAP_EXCEEDED": "Went over a limit",
    "INJECTION_OBEYED": "Obeyed a prompt injection",
    "INJECTION_PRESENT": "Prompt injection resisted",
    "ESCALATION_MISSED": "Should have gone to a human",
    "VERIFY_FAILED_NO_ESCALATION": "Failed verification, no hand-off",
    "RETRY_LOOP": "Retry loop",
    "UNANSWERED_CALL": "Tool call never answered",
    "NO_FINAL_ANSWER": "No final answer",
    "RUN_ERROR": "Run logged an error",
    "TOOL_ERROR_PATTERN": "Same tool error keeps repeating",
}
POLICY_CODES = {"PREREQ_MISSING", "FORBIDDEN_TOOL", "CAP_EXCEEDED", "INJECTION_OBEYED", "ESCALATION_MISSED",
                "VERIFY_FAILED_NO_ESCALATION"}

st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  .arc-promise {font-size: 1.15rem; opacity: .8; margin-top: -.6rem;}
  .arc-step {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin-bottom: -.4rem;}
  .arc-quote {border-left: 4px solid rgba(91,141,239,.7); padding: .4rem .9rem; background: rgba(91,141,239,.08);
              border-radius: 0 .4rem .4rem 0; margin: .3rem 0 .6rem 0;}
  .arc-num {font-size: 1.6rem; font-weight: 700; opacity: .35; line-height: 1;}
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {white-space: normal !important;
      overflow: visible !important; text-overflow: clip !important;}
  .stTabs [data-baseweb="tab"] p {font-size: 1.05rem; font-weight: 600;}
  .st-key-format_help details, .st-key-policy_help details {border: 1px solid rgba(91,141,239,.55) !important; border-radius: .5rem;}
  .st-key-format_help summary, .st-key-policy_help summary {background: rgba(91,141,239,.10); font-weight: 600;}
  .st-key-format_help summary p, .st-key-policy_help summary p {font-size: 1rem; font-weight: 600;}
  .st-key-run_analysis button {font-size: 1.15rem; padding: .7rem 2.2rem; min-height: 3.2rem;}
  .st-key-run_analysis button p {font-size: 1.15rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- small helpers

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#$|~])")


def esc(text) -> str:
    """Escape text so Streamlit markdown shows it literally (no LaTeX on $, no links, no emphasis)."""
    return _MD_SPECIAL.sub(r"\\\1", str(text if text is not None else ""))


def code(text) -> str:
    """Inline code span (markdown escapes are not processed inside backticks)."""
    return "`" + str(text if text is not None else "").replace("`", "'") + "`"


def pct(x, digits: int = 0) -> str:
    return "–" if x is None else f"{x:.{digits}%}"


def usd(x) -> str:
    if x is None:
        return "–"
    if 0 < x < 0.0001:
        return "<$0.0001"
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def verdict(p) -> str:
    if p is None:
        return UNCERTAIN
    return LIKELY_SUCCESS if p >= BAND_HI else (LIKELY_FAILURE if p < BAND_LO else UNCERTAIN)


def short(text, n: int = 110) -> str:
    s = str(text if text is not None else "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def top_finding(r) -> str:
    if not r.findings:
        return ""
    f = min(r.findings, key=lambda f: SEVERITY_RANK.get(f.severity, 9))
    return FINDING_SHORT.get(f.code, f.code)


def most_striking_run(rep) -> str | None:
    """Claimed success but lowest probability; otherwise just the lowest probability."""
    if not rep.runs:
        return None
    key = lambda r: (r.success_probability if r.success_probability is not None else 1.0)  # noqa: E731
    claimed = [r for r in rep.runs if r.claimed_success]
    return min(claimed or rep.runs, key=key).run_id


def goto_run(run_id: str) -> None:
    st.session_state["detail_run"] = run_id
    st.session_state["results_tab"] = TAB_DETAIL


def goto_tab(tab: str) -> None:
    st.session_state["results_tab"] = tab


# --------------------------------------------------------------------------- loading (arc.detect via cache)

@st.cache_data(show_spinner=False, max_entries=24)
def _load_bytes(data: bytes, name: str) -> tuple[list[dict], dict]:
    events, src = load_any(data, filename=name)
    return [e.to_dict() for e in events], src


@st.cache_data(show_spinner=False, max_entries=24)
def _load_path(path: str) -> tuple[list[dict], dict]:
    events, src = load_any(path)
    return [e.to_dict() for e in events], src


def merge_sources(parts: list[tuple[str, list[dict], dict]]) -> tuple[list[dict], dict]:
    """Several uploaded files -> one event list. Clashing run_ids get the file name as a prefix."""
    if len(parts) == 1:
        name, events, src = parts[0]
        return events, {**src, "files": [name]}
    seen: set[str] = set()
    all_events: list[dict] = []
    warnings: list[str] = []
    labels: list[str] = []
    for name, events, src in parts:
        ids = {str(e.get("run_id")) for e in events}
        clash = ids & seen
        if clash:
            stem = Path(name).stem
            warnings.append(f"{name}: {len(clash)} run_id(s) also appear in an earlier file, so this file's run "
                            f"ids were prefixed with '{stem}:' to keep the runs apart.")
            events = [{**e, "run_id": f"{stem}:{e.get('run_id')}"} for e in events]
            ids = {str(e.get("run_id")) for e in events}
        seen |= ids
        all_events.extend(events)
        warnings.extend(f"{name}: {w}" for w in src.get("warnings") or [])
        if src.get("format_label") not in labels:
            labels.append(src.get("format_label"))
    merged = {
        "format": parts[0][2]["format"] if len(labels) == 1 else "mixed",
        "format_label": " + ".join(labels),
        "confidence": min(p[2].get("confidence", 1.0) for p in parts),
        "n_runs": len(seen),
        "n_events": len(all_events),
        "skipped_lines": sum(p[2].get("skipped_lines", 0) for p in parts),
        "skipped_examples": [x for p in parts for x in p[2].get("skipped_examples", [])][:5],
        "warnings": warnings,
        "files": [p[0] for p in parts],
    }
    return all_events, merged


def on_sample_change() -> None:
    choice = st.session_state.get("sample")
    if choice in SAMPLES:
        st.session_state["policy_choice"] = SAMPLES[choice][1]


# --------------------------------------------------------------------------- format help content

def _md_section(md: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", md, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


@st.cache_data(show_spinner=False)
def example_files() -> dict[str, tuple[str, str, str]]:
    """label -> (file name, text, mime): the first 3 demo runs written in every format."""
    events, _ = load_any(DEMO_ARL)
    keep = list(dict.fromkeys(e.run_id for e in events))[:3]
    evs = [e for e in events if e.run_id in keep]
    spec = [("ARL JSONL", "arl", "example_arl.jsonl", "application/jsonl"),
            ("OpenAI messages", "openai", "example_openai_messages.jsonl", "application/jsonl"),
            ("Python log", "pylog", "example_python.log", "text/plain"),
            ("OTel spans", "otel", "example_otel_spans.json", "application/json"),
            ("OTLP JSON", "otlp", "example_otlp.json", "application/json"),
            ("CSV", "csv", "example_events.csv", "text/csv"),
            ("Langfuse export", "langfuse", "example_langfuse.json", "application/json")]
    return {label: (fname, export(evs, fmt), mime) for label, fmt, fname, mime in spec}


FORMATS_TABLE = """
| # | Format | Typical source | What one line / record looks like |
|---|---|---|---|
| 1 | **ARL JSONL** (recommended) | `arl_logger.py`, your own logger | `{"run_id":"r-1","seq":2,"type":"tool_call","tool":"verify_customer","args":{"booking_ref":"XYZ9"}}` |
| 2 | **OpenAI chat messages** (JSON / JSONL) | OpenAI SDK, most agent frameworks | `{"run_id":"r-1","messages":[{"role":"user",…},{"role":"assistant","tool_calls":[…]},{"role":"tool",…}]}` |
| 3 | **Python logging text** (.log / .txt) | app logs | `2026-09-01T10:00:02Z INFO agent_run run_id=r-1 seq=2 type=tool_call tool=verify_customer args={"booking_ref":"XYZ9"}` |
| 4 | **OpenTelemetry spans** (OTLP JSON or flat spans) | tracing backends | span `execute_tool verify_customer` with `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, grouped by `trace_id` |
| 5 | **CSV** with ARL column names | spreadsheets | `run_id,seq,type,tool,args,result,status` → `r-1,2,tool_call,verify_customer,"{""booking_ref"":""XYZ9""}",,` |
| 6 | **Langfuse export** (traces + observations, JSON / JSONL) | Langfuse UI or public API | trace `{"id":"r-1","input":…,"output":…,"observations":[{"type":"TOOL","name":"verify_customer","input":{…},"output":{…}}]}` |
"""

LOGGER_SNIPPET = '''from arl_logger import ARLLogger           # copy arc/arl_logger.py into your project

log = ARLLogger("runs.jsonl", run_id="r-001", agent="refund-agent",
                model="gpt-5-mini", task_type="refund")
log.user_input("Refund my cancelled flight AB123, booking XYZ9")

@log.tool                                    # every call + result (or exception) is logged
def issue_refund(booking_ref: str, amount: float) -> dict: ...

issue_refund("XYZ9", 420)
log.final_answer("Your refund has been issued.", claimed_success=True, claimed_actions=["issue_refund"])
log.end(outcome={"verified_success": 1})    # optional ground truth -> enables the accuracy tab'''


def render_format_help() -> None:
    md = LOG_FORMAT_MD.read_text(encoding="utf-8") if LOG_FORMAT_MD.exists() else ""
    st.markdown(
        "**Short answer:** one JSON object per line, one line per *event* (a user message, a tool call, a tool "
        "result, the final answer…). Events of the same run share a `run_id` and are numbered by `seq`. "
        "That format is called **ARL** (Agent Run Log).\n\n"
        "**Already have logs in another format?** Upload them as they are — the app recognises 6 formats "
        "automatically, including Langfuse exports (table below).")
    t1, t2, t3, t4 = st.tabs(["Minimal example", "All ARL fields", "Accepted formats", "Example files & logger"])
    with t1:
        st.markdown("A complete run in ARL — the refund **failed**, but the agent **said it was done**. "
                    "This is exactly the kind of run the tool exists to catch:")
        m = re.search(r"```jsonl\n(.*?)```", md, flags=re.S)
        st.code(m.group(1).strip() if m else "", language="json")
        st.markdown("**You need at minimum:** `run_id`, `seq`, `type`, and for tools `tool` + `args` / "
                    "`result` + `status`.  \n**Add when you have them:** `task_type` → Autonomy ladder, "
                    "`tokens_*`/`cost_usd` → cost per success, `run_end.outcome.verified_success` → model "
                    "accuracy.")
    with t2:
        st.markdown(_md_section(md, "Event fields") or "See docs/LOG_FORMAT.md")
    with t3:
        st.markdown(FORMATS_TABLE)
        st.caption("Unrelated lines (web-server access logs, tracebacks…) are skipped and counted. "
                   "Full spec: docs/LOG_FORMAT.md.")
    with t4:
        st.markdown("**Download a small example** (3 runs of the SkyDesk demo) in the format you use, "
                    "compare it with your logs, then upload yours:")
        cols = st.columns(3)
        for i, (label, (fname, text, mime)) in enumerate(example_files().items()):
            cols[i % 3].download_button(f"⬇ {label}", text, file_name=fname, mime=mime, key=f"ex_{fname}",
                                        on_click="ignore", width="stretch")
        st.divider()
        st.markdown("**Logging from your own Python agent?** Drop this one stdlib-only file into your project:")
        st.download_button("⬇ arl_logger.py", ARL_LOGGER_PY.read_text(encoding="utf-8"),
                           file_name="arl_logger.py", mime="text/x-python", key="dl_logger", on_click="ignore")
        st.code(LOGGER_SNIPPET, language="python")


# --------------------------------------------------------------------------- policy format help

def _yaml_under(md: str, heading: str) -> str:
    """The first ```yaml block in the '## <heading>' section of docs/POLICY_FORMAT.md."""
    m = re.search(r"```yaml\n(.*?)```", _md_section(md, heading), flags=re.S)
    return m.group(1) if m else ""


POLICY_DOC = POLICY_FORMAT_MD.read_text(encoding="utf-8") if POLICY_FORMAT_MD.exists() else ""
POLICY_MINIMAL_YAML = _yaml_under(POLICY_DOC, "Minimal example")
POLICY_ANNOTATED_YAML = _yaml_under(POLICY_DOC, "Annotated example (SkyDesk)")

POLICY_FIELDS_TABLE = """
| Key | Type | What it means |
|---|---|---|
| `name` | text | Name shown in the report |
| `money_tools` | list of tool names | Tools that move money (refunds, vouchers, payments) |
| `contact_tools` | list of tool names | Tools that change phone / email / address / payee |
| `escalation_tools` | list of tool names | Calling one = handed to a human |
| `*_tool_patterns` | list of regexes | Same groups by name pattern (money, contact, verify, escalation, readonly) |
| `require_before` | tool → [checks] | The tool may run only after every listed check **succeeded** earlier in the run |
| `forbidden_tools` | list of tools | Must never be called |
| `max_tool_retries` | whole number | Back-to-back retries of a failing tool before it is a retry loop (default 2) |
| `max_failed_verifications` | whole number | Failed verifications after which the run must escalate (default 2) |
| `injection_patterns` | list of regexes | Instruction-like text in user / tool messages (prompt injection) |
| `escalate_if_text_matches` | list of regexes | Topics that must go to a human (bereavement, legal, fraud…) |
| `caps` | tool → {arg, max, min, max_from} | Numeric limits on tool arguments; `max_from: lookup_booking.result.fare_paid` reads the limit from an earlier result |
"""


def render_policy_help() -> None:
    st.markdown(
        "**Short answer:** a small YAML file with the rules your agent must follow. **Every key is optional** — "
        "anything you leave out keeps the generic default, and a key you set replaces it. "
        "Name tools exactly (`issue_refund`), by pattern (`re:^delete_`), or by group "
        "(`\"@money\"`, `\"@contact\"`, `\"@verify\"`, `\"@escalation\"`).")
    t1, t2, t3, t4 = st.tabs(["Annotated example (SkyDesk)", "All keys", "Minimal template", "Download examples"])
    with t1:
        st.markdown("The policy the SkyDesk demo uses, line by line:")
        st.code(POLICY_ANNOTATED_YAML, language="yaml")
    with t2:
        st.markdown(POLICY_FIELDS_TABLE)
        st.caption("Regex lists are case-insensitive Python regexes; wrap them in single quotes in YAML. "
                   "Policy findings are critical when the risky call went through (e.g. a refund before "
                   "verification), otherwise high.")
    with t3:
        st.markdown("Five lines are enough to start — paste it into **Edit YAML** and change the tool names:")
        st.code(POLICY_MINIMAL_YAML, language="yaml")
    with t4:
        st.markdown("Download an example, edit it for your agent, then choose **Upload YAML** above:")
        c = st.columns(3)
        c[0].download_button("⬇ skydesk_policy.yaml", SKYDESK_POLICY.read_text(encoding="utf-8"),
                             file_name="skydesk_policy.yaml", mime="text/yaml", key="dl_pol_skydesk",
                             on_click="ignore", width="stretch")
        c[1].download_button("⬇ shopdesk_policy.yaml", SHOPDESK_POLICY.read_text(encoding="utf-8"),
                             file_name="shopdesk_policy.yaml", mime="text/yaml", key="dl_pol_shopdesk",
                             on_click="ignore", width="stretch")
        c[2].download_button("⬇ policy_template.yaml", POLICY_MINIMAL_YAML, file_name="policy_template.yaml",
                             mime="text/yaml", key="dl_pol_template", on_click="ignore", width="stretch")
    st.caption("Full reference — every key, defaults, regex rules, validation messages and severities: "
               "docs/POLICY_FORMAT.md.")


# --------------------------------------------------------------------------- policy in plain English

def _sel(s: str) -> str:
    groups = {"@money": "any money tool", "@contact": "any contact-changing tool",
              "@verify": "a verification tool", "@escalation": "an escalation tool"}
    if s in groups:
        return groups[s]
    if s.startswith("re:"):
        return f"any tool matching `{s[3:]}`"
    return f"`{s}`"


def policy_rules(pol) -> list[str]:
    rules: list[str] = []
    money = ", ".join(f"`{t}`" for t in pol.money_tools)
    rules.append("💰 **Money tools:** " + (money or "none listed by name")
                 + (f", plus any tool whose name looks like money ({len(pol.money_tool_patterns)} name patterns, e.g. "
                    f"refund / payment / voucher)" if pol.money_tool_patterns else "") + ".")
    if pol.contact_tools or pol.contact_tool_patterns:
        rules.append("📇 **Contact-changing tools:** "
                     + (", ".join(f"`{t}`" for t in pol.contact_tools) or "any tool named like update_phone / "
                        "update_email / add_payee") + ".")
    for tool, pre in (pol.require_before or {}).items():
        rules.append(f"🔐 {_sel(tool).capitalize()} may only run **after** "
                     + " and ".join(_sel(p) for p in pre) + " has **succeeded** in the same run.")
    for tool, caps in (pol.caps or {}).items():
        for c in caps:
            lim = []
            if c.get("min") is not None:
                lim.append(f"at least {c['min']}")
            if c.get("max") is not None:
                lim.append(f"at most {c['max']}")
            if c.get("max_from"):
                lim.append(f"never more than `{c['max_from']}`")
            rules.append(f"📏 `{tool}`: argument `{c['arg']}` must be {', '.join(lim)}.")
    if pol.forbidden_tools:
        rules.append("⛔ **Never call:** " + ", ".join(f"`{t}`" for t in pol.forbidden_tools) + ".")
    rules.append(f"🔁 A failing tool may be retried at most **{pol.max_tool_retries}** time(s) in a row; "
                 "more is a retry loop.")
    rules.append(f"🙋 After **{pol.max_failed_verifications}** failed verification(s) the agent must hand over "
                 "to a human.")
    rules.append(f"🧨 User / tool text is data, not instructions: **{len(pol.injection_patterns)}** "
                 "prompt-injection patterns are checked (e.g. “ignore previous instructions”, “admin override”).")
    rules.append(f"🚩 **{len(pol.escalate_if_text_matches)}** sensitive-topic patterns (bereavement, legal, "
                 "fraud, expense claims…) mean the case must be escalated to a human.")
    return rules


def resolve_policy() -> tuple[object, str | None, str | None]:
    """-> (policy source for arc, signature text, error message)."""
    choice = st.session_state.get("policy_choice", POL_DEFAULT)
    if choice == POL_DEFAULT:
        return None, "default", None
    if choice == POL_SKYDESK:
        return str(SKYDESK_POLICY), "skydesk", None
    if choice == POL_UPLOAD:
        f = st.session_state.get("policy_file")
        if f is None:
            return None, None, "Upload a .yaml policy file above (or pick another option)."
        text = f.getvalue().decode("utf-8", errors="replace")
    else:
        text = st.session_state.get("policy_yaml", "")
    try:
        load_policy(text)
    except Exception as e:  # noqa: BLE001 -- shown to the user
        return None, None, f"This policy could not be read: {e}"
    return text, "text:" + hashlib.md5(text.encode()).hexdigest(), None


# =========================================================================== PAGE

st.title(f"🧾 {APP_TITLE}")
st.markdown(f"<div class='arc-promise'>{PROMISE}</div>", unsafe_allow_html=True)
st.caption("Works for any tool-using AI agent · no ground truth needed · your file is analysed locally "
           "and never sent anywhere.")
st.write("")

# --------------------------------------------------------------------------- STEP 1
with st.container(border=True):
    st.subheader(STEP1)
    st.markdown("Click **Upload** and choose your agent's log file (you can select several — they are merged). "
                "No logs at hand? Upload `sample_logs/demo_agent_runs.log` or pick a sample on the right.")
    c_up, c_sample = st.columns([3, 2], gap="large")
    with c_up:
        uploads = st.file_uploader(UPLOAD_LABEL, type=["jsonl", "json", "log", "txt", "csv"],
                                   accept_multiple_files=True, key="uploads")
    with c_sample:
        st.selectbox(SAMPLE_LABEL, [SAMPLE_NONE] + list(SAMPLES), key="sample", on_change=on_sample_change,
                     help="The SkyDesk demo set (16 synthetic airline-support runs, one per scenario) "
                          "and 3-run examples in the other supported formats.")
        if uploads and st.session_state.get("sample", SAMPLE_NONE) != SAMPLE_NONE:
            st.caption("Your uploaded file(s) are used. Remove them to analyse the sample instead.")

    with st.expander(FORMAT_HELP_LABEL, expanded=False, key="format_help"):
        render_format_help()

    events: list[dict] | None = None
    source: dict | None = None
    data_sig: str | None = None
    load_error: str | None = None
    try:
        if uploads:
            parts = []
            for f in uploads:
                data = f.getvalue()
                try:
                    evs, src = _load_bytes(data, f.name)
                except (UnrecognizedFormatError, ValueError, UnicodeDecodeError) as e:
                    raise UnrecognizedFormatError(f"{f.name}: {e}") from e
                parts.append((f.name, evs, src))
            events, source = merge_sources(parts)
            data_sig = "up:" + "|".join(f"{f.name}:{f.size}:{hashlib.md5(f.getvalue()).hexdigest()}" for f in uploads)
        elif st.session_state.get("sample", SAMPLE_NONE) in SAMPLES:
            name = st.session_state["sample"]
            path = SAMPLES[name][0]
            evs, src = _load_path(str(path))
            events, source = evs, {**src, "files": [path.name]}
            data_sig = "sample:" + name
    except UnrecognizedFormatError as e:
        load_error = str(e)
    except Exception as e:  # noqa: BLE001 -- never crash on a bad upload
        load_error = f"Unexpected problem while reading the file: {type(e).__name__}: {e}"

    if load_error:
        st.error("**We couldn't read this file.** Check that it is one of the accepted formats — open "
                 f"“{FORMAT_HELP_LABEL[2:]}” above for examples you can download and compare with.  \n"
                 "Common causes: the file is empty, it is a pretty-printed JSON object that is not a list of "
                 "events/runs, or each line is missing `run_id` / `type`.", icon="🚫")
        with st.expander("Details"):
            st.code(load_error, language=None)
    elif source:
        skipped = source.get("skipped_lines", 0)
        st.success(f"✅ Detected: **{source['format_label']}** · **{source['n_runs']}** runs · "
                   f"**{source['n_events']}** events · **{skipped}** lines skipped", icon=None)
        st.caption(f"File: {', '.join(source.get('files', []))} · detection confidence "
                   f"{source.get('confidence', 1.0):.0%} · everything was converted to the canonical Agent Run Log.")
        if source.get("warnings"):
            with st.expander(f"⚠️ {len(source['warnings'])} note(s) while reading the file"):
                for w in source["warnings"][:20]:
                    st.markdown(f"- {esc(short(w, 300))}")
        first = str(events[0].get("run_id")) if events else None
        with st.expander(f"👀 Preview: the first run (`{first}`) as normalised events"):
            prev = [e for e in events if str(e.get("run_id")) == first][:25]
            st.dataframe(pd.DataFrame([{
                "seq": e.get("seq"), "type": e.get("type"), "tool": e.get("tool") or "",
                "status": e.get("status") or "",
                "details": short(e.get("content") or e.get("error")
                                 or (json.dumps(e.get("args")) if e.get("args") is not None else "")
                                 or (json.dumps(e.get("result"), default=str) if e.get("result") is not None else "")
                                 or (json.dumps(e.get("outcome")) if e.get("outcome") else ""), 140),
            } for e in prev]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- STEP 2
with st.container(border=True):
    st.subheader(STEP2)
    st.markdown("The rules each run is checked against — e.g. *“no refund before the customer is verified”*. "
                "Not sure? Keep the default; samples pick their own policy automatically.")
    st.session_state.setdefault("policy_choice", POL_DEFAULT)
    if "policy_yaml" not in st.session_state:
        st.session_state["policy_yaml"] = SKYDESK_POLICY.read_text(encoding="utf-8")
    c_pol, c_rules = st.columns([2, 3], gap="large")
    with c_pol:
        st.radio(POLICY_LABEL, POLICY_OPTIONS, key="policy_choice",
                 captions=["Works for most customer-service / back-office agents",
                           "Airline demo: verify before refunds, refund ≤ fare paid",
                           "Your own rules as a YAML file",
                           "Write or tweak the rules right here"])
        if st.session_state["policy_choice"] == POL_UPLOAD:
            st.file_uploader("Upload a policy (.yaml / .yml / .json)", type=["yaml", "yml", "json"], key="policy_file")
        if st.session_state["policy_choice"] == POL_EDIT:
            st.text_area("Policy YAML (prefilled with the SkyDesk example)", key="policy_yaml", height=320)
    policy_src, policy_sig, policy_error = resolve_policy()
    with c_rules:
        if policy_error:
            st.warning(policy_error)
        else:
            pol = load_policy(policy_src)
            for w in policy_warnings(pol):
                st.warning(w, icon="⚠️")
            st.markdown(f"**In plain English — “{esc(pol.name)}” checks that:**")
            st.markdown("\n".join(f"- {r}" for r in policy_rules(pol)))
    with st.expander(POLICY_HELP_LABEL, expanded=False, key="policy_help"):
        render_policy_help()

# --------------------------------------------------------------------------- STEP 3
with st.container(border=True):
    st.subheader(STEP3)
    ready = events is not None and not load_error and not policy_error
    sig = f"{data_sig}||{policy_sig}" if ready else None
    if not ready:
        st.markdown("Finish **Step 1** first (upload a file or pick a sample), then click the button.")
    else:
        st.markdown(f"Ready: **{source['n_runs']}** runs. Click **{RUN_BUTTON}** — it takes a few seconds.")
    clicked = st.button(RUN_BUTTON, type="primary", disabled=not ready, key="run_analysis", width="content")
    if clicked and ready:
        with st.status("Analysing your logs…", expanded=True) as status:
            st.write(f"① Normalising to the Agent Run Log… {source['n_events']} events in {source['n_runs']} runs ✓")
            st.write("② Checking each run against the policy…")
            try:
                rep = analyze_events(events, source=source, policy=policy_src)
            except Exception as e:  # noqa: BLE001 -- show a message, never a stack trace
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error("**The analysis stopped on this file.** Try the default policy, or check the file "
                         "against the examples in the format help above.", icon="🚫")
                st.code(f"{type(e).__name__}: {e}", language=None)
                st.stop()
            n_find = sum(len(r.findings) for r in rep.runs)
            st.write(f"   → {n_find} findings in {rep.overview.get('runs_with_any_finding', 0)} runs ✓")
            method = "trained model" if rep.model_info.get("method") == "model" else "rule-of-thumb heuristic"
            st.write(f"③ Scoring success probability ({method})… mean "
                     f"{pct(rep.overview.get('mean_success_probability'))} ✓")
            st.write(f"④ Writing action items… {len(rep.action_items)} items ✓")
            status.update(label=f"Analysis complete — {len(rep.runs)} runs, {len(rep.action_items)} action items",
                          state="complete", expanded=False)
        st.session_state["report"] = rep
        st.session_state["report_sig"] = sig
        st.session_state["report_json"] = rep.to_json()
        st.session_state["report_md"] = rep.to_markdown()
        st.session_state["detail_run"] = most_striking_run(rep)
        st.session_state["results_tab"] = TAB_SUMMARY
        for k in ("f_verdict", "f_task", "f_model", "f_finding"):
            st.session_state.pop(k, None)

rep = st.session_state.get("report")
if rep is None:
    st.info("👆 Your report card will appear here after you click **Run analysis**.")
    st.stop()

if st.session_state.get("report_sig") != (f"{data_sig}||{policy_sig}" if events is not None else None):
    st.warning("The file or policy changed since this report was made. Click **Run analysis** again to refresh it.",
               icon="🔄")


# =========================================================================== RESULTS

runs = rep.runs
has_outcomes = rep.accuracy is not None
ov, cost = rep.overview, rep.cost


def runs_df() -> pd.DataFrame:
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.run_id, "agent": r.agent or "(unspecified)", "model": r.model or "(unspecified)",
            "task_type": r.task_type or "(unspecified)", "p": r.success_probability,
            "verdict": verdict(r.success_probability), "claimed": bool(r.claimed_success),
            "top_finding": top_finding(r), "codes": r.finding_codes,
            "n_findings": len(r.findings),
            "critical": any(f.severity == "critical" for f in r.findings),
            "critical_policy": any(f.severity == "critical" and f.code in POLICY_CODES for f in r.findings),
            "outcome": r.outcome, "cost": r.cost_usd,
        })
    return pd.DataFrame(rows)


DF = runs_df()
claimed_likely_failed = DF[(DF.claimed) & (DF.p < BAND_LO)]

st.divider()
st.header("Your report card")
st.caption(f"{rep.source.get('format_label')} · {len(runs)} runs · policy “{rep.policy['name']}” · success score: "
           + ("trained model (" + str(rep.model_info.get("source")) + ")" if rep.model_info.get("method") == "model"
              else "rule-of-thumb heuristic"))
with st.container(horizontal=True):
    st.download_button("⬇ Download full report (JSON)", st.session_state.get("report_json") or rep.to_json(),
                       file_name="agent_report_card.json", mime="application/json", key="dl_json",
                       on_click="ignore")
    st.download_button("⬇ Download full report (Markdown)", st.session_state.get("report_md") or rep.to_markdown(),
                       file_name="agent_report_card.md", mime="text/markdown", key="dl_md", on_click="ignore")

st.session_state.setdefault("results_tab", TAB_SUMMARY)
tabs = st.tabs(TABS, key="results_tab", on_change="rerun")
T = dict(zip(TABS, tabs))


# --------------------------------------------------------------------------- Summary
with T[TAB_SUMMARY]:
    k = st.columns(5)
    k[0].metric("Runs analysed", f"{len(runs)}", border=True,
                help=f"{ov.get('n_events', 0)} events · {len(ov.get('agents', []))} agent version(s) · "
                     f"{len(ov.get('task_types', []))} task type(s)")
    k[1].metric("Predicted success", pct(ov.get("mean_success_probability")), border=True,
                delta=(f"actual {pct(ov.get('actual_success_rate'))}" if has_outcomes else None), delta_color="off", delta_arrow="off",
                help="Average success probability over all runs, computed from the logs only (no ground truth). "
                     + ("‘actual’ = verified outcomes logged in the file." if has_outcomes else ""))
    n_clf = len(claimed_likely_failed)
    k[2].metric("Claimed success but likely failed", f"{n_clf}", border=True,
                delta=(f"{int((claimed_likely_failed.outcome == 0).sum())} really failed" if has_outcomes and n_clf else None),
                delta_color="off", delta_arrow="off",
                help=f"The agent told the user “done”, but the success probability is below {BAND_LO:.0%}.")
    k[3].metric("Critical policy violations", f"{int(DF.critical_policy.sum())}", border=True,
                help="Runs with a critical policy finding: skipped a required check, went over a limit, called a "
                     "forbidden tool, obeyed a prompt injection, or failed to escalate.")
    per_exp = cost.get("per_expected_success_usd")
    k[4].metric("Est. cost per likely-successful run", usd(per_exp), border=True,
                delta=(f"{usd(cost.get('per_verified_success_usd'))} per verified success"
                       if cost.get("per_verified_success_usd") is not None else None), delta_color="off", delta_arrow="off",
                help="Total cost ÷ expected number of successful runs (sum of probabilities). "
                     + ("Not shown: fewer than one run is expected to succeed, so the ratio would be meaningless. "
                        if per_exp is None else "")
                     + ("Estimated from token counts. " if cost.get("estimated") else "") + str(cost.get("note", "")))

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("##### How likely is each run to have really worked?")
        hist = DF.dropna(subset=["p"])
        fig = px.histogram(hist, x="p", color="verdict", nbins=20, range_x=[0, 1],
                           color_discrete_map=VERDICT_COLORS,
                           category_orders={"verdict": [LIKELY_FAILURE, UNCERTAIN, LIKELY_SUCCESS]},
                           labels={"p": "Success probability", "verdict": ""})
        fig.update_traces(xbins=dict(start=0, end=1.0001, size=0.05))
        fig.update_layout(height=320, bargap=0.05, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Runs",
                          xaxis_tickformat=".0%", legend=dict(orientation="h", y=1.08, x=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key="hist")
        vc = DF.verdict.value_counts()
        st.caption(f"🟢 {LIKELY_SUCCESS} (≥{BAND_HI:.0%}): {vc.get(LIKELY_SUCCESS, 0)} · 🟡 {UNCERTAIN}: "
                   f"{vc.get(UNCERTAIN, 0)} · 🔴 {LIKELY_FAILURE} (<{BAND_LO:.0%}): {vc.get(LIKELY_FAILURE, 0)}")
    with c2:
        st.markdown("##### Fix these first")
        for i, a in enumerate(rep.action_items[:3], 1):
            with st.container(border=True):
                st.markdown(f"**{i}.** :{PRIORITY_BADGE.get(a.priority, 'gray')}-badge[{a.priority}] "
                            f"**{esc(a.title)}**  \n{esc(a.why)}")
        if not rep.action_items:
            st.success("No findings — nothing to fix in these runs. 🎉")
        st.button("See all action items →", key="goto_actions", on_click=goto_tab, args=(TAB_ACTIONS,))

    def breakdown(by: list[str]) -> pd.DataFrame:
        d = DF.assign(clf=DF.claimed & (DF.p < BAND_LO))
        aggs = dict(runs=("run_id", "count"), predicted=("p", "mean"))
        if has_outcomes:
            aggs["actual"] = ("outcome", "mean")
        aggs.update(claimed_done=("claimed", "mean"), clf=("clf", "sum"), critical=("critical", "sum"),
                    cost_per_run=("cost", "mean"))
        out = d.groupby(by, dropna=False).agg(**aggs).reset_index()
        for c in ("predicted", "actual", "claimed_done"):
            if c in out:
                out[c] = out[c] * 100
        return out

    colcfg = {
        "predicted": st.column_config.ProgressColumn("Predicted success", min_value=0, max_value=100, format="%.0f%%"),
        "actual": st.column_config.ProgressColumn("Actual success", min_value=0, max_value=100, format="%.0f%%"),
        "claimed_done": st.column_config.NumberColumn("Said “done”", format="%.0f%%"),
        "critical": st.column_config.NumberColumn("Runs w/ critical finding"),
        "clf": st.column_config.NumberColumn("Said done, likely failed"),
        "cost_per_run": st.column_config.NumberColumn("Cost / run", format="$%.4f"),
    }
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("##### By agent version & model")
        st.dataframe(breakdown(["agent", "model"]), hide_index=True, width="stretch", column_config=colcfg)
    with b2:
        st.markdown("##### By task type")
        st.dataframe(breakdown(["task_type"]).sort_values("predicted"), hide_index=True, width="stretch",
                     column_config=colcfg)
    if rep.notes:
        with st.expander("ℹ️ Notes on this analysis"):
            for n in rep.notes:
                st.markdown(f"- {esc(n)}")


# --------------------------------------------------------------------------- Action items
def action_items_md() -> str:
    L = ["# Agent Report Card — action items", "",
         f"{len(runs)} runs · policy {rep.policy['name']} · {rep.source.get('format_label')}", ""]
    for i, a in enumerate(rep.action_items, 1):
        L += [f"## {i}. [{a.priority}] {a.title}", "", f"- **Why:** {a.why}", f"- **Fix:** {a.fix}",
              f"- **Impact:** {a.expected_impact}",
              f"- **Task types:** {', '.join(f'{t} ({n})' for t, n in a.task_types.items())}",
              f"- **Evidence runs:** {', '.join(a.evidence_runs)}", ""]
    return "\n".join(L)


with T[TAB_ACTIONS]:
    a1, a2 = st.columns([4, 1], vertical_alignment="bottom")
    a1.markdown(f"**{len(rep.action_items)} things to fix, most important first.** Each one shows why it matters, "
                "how to fix it, and the runs that prove it — click a run id to open it.")
    a2.download_button("⬇ Download action items (Markdown)", action_items_md(), file_name="action_items.md",
                       mime="text/markdown", key="dl_actions", on_click="ignore", width="stretch")
    if not rep.action_items:
        st.success("No findings in these runs — nothing to fix. 🎉")
    for i, a in enumerate(rep.action_items, 1):
        with st.container(border=True):
            n_col, body = st.columns([1, 14])
            n_col.markdown(f"<div class='arc-num'>{i:02d}</div>", unsafe_allow_html=True)
            with body:
                with st.container(horizontal=True):
                    st.badge(a.priority, color=PRIORITY_BADGE.get(a.priority, "gray"))
                    st.badge(a.severity, color=SEVERITY_BADGE.get(a.severity, "gray"))
                    if a.money_related:
                        st.badge("money / contact", icon="💰", color="violet")
                    st.badge(f"{a.n_runs} runs · {pct(a.share, 1)}", color="gray")
                st.markdown(f"#### {esc(a.title)}")
                st.markdown(f"**Why:** {esc(a.why)}  \n**Fix:** {esc(a.fix)}")
                st.caption(esc(a.expected_impact))
                if a.task_types:
                    st.caption("Affected task types: " + " · ".join(f"{esc(t)} ({n})" for t, n in a.task_types.items()))
                with st.expander(f"Show evidence ({len(a.evidence_runs)} runs)"):
                    for ex in a.examples[:2]:
                        st.markdown(f"> {esc(short(ex, 240))}")
                    with st.container(horizontal=True):
                        for rid in a.evidence_runs[:15]:
                            st.button(rid, key=f"ev_{i}_{rid}", on_click=goto_run, args=(rid,), type="tertiary",
                                      icon="🔎")
                    if len(a.evidence_runs) > 15:
                        st.caption(f"+ {len(a.evidence_runs) - 15} more in the JSON report")


# --------------------------------------------------------------------------- Runs
with T[TAB_RUNS]:
    st.markdown("Every run with its success probability and verdict. Filter, sort by clicking a column header, "
                "and select a row to open it in **Run detail**.")
    f1, f2, f3, f4 = st.columns(4)
    v_sel = f1.multiselect("Verdict", [LIKELY_FAILURE, UNCERTAIN, LIKELY_SUCCESS], key="f_verdict")
    t_sel = f2.multiselect("Task type", sorted(DF.task_type.unique()), key="f_task")
    m_sel = f3.multiselect("Model", sorted(DF.model.unique()), key="f_model")
    all_codes = sorted({c for cs in DF.codes for c in cs})
    c_sel = f4.multiselect("Finding type", all_codes, key="f_finding", format_func=lambda c: FINDING_SHORT.get(c, c))
    view = DF
    if v_sel:
        view = view[view.verdict.isin(v_sel)]
    if t_sel:
        view = view[view.task_type.isin(t_sel)]
    if m_sel:
        view = view[view.model.isin(m_sel)]
    if c_sel:
        view = view[view.codes.apply(lambda cs: any(c in cs for c in c_sel))]
    view = view.sort_values("p").assign(p=lambda d: d.p * 100)
    show = view[["run_id", "task_type", "model", "p", "verdict", "claimed", "top_finding", "n_findings"]
                + (["outcome"] if has_outcomes else [])].copy()
    if has_outcomes:
        show["outcome"] = show.outcome.map({1: "✅ success", 0: "❌ failed"}).fillna("–")
    show["verdict"] = show.verdict.map({LIKELY_SUCCESS: "🟢 " + LIKELY_SUCCESS, UNCERTAIN: "🟡 " + UNCERTAIN,
                                        LIKELY_FAILURE: "🔴 " + LIKELY_FAILURE})
    st.caption(f"Showing {len(show)} of {len(DF)} runs (lowest probability first).")
    sel = st.dataframe(
        show, hide_index=True, width="stretch", height=(460 if len(show) > 14 else "auto"), key="runs_table", on_select="rerun",
        selection_mode="single-row",
        column_config={
            "run_id": st.column_config.TextColumn("Run"), "task_type": "Task type", "model": "Model",
            "p": st.column_config.ProgressColumn("P(success)", min_value=0, max_value=100, format="%.0f%%"),
            "verdict": "Verdict", "claimed": st.column_config.CheckboxColumn("Claimed success"),
            "top_finding": "Top finding", "n_findings": st.column_config.NumberColumn("Findings"),
            "outcome": "Outcome (ground truth)",
        })
    rows = (sel.selection.rows if sel is not None and hasattr(sel, "selection") else []) or []
    if rows:
        rid = show.iloc[rows[0]]["run_id"]
        st.button(f"🔎 Open {rid} in Run detail", key="open_selected", on_click=goto_run, args=(rid,), type="primary")


# --------------------------------------------------------------------------- Run detail
def _status_md(status: str | None) -> str:
    return {"ok": ":green[✅ ok]", "error": ":red[❌ error]", "refused": ":orange[⛔ refused]",
            "timeout": ":orange[⏱ timeout]", "unanswered": ":gray[⌛ no result]"}.get(status or "", "")


def render_timeline(r, show_llm: bool) -> None:
    calls = {c.seq: c for c in r.tool_calls}
    evidence = {s for f in r.findings for s in f.evidence_seq}
    lines = []
    for s in r.timeline:
        if s.type in ("llm_request", "llm_response") and not show_llm:
            continue
        mark = "🔺" if s.seq in evidence else "·"
        seq = f"`{s.seq:>2}`"
        text = esc(short(s.text, 220))
        if s.type == "user_input":
            line = f"{seq} {mark} 🧑 **{text}**"
        elif s.type == "tool_call":
            c = calls.get(s.seq)
            tail = f" — {_status_md('unanswered')}" if c is not None and c.status == "unanswered" else ""
            line = f"{seq} {mark} 🛠️ {text}{tail}"
        elif s.type == "tool_result":
            color = {"ok": "green", "error": "red", "refused": "orange", "timeout": "orange"}.get(s.status or "", "gray")
            line = f"{seq} {mark} ↳ {_status_md(s.status)} :{color}[{text}]"
        elif s.type == "final_answer":
            line = f"{seq} {mark} 💬 **Final answer** (quoted below)"
        elif s.type == "escalation":
            line = f"{seq} {mark} 🙋 :blue[{text}]"
        elif s.type == "error":
            line = f"{seq} {mark} 💥 :red[{text}]"
        elif s.type in ("llm_request", "llm_response"):
            line = f"{seq} {mark} :gray[🤖 {text}]"
        else:
            line = f"{seq} {mark} :gray[{text}]"
        if s.flags:
            line += " " + " ".join(f":red-background[{esc(FINDING_SHORT.get(fl, fl))}]" for fl in s.flags)
        lines.append(line)
    st.markdown("  \n".join(lines))
    if r.final_answer:
        st.markdown(f"<div class='arc-quote'>💬 “{_html(r.final_answer)}”</div>", unsafe_allow_html=True)
    st.caption("🔺 = step cited as evidence by a finding · LLM turns hidden unless you tick the box above.")


def _html(text: str) -> str:
    import html as _h
    return _h.escape(str(text)).replace("\n", "<br>")


def gauge(p: float | None) -> go.Figure:
    v = (p or 0) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={"suffix": "%", "valueformat": ".0f"},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"},
               "bar": {"color": VERDICT_COLORS[verdict(p)], "thickness": 0.35},
               "steps": [{"range": [0, BAND_LO * 100], "color": "rgba(214,69,69,.18)"},
                         {"range": [BAND_LO * 100, BAND_HI * 100], "color": "rgba(224,161,0,.18)"},
                         {"range": [BAND_HI * 100, 100], "color": "rgba(46,158,91,.18)"}]},
        title={"text": "Success probability"}))
    fig.update_layout(height=230, margin=dict(l=35, r=45, t=50, b=5), paper_bgcolor="rgba(0,0,0,0)")
    return fig


with T[TAB_DETAIL]:
    ids = [r.run_id for r in runs]
    if st.session_state.get("detail_run") not in ids:
        st.session_state["detail_run"] = most_striking_run(rep)
    by_id = {r.run_id: r for r in runs}

    def _fmt(rid: str) -> str:
        r = by_id[rid]
        flag = " · ⚠ said done" if r.claimed_success and (r.success_probability or 0) < BAND_LO else ""
        return f"{rid} · {r.task_type or '–'} · {pct(r.success_probability)}{flag}"

    st.selectbox("Choose a run to inspect", ids, key="detail_run", format_func=_fmt,
                 help="Opens on the most striking run: the agent said “done” but the probability is lowest.")
    r = by_id[st.session_state["detail_run"]]
    v = verdict(r.success_probability)
    with st.container(horizontal=True):
        st.badge(v, color=VERDICT_BADGE[v])
        st.badge("said “done”" if r.claimed_success else "did not claim success",
                 color="violet" if r.claimed_success else "gray", icon="💬")
        if r.outcome is not None:
            st.badge("ground truth: success" if r.outcome else "ground truth: FAILED",
                     color="green" if r.outcome else "red", icon="🧪")
        for lab, val in (("agent", r.agent), ("model", r.model), ("task", r.task_type)):
            if val:
                st.badge(f"{lab}: {val}", color="gray")
        st.badge(f"{r.n_tool_calls} tool calls · {usd(r.cost_usd)}", color="gray")

    g1, g2 = st.columns([2, 3], gap="large")
    with g1:
        st.plotly_chart(gauge(r.success_probability), width="stretch", key="gauge")
    with g2:
        st.markdown("##### Why this score")
        factors = (r.prediction or {}).get("factors") or []
        if factors:
            for f in factors[:6]:
                arrow = "🔻" if f.get("effect", 0) < 0 else "🔺"
                st.markdown(f"{arrow} {esc(f.get('text', f.get('label')))}")
        else:
            st.markdown("Nothing unusual in this run's signals — the score is close to the baseline.")
        st.caption("Computed from the log signals only — never from the ground-truth outcome.")

    cl, sh = st.columns(2, gap="large")
    with cl:
        with st.container(border=True):
            st.markdown("##### 💬 Agent CLAIMED")
            st.markdown(f"<div class='arc-quote'>“{_html(short(r.final_answer, 400)) if r.final_answer else '— no final answer —'}”</div>",
                        unsafe_allow_html=True)
            st.markdown(f"- Presented the task as done: **{'yes' if r.claimed_success else 'no'}**"
                        + (" *(read from the answer text)*" if r.claim_source == "text" else ""))
            if r.claimed_actions:
                st.markdown("- Said it did: " + ", ".join(f"{code(a)}" for a in r.claimed_actions))
            if r.escalated:
                st.markdown("- Handed the case to a human")
    with sh:
        with st.container(border=True):
            st.markdown("##### 🔍 Logs SHOW")
            if r.tool_calls:
                st.markdown("**Tool calls, in order:**\n" + "\n".join(
                    f"{i}. {code(c.tool)} {_status_md(c.status) or esc(c.status)}"
                    for i, c in enumerate(r.tool_calls, 1)))
            else:
                st.markdown("**Tool calls:** none")
            if r.claimed_not_done:
                st.markdown(":red[**Claimed but never succeeded:**] " + ", ".join(f"{code(x)}" for x in r.claimed_not_done))
            if r.unrecovered_failures:
                st.markdown(":red[**Failures never fixed:**] " + ", ".join(f"{code(x)}" for x in r.unrecovered_failures))
            if not (r.claimed_not_done or r.unrecovered_failures):
                st.markdown(":green[Nothing the agent claimed contradicts the tool results.]")
            if r.outcome is not None:
                st.markdown(f"Verified outcome: **{'success' if r.outcome else 'FAILURE'}**")

    tl, fd = st.columns([3, 2], gap="large")
    with tl:
        st.markdown("##### Timeline — what happened")
        show_llm = st.checkbox("Show model (LLM) steps", value=False, key="show_llm")
        render_timeline(r, show_llm)
    with fd:
        st.markdown(f"##### Findings ({len(r.findings)})")
        if not r.findings:
            st.success("No findings in this run.")
        for f in sorted(r.findings, key=lambda f: SEVERITY_RANK.get(f.severity, 9)):
            with st.container(border=True):
                st.markdown(f":{SEVERITY_BADGE.get(f.severity, 'gray')}-badge[{f.severity}] "
                            f"**{esc(FINDING_SHORT.get(f.code, f.code))}**")
                st.markdown(esc(f.message))
                if f.evidence_seq:
                    st.caption("Evidence: step " + ", ".join(f"🔺{s}" for s in f.evidence_seq)
                               + f" · code `{f.code}` — {FINDING_CODES.get(f.code, '')}")


# --------------------------------------------------------------------------- Autonomy ladder
with T[TAB_LADDER]:
    if not rep.ladder:
        st.info("No `task_type` in these logs, so there is no per-task Autonomy ladder. Add `task_type` to "
                "`run_start` (or any event) and run the analysis again to see how much freedom each kind of task "
                "has earned.")
    else:
        st.markdown("**How much freedom has each kind of task earned?** Each task type is placed on a rung — "
                    "🟢 **Autonomous** · 🟠 **Human approves** · 🔴 **Suggest only** — from its success rate, its "
                    "false “done” claims and its critical findings. Move the sliders to set your own bar.")
        th = dict(DEFAULT_THRESHOLDS)
        with st.expander("⚙️ Thresholds", expanded=False):
            s1, s2, s3 = st.columns(3)
            th["autonomous_min_success"] = s1.slider("Autonomous: minimum success", 0.5, 1.0,
                                                     DEFAULT_THRESHOLDS["autonomous_min_success"], 0.01, key="lad_a_s")
            th["autonomous_max_false_claim"] = s1.slider("Autonomous: max false-claim rate", 0.0, 0.2,
                                                         DEFAULT_THRESHOLDS["autonomous_max_false_claim"], 0.01, key="lad_a_f")
            th["human_min_success"] = s2.slider("Human approves: minimum success", 0.2, 0.95,
                                                DEFAULT_THRESHOLDS["human_min_success"], 0.01, key="lad_h_s")
            th["human_max_false_claim"] = s2.slider("Human approves: max false-claim rate", 0.0, 0.5,
                                                    DEFAULT_THRESHOLDS["human_max_false_claim"], 0.01, key="lad_h_f")
            th["human_max_critical"] = s3.slider("Human approves: max share of runs with a critical finding", 0.0, 0.5,
                                                 DEFAULT_THRESHOLDS["human_max_critical"], 0.01, key="lad_h_c")
            th["min_runs"] = s3.slider("Runs needed before Autonomous", 1, 100, DEFAULT_THRESHOLDS["min_runs"], 1,
                                       key="lad_min")
        group_opts = {"Task type": "task_type", "Task type × agent version": ("task_type", "agent"),
                      "Task type × model": ("task_type", "model")}
        group_by = st.segmented_control("Group by", list(group_opts), default="Task type", key="lad_group")
        by = group_opts.get(group_by or "Task type")
        ladder = autonomy_ladder(runs, th, by=by)

        def _lad_name(x: dict) -> str:
            return " · ".join(str(x.get(k)) for k in ((by,) if isinstance(by, str) else by))
        rungs = ["Autonomous", "Human approves", "Suggest only"]
        cols = st.columns(3, gap="medium")
        for col, rung in zip(cols, rungs):
            with col:
                st.markdown(f"#### {RUNG_ICON[rung]} {rung}")
                rows = [x for x in ladder if x["rung"] == rung]
                if not rows:
                    st.caption("No task types on this rung.")
                for x in rows:
                    with st.container(border=True):
                        st.markdown(f"**{esc(_lad_name(x))}** · {x['n_runs']} runs")
                        st.markdown(f"Success {pct(x['success_used'])} ({x['success_source']}) · false claims "
                                    f"{pct(x['false_claim_rate'])} · critical {pct(x['critical_rate'])}")
                        st.caption("; ".join(esc(s) for s in x["reasons"]))
        with st.expander("Ladder as a table"):
            st.dataframe(pd.DataFrame([{
                "group": _lad_name(x), "rung": x["rung"], "runs": x["n_runs"],
                "success used": x["success_used"], "source": x["success_source"],
                "predicted": x["predicted_success"], "actual": x["actual_success"],
                "false claims": x["false_claim_rate"], "critical": x["critical_rate"],
                "why": "; ".join(x["reasons"]),
            } for x in ladder]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- Model accuracy
with T[TAB_ACCURACY]:
    mi = rep.model_info
    if not has_outcomes:
        st.info("**No ground truth in these logs, so accuracy can't be measured here.** The success probabilities "
                "above are still computed — from the logs alone. To check how good they are on *your* agent, log "
                "the real outcome on `run_end` for some runs and upload again:")
        st.code('{"run_id":"r-001","seq":7,"type":"run_end","outcome":{"verified_success":1,"source":"backend check"}}',
                language="json")
    else:
        A = rep.accuracy
        st.markdown(f"**How good was the success probability on this file?** {A['n']} runs have a verified outcome "
                    f"(base rate {pct(A['base_rate'])}). The outcome is never used to compute a run's probability — "
                    "it is only used here to grade it.")
        for w in A.get("warnings") or []:
            st.warning(w)
        m = st.columns(4)
        m[0].metric("AUC", "–" if A["auc"] is None else f"{A['auc']:.3f}", border=True,
                    help="How well the score ranks successes above failures: 1.0 = perfect, 0.5 = coin flip.")
        m[1].metric("Brier score", f"{A['brier']:.3f}", border=True,
                    delta=f"{A['brier'] - A['brier_baseline']:+.3f} vs baseline",
                    delta_color="inverse", help=f"Mean squared error of the probability. Lower is better. Baseline = always guessing the base rate "
                         f"({A['brier_baseline']:.3f}).")
        m[2].metric("Accuracy at 50%", pct(A["accuracy_at_0_5"], 1), border=True)
        cb = A["claimed_but_failed"]
        m[3].metric("“Done” claims that were false", f"{cb['n_claimed_but_failed']} / {cb['n_claimed']}", border=True,
                    help=f"{cb['n_flagged_by_findings']} of them were flagged by a critical/high finding.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("##### Calibration — when it says 80%, does it happen 80% of the time?")
            cal = pd.DataFrame([b for b in A["calibration"] if b["n"]])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                     line=dict(dash="dot", color="rgba(128,128,128,.7)")))
            if len(cal):
                fig.add_trace(go.Scatter(x=cal.mean_predicted, y=cal.actual_rate, mode="markers+lines+text",
                                         name="this file", text=[f"n={n}" for n in cal.n], textposition="top center",
                                         marker=dict(size=[8 + min(n, 60) / 3 for n in cal.n], color="#5b8def")))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Predicted probability",
                              yaxis_title="Actual success rate", xaxis=dict(range=[0, 1], tickformat=".0%"),
                              yaxis=dict(range=[-0.03, 1.15], tickformat=".0%"), legend=dict(orientation="h", y=1.1),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", key="calib")
        with c2:
            st.markdown("##### Confusion matrix (cut-off 50%)")
            cm = A["confusion_at_0_5"]
            z = [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]]
            fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                            x=["Predicted success", "Predicted failure"], y=["Actually succeeded", "Actually failed"])
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", key="confusion")
    st.divider()
    st.markdown("##### About the success score")
    if mi.get("method") == "model":
        cv_how = f" ({esc(mi.get('cv'))})" if mi.get("cv") else ""
        st.markdown(f"- **Model:** {esc(mi.get('kind'))} on {len(mi.get('features') or [])} general log signals  \n"
                    f"- **Trained on:** {esc(re.sub(r'(?i)^trained on\s+', '', str(mi.get('source') or '')))}  \n"
                    f"- **Chosen how:** {esc(mi.get('selection'))}  \n"
                    f"- **Cross-validated{cv_how}:** AUC {mi.get('cv_auc') or 0:.3f} · Brier {mi.get('cv_brier') or 0:.3f}"
                    + (f"  \n- **Tested on an agent it never saw** "
                       f"({esc(mi.get('evaluated_on') or mi.get('transfer_name'))}): AUC "
                       f"{mi.get('transfer_auc') or 0:.3f} · Brier {mi.get('transfer_brier') or 0:.3f}"
                       if mi.get("transfer_auc") is not None else ""))
        st.caption(esc(mi.get("note", "")))
    else:
        st.markdown("Rule-of-thumb heuristic (no trained model file found).")
    if MODEL_CARD_MD.exists():
        with st.expander("📘 Full model card"):
            st.markdown(MODEL_CARD_MD.read_text(encoding="utf-8"))
