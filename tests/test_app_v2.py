"""AppTest coverage for the v2 app (app.py): every sample, every upload format, malformed uploads,
policy options, navigation from evidence to Run detail. All analysis comes from arc.pipeline."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
TIMEOUT = 180

TABS = ["Summary", "Action items", "Runs", "Run detail", "Autonomy ladder", "Model accuracy"]
# exact sample labels (also used by the automated demo recording)
SAMPLES = [
    "SkyDesk demo · 16 runs · Python log (demo_agent_runs.log) — start here",
    "SkyDesk demo · 16 runs · ARL JSONL (demo_agent_runs.jsonl)",
    "Example · 3 runs · ARL JSONL",
    "Example · 3 runs · Python log",
    "Example · 3 runs · OpenAI chat messages",
    "Example · 3 runs · OpenTelemetry spans",
    "Example · 3 runs · OpenTelemetry OTLP JSON",
    "Example · 3 runs · CSV",
    "Example · 3 runs · Langfuse export",
]
QUICKSTART = SAMPLES[0]


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _fresh():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    _ok(at)
    return at


def _analyse(at):
    at.button(key="run_analysis").click().run()
    _ok(at)
    return at


def _assert_results(at, has_outcomes=None):
    labels = [t.label for t in at.tabs]
    for t in TABS:
        assert t in labels, labels
    metrics = {m.label: m.value for m in at.metric}
    for want in ["Runs analysed", "Predicted success", "Claimed success but likely failed",
                 "Critical policy violations", "Est. cost per likely-successful run"]:
        assert want in metrics, metrics
    if has_outcomes is True:
        assert "AUC" in metrics
    if has_outcomes is False:
        assert "AUC" not in metrics
        assert any("No ground truth" in i.value for i in at.info)
    assert at.selectbox(key="detail_run").value
    assert any(d.label.startswith("⬇ Download full report (JSON)") for d in at.get("download_button"))
    return metrics


def test_landing_page_guides_the_user():
    at = _fresh()
    subs = [s.value for s in at.subheader]
    assert "Step 1 · Upload your agent logs" in subs
    assert "Step 2 · Policy (optional)" in subs
    assert "Step 3 · Run the analysis" in subs
    assert at.button(key="run_analysis").disabled
    assert at.button(key="run_analysis").label == "▶ Run analysis"
    assert any("Run analysis" in i.value for i in at.info)
    assert at.selectbox(key="sample").options[1:] == SAMPLES
    # the format help is present with the minimal example and download buttons
    assert any(e.label == "📄 What format should my logs be in?" for e in at.expander)
    dl = [d.label for d in at.get("download_button")]
    assert "⬇ arl_logger.py" in dl and "⬇ ARL JSONL" in dl and "⬇ CSV" in dl and "⬇ Langfuse export" in dl


@pytest.mark.parametrize("sample", SAMPLES)
def test_each_sample_loads_and_analyses(sample):
    at = _fresh()
    at.selectbox(key="sample").set_value(sample).run()
    _ok(at)
    assert any("Detected:" in s.value for s in at.success)
    assert at.radio(key="policy_choice").value == "SkyDesk policy"
    _analyse(at)
    m = _assert_results(at, has_outcomes=True)
    assert m["Runs analysed"] == ("16" if "16 runs" in sample else "3")


def test_demo_numbers_and_default_run():
    at = _fresh()
    at.selectbox(key="sample").set_value(QUICKSTART).run()
    _analyse(at)
    m = _assert_results(at, has_outcomes=True)
    assert m["Runs analysed"] == "16"
    assert m["Claimed success but likely failed"] == "7"
    # numbers quoted in README, docs/DEMO_SCRIPT.md and video/narration.json
    assert m["Predicted success"] == "31%" and m["Critical policy violations"] == "3"
    assert m["Est. cost per likely-successful run"] == "$0.0123" and m["“Done” claims that were false"] == "7 / 11"
    assert m["AUC"] == "1.000" and m["Accuracy at 50%"] == "100.0%"
    acc_md = " ".join(x.value for x in at.markdown)
    assert "Tested on an agent it never saw" in acc_md and "AUC 0.991" in acc_md and "AUC 0.985" in acc_md
    assert "ShopDesk" in acc_md
    # most striking run: claimed success with the lowest probability
    assert at.selectbox(key="detail_run").value == "sky-0904-0028"
    md = " ".join(x.value for x in at.markdown)
    assert "Agent CLAIMED" in md and "Logs SHOW" in md
    assert "Timeline" in md


def test_evidence_button_opens_run_detail():
    at = _fresh()
    at.selectbox(key="sample").set_value(QUICKSTART).run()
    _analyse(at)
    ev = [b for b in at.button if b.key and b.key.startswith("ev_")]
    assert ev, "action items should offer evidence run buttons"
    target = ev[-1].label
    ev[-1].click().run()
    _ok(at)
    assert at.session_state["results_tab"] == "Run detail"
    assert at.selectbox(key="detail_run").value == target


def test_runs_filters_and_ladder_sliders():
    at = _fresh()
    at.selectbox(key="sample").set_value(QUICKSTART).run()
    _analyse(at)
    at.multiselect(key="f_verdict").set_value(["Likely failure"]).run()
    _ok(at)
    at.multiselect(key="f_finding").set_value(["RETRY_LOOP"]).run()
    _ok(at)
    assert any("Showing 1 of 16 runs" in c.value for c in at.caption)
    at.slider(key="lad_min").set_value(1).run()
    at.slider(key="lad_h_s").set_value(0.3).run()
    _ok(at)
    at.segmented_control(key="lad_group").set_value("Task type × agent version").run()
    _ok(at)
    at.checkbox(key="show_llm").check().run()
    _ok(at)


@pytest.mark.parametrize("fname", [
    "sample_logs/demo_agent_runs.jsonl", "sample_logs/demo_agent_runs.log", "examples/example_openai_messages.jsonl",
    "examples/example_otel_spans.json", "examples/example_otlp.json", "examples/example_events.csv",
    "examples/example_langfuse.json",
])
def test_upload_each_format_as_bytes(fname):
    at = _fresh()
    data = (ROOT / fname).read_bytes()
    at.file_uploader(key="uploads").set_value((fname, data, "application/octet-stream")).run()
    _ok(at)
    assert any("Detected:" in s.value for s in at.success)
    at.radio(key="policy_choice").set_value("SkyDesk policy").run()
    _analyse(at)
    _assert_results(at)


def test_upload_without_ground_truth_hides_accuracy():
    """The demo runs with run_end.outcome removed: probabilities, findings and actions, no accuracy block."""
    import json
    lines = []
    for ln in (ROOT / "sample_logs" / "demo_agent_runs.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(ln)
        e.pop("outcome", None)
        lines.append(json.dumps(e))
    at = _fresh()
    at.file_uploader(key="uploads").set_value(("unlabeled.jsonl", "\n".join(lines).encode(), "application/jsonl")).run()
    at.radio(key="policy_choice").set_value("SkyDesk policy").run()
    _analyse(at)
    m = _assert_results(at, has_outcomes=False)
    assert m["Runs analysed"] == "16"


def test_upload_sample_raw_as_bytes():
    at = _fresh()
    p = ROOT / "examples" / "example_openai_messages.jsonl"
    at.file_uploader(key="uploads").set_value((p.name, p.read_bytes(), "application/jsonl")).run()
    _ok(at)
    assert any("OpenAI" in s.value for s in at.success)


def test_multiple_uploads_are_merged_with_clash_prefix():
    at = _fresh()
    a = (ROOT / "sample_logs" / "demo_agent_runs.jsonl").read_bytes()
    b = (ROOT / "sample_logs" / "demo_agent_runs.log").read_bytes()
    at.file_uploader(key="uploads").set_value([("a.jsonl", a, "application/jsonl"), ("b.log", b, "text/plain")]).run()
    _ok(at)
    assert any("32" in s.value and "runs" in s.value for s in at.success)
    _analyse(at)
    assert {m.label: m.value for m in at.metric}["Runs analysed"] == "32"


@pytest.mark.parametrize("name,data", [
    ("garbage.jsonl", b"this is not a log\nneither is this\n"),
    ("empty.json", b""),
    ("weird.json", b'{"hello": "world"}'),
    ("binary.log", bytes(range(256)) * 4),
    ("bad.csv", b"a,b,c\n1,2,3\n"),
])
def test_malformed_upload_shows_error_not_crash(name, data):
    at = _fresh()
    at.file_uploader(key="uploads").set_value((name, data, "application/octet-stream")).run()
    _ok(at)
    assert any("couldn't read this file" in e.value for e in at.error)
    assert at.button(key="run_analysis").disabled


def test_policy_options():
    at = _fresh()
    at.selectbox(key="sample").set_value(QUICKSTART).run()
    for opt in ["Default generic policy", "SkyDesk policy"]:
        at.radio(key="policy_choice").set_value(opt).run()
        _ok(at)
        assert any("In plain English" in m.value for m in at.markdown)
    at.radio(key="policy_choice").set_value("Edit YAML").run()
    _ok(at)
    assert "SkyDesk" in at.text_area(key="policy_yaml").value
    at.text_area(key="policy_yaml").set_value("name: tiny\nmax_tool_retries: 0\nforbidden_tools: [update_contact]\n").run()
    _ok(at)
    assert any("tiny" in m.value for m in at.markdown)
    _analyse(at)
    _assert_results(at)
    at.text_area(key="policy_yaml").set_value("caps: [not, a, mapping\n").run()
    _ok(at)
    assert any("could not be read" in w.value for w in at.warning)
    assert at.button(key="run_analysis").disabled
    at.radio(key="policy_choice").set_value("Upload YAML").run()
    _ok(at)
    assert at.button(key="run_analysis").disabled
    pol = (ROOT / "sample_logs" / "skydesk_policy.yaml").read_bytes()
    at.file_uploader(key="policy_file").set_value(("p.yaml", pol, "application/x-yaml")).run()
    _ok(at)
    assert not at.button(key="run_analysis").disabled


def test_stale_report_warning_after_input_change():
    at = _fresh()
    at.selectbox(key="sample").set_value(QUICKSTART).run()
    _analyse(at)
    at.radio(key="policy_choice").set_value("Default generic policy").run()
    _ok(at)
    assert any("changed since this report" in w.value for w in at.warning)


def test_policy_format_doc_examples_are_valid():
    """docs/POLICY_FORMAT.md: every YAML example loads; the annotated one equals the SkyDesk policy."""
    from arc.policy import DEFAULT_POLICY, load_policy, policy_warnings
    md = (ROOT / "docs" / "POLICY_FORMAT.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", md, flags=re.S)
    assert len(blocks) >= 3
    for b in blocks:
        assert not policy_warnings(load_policy(b))
    minimal = blocks[0]
    assert len([ln for ln in minimal.splitlines() if ln.strip()]) == 5
    pol = load_policy(minimal)
    assert pol.require_before == {"issue_refund": ["verify_customer"]} and pol.caps and pol.forbidden_tools
    annotated = load_policy(blocks[1]).to_dict()
    assert annotated == load_policy(str(ROOT / "sample_logs" / "skydesk_policy.yaml")).to_dict()
    for key in DEFAULT_POLICY:                     # every real key is documented
        assert f"`{key}`" in md, key
    assert "max_from" in md and "re:" in md


def test_policy_format_expander():
    at = _fresh()
    exp = [e for e in at.expander if e.label == "📐 What should a policy look like?"]
    assert len(exp) == 1
    labels = [t.label for t in exp[0].tabs]
    assert labels == ["Annotated example (SkyDesk)", "All keys", "Minimal template", "Download examples"]
    codes = [c.value for c in exp[0].code]
    assert any("max_from: lookup_booking.result.fare_paid" in c and "require_before" in c for c in codes)
    assert any(c.startswith("name: my-support-agent") for c in codes)
    assert any("max_failed_verifications" in m.value for m in exp[0].markdown)
    assert any("docs/POLICY_FORMAT.md" in c.value for c in exp[0].caption)
    dls = [d.label for d in at.get("download_button")]
    for want in ["⬇ skydesk_policy.yaml", "⬇ shopdesk_policy.yaml", "⬇ policy_template.yaml"]:
        assert want in dls, dls


def test_minimal_template_works_in_edit_yaml():
    md = (ROOT / "docs" / "POLICY_FORMAT.md").read_text(encoding="utf-8")
    minimal = re.findall(r"```yaml\n(.*?)```", md, flags=re.S)[0]
    at = _fresh()
    at.selectbox(key="sample").set_value(QUICKSTART).run()
    at.radio(key="policy_choice").set_value("Edit YAML").run()
    at.text_area(key="policy_yaml").set_value(minimal).run()
    _ok(at)
    assert any("my-support-agent" in m.value for m in at.markdown)
    assert not any("could not be read" in w.value for w in at.warning)
    _analyse(at)
    _assert_results(at)
    # a misspelt key is a warning, not an error
    at.text_area(key="policy_yaml").set_value(minimal + "money_tool: [x]\n").run()
    _ok(at)
    assert any("Unknown policy key(s) ignored: money_tool" in w.value for w in at.warning)
    assert not at.button(key="run_analysis").disabled
