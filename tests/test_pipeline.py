"""End-to-end pipeline tests: the same SkyDesk runs in every format (six files, five formats) must give the same report.

The 300-run bulk files are generated per session into a temp folder (fixture `bulk_samples`, see
conftest.py); the shipped 16-run demo set is tested as it is in sample_logs/.

What "the same" means per format (decided from the format conventions, see docs/LOG_FORMAT.md):
  * Python log, CSV, OpenTelemetry flat spans, OTLP: carry every field the analysis uses ->
    identical run ids, tool sequences, statuses, findings, probabilities (1e-6) and action items.
  * OpenAI messages: the format has no claimed_success / claimed_actions / escalation / error events,
    so it is compared (a) directly with ARL on everything it CAN carry (run ids, tool sequences,
    statuses, outcomes, task types, models) and (b) fully against the ARL file passed through the
    OpenAI writer -- i.e. same information in, same report out.
  * Unlabeled ARL: identical analysis, no accuracy block.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc import pipeline  # noqa: E402
from arc.detect import UnrecognizedFormatError, load_any  # noqa: E402
from arc.export import export  # noqa: E402

from conftest import DEMO_ARL, DEMO_LOG, SKY_POLICY as POLICY  # noqa: E402

FULL_FORMATS = ["skydesk_python.log", "skydesk_otel_spans.json", "skydesk_events.csv"]
TOL = 1e-6


@pytest.fixture(scope="module")
def reports(bulk_samples):
    cache: dict[str, pipeline.Report] = {}

    def get(name: str) -> pipeline.Report:
        if name not in cache:
            cache[name] = pipeline.run(bulk_samples / name, policy=POLICY)
        return cache[name]
    return get


def by_id(rep):
    return {r.run_id: r for r in rep.runs}


def finding_keys(r):
    return sorted((f.code, f.severity, f.tool) for f in r.findings)


def assert_same_analysis(a, b, run_ids=None, label=""):
    A, B = by_id(a), by_id(b)
    ids = set(run_ids) if run_ids is not None else set(A)
    assert ids <= set(A) and ids <= set(B), label
    for rid in sorted(ids):
        x, y = A[rid], B[rid]
        assert x.tools_called == y.tools_called, (label, rid)
        assert [c.status for c in x.tool_calls] == [c.status for c in y.tool_calls], (label, rid)
        assert finding_keys(x) == finding_keys(y), (label, rid)
        assert x.claimed_success == y.claimed_success, (label, rid)
        assert x.escalated == y.escalated and x.has_final_answer == y.has_final_answer, (label, rid)
        assert abs(x.success_probability - y.success_probability) <= TOL, (label, rid)
        assert x.features() == pytest.approx(y.features(), abs=TOL), (label, rid)


# --------------------------------------------------------------------------- formats that carry everything

@pytest.mark.parametrize("name", FULL_FORMATS)
def test_full_formats_match_arl(reports, name):
    arl, other = reports("skydesk_arl.jsonl"), reports(name)
    assert set(by_id(arl)) == set(by_id(other)) and len(arl.runs) == 300
    assert_same_analysis(arl, other, label=name)
    # outcomes, and therefore the accuracy block, agree too
    assert [r.outcome for r in arl.runs] == [by_id(other)[r.run_id].outcome for r in arl.runs]
    assert other.accuracy["auc"] == pytest.approx(arl.accuracy["auc"], abs=TOL)
    assert other.accuracy["brier"] == pytest.approx(arl.accuracy["brier"], abs=TOL)
    # same prioritised action items
    key = lambda rep: [(a.code, a.title, a.n_runs, a.severity) for a in rep.action_items]  # noqa: E731
    assert key(other) == key(arl)


def test_otlp_subset_matches_arl(reports):
    arl, otlp = reports("skydesk_arl.jsonl"), reports("skydesk_otlp.json")
    assert otlp.source["format"] == "otel" and len(otlp.runs) == 60
    assert_same_analysis(arl, otlp, run_ids=by_id(otlp).keys(), label="otlp")


# --------------------------------------------------------------------------- OpenAI messages

def test_openai_matches_arl_on_what_it_carries(reports):
    arl, oa = reports("skydesk_arl.jsonl"), reports("skydesk_openai_messages.jsonl")
    A, B = by_id(arl), by_id(oa)
    assert set(A) == set(B)
    for rid, x in A.items():
        y = B[rid]
        assert x.tools_called == y.tools_called, rid
        assert [c.status for c in x.tool_calls] == [c.status for c in y.tool_calls], rid
        assert (x.outcome, x.task_type, x.model, x.agent) == (y.outcome, y.task_type, y.model, y.agent), rid
        assert x.final_answer == y.final_answer, rid
    assert oa.accuracy is not None and oa.accuracy["n"] == 300


def test_openai_matches_arl_through_openai_writer(reports, bulk_samples):
    """Same information in -> same report out (claims inferred from text in both)."""
    events, _ = load_any(bulk_samples / "skydesk_arl.jsonl")
    projected = pipeline.run(export(events, "openai").encode(), "projected.jsonl", policy=POLICY)
    assert_same_analysis(projected, reports("skydesk_openai_messages.jsonl"), label="openai")


def test_claim_inference_agrees_with_logged_claims(reports):
    """Where ARL logs claimed_success, the text heuristic (used for OpenAI logs) must agree."""
    arl, oa = reports("skydesk_arl.jsonl"), reports("skydesk_openai_messages.jsonl")
    B = by_id(oa)
    explicit = [r for r in arl.runs if r.claim_source == "explicit"]
    agree = sum(r.claimed_success == B[r.run_id].claimed_success for r in explicit)
    assert len(explicit) > 200 and agree == len(explicit)


# --------------------------------------------------------------------------- unlabeled + report contents

def test_unlabeled_file_has_no_accuracy_block(reports):
    arl, un = reports("skydesk_arl.jsonl"), reports("skydesk_unlabeled_arl.jsonl")
    assert un.accuracy is None and not un.has_outcomes
    assert all(r.outcome is None for r in un.runs)
    assert_same_analysis(arl, un, label="unlabeled")
    assert un.overview["actual_success_rate"] is None
    assert un.cost["per_verified_success_usd"] is None and un.cost["per_expected_success_usd"] > 0
    d = un.to_dict(include_runs=False)
    assert d["accuracy"] is None and d["action_items"]


def test_report_blocks(reports):
    rep = reports("skydesk_arl.jsonl")
    acc = rep.accuracy
    assert acc["n"] == 300 and 0 <= acc["brier"] <= 1 and acc["auc"] > 0.8
    cm = acc["confusion_at_0_5"]
    assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == 300
    assert len(acc["calibration"]) == 5 and sum(b["n"] for b in acc["calibration"]) == 300
    assert rep.model_info["method"] in ("model", "heuristic")
    assert rep.cost["total_usd"] > 0 and len(rep.cost["by_agent_model"]) == 2
    assert {row["task_type"] for row in rep.ladder} >= {"refund_request", "rebooking"}
    r = rep.runs[0]
    assert r.timeline and r.prediction["factors"] is not None and 0 < r.success_probability < 1
    top = [a.code for a in rep.action_items[:4]]
    assert "CLAIMED_NOT_DONE" in top and "FALSE_SUCCESS_AFTER_FAILURE" in top
    json.loads(rep.to_json())                      # serialisable
    md = rep.to_markdown()
    assert "# Agent Report Card" in md and "Fix these first" in md and "How accurate" in md


def test_planted_failures_are_found(reports):
    """Each planted failure mode is caught by the matching finding (ground truth from run_end.meta)."""
    rep = reports("skydesk_arl.jsonl")
    planted = {e.run_id: (e.meta or {}).get("planted_failure") for e in rep.events if e.type == "run_end"}
    expect = {"obeyed_injection": "INJECTION_OBEYED", "money_before_verification": "PREREQ_MISSING",
              "cap_exceeded": "CAP_EXCEEDED", "false_success_claim": "FALSE_SUCCESS_AFTER_FAILURE",
              "missed_escalation": "ESCALATION_MISSED", "missing_final_answer": "NO_FINAL_ANSWER",
              "unanswered_last_call": "UNANSWERED_CALL", "retry_loop": "RETRY_LOOP"}
    for mode, code in expect.items():
        runs = [r for r in rep.runs if planted[r.run_id] == mode]
        assert runs and all(code in r.finding_codes for r in runs), mode
    clean = [r for r in rep.runs if planted[r.run_id] in ("clean_success", "correct_escalation")]
    assert not any(f.severity in ("critical", "high") for r in clean for f in r.findings)


def test_demo_files(tmp_path):
    a = pipeline.run(DEMO_ARL, policy=POLICY)
    b = pipeline.run(DEMO_LOG, policy=POLICY)
    assert len(a.runs) == 16 and b.source["skipped_lines"] == 3
    assert_same_analysis(a, b, label="demo")
    assert [r.outcome for r in a.runs] == [r.outcome for r in b.runs]


def test_demo_set_tells_every_story():
    """One run per scenario; the analyzer gets each one right (finding + probability side)."""
    rep = pipeline.run(DEMO_ARL, policy=POLICY)
    planted = {e.run_id: (e.meta or {}).get("planted_failure") for e in rep.events if e.type == "run_end"}
    by_mode = {}
    for r in rep.runs:
        by_mode.setdefault(planted[r.run_id], []).append(r)
    expect = {"obeyed_injection": "INJECTION_OBEYED", "money_before_verification": "PREREQ_MISSING",
              "cap_exceeded": "CAP_EXCEEDED", "false_success_claim": "FALSE_SUCCESS_AFTER_FAILURE",
              "claimed_action_never_executed": "CLAIMED_NOT_DONE", "missed_escalation": "ESCALATION_MISSED",
              "missing_final_answer": "NO_FINAL_ANSWER", "unanswered_last_call": "UNANSWERED_CALL",
              "retry_loop": "RETRY_LOOP"}
    for mode, code in expect.items():
        assert by_mode[mode] and all(code in r.finding_codes for r in by_mode[mode]), mode
    # human-in-the-loop done right: escalated, no serious finding, verified and predicted success
    for mode in ("correct_escalation", "approval_escalation"):
        (r,) = by_mode[mode]
        assert r.escalated and r.outcome == 1 and r.success_probability > 0.9, mode
        assert not any(f.severity in ("critical", "high") for f in r.findings), mode
    assert {"timeout", "unnecessary_escalation", "clean_success"} <= set(by_mode)
    assert any("INJECTION_PRESENT" in r.finding_codes for r in by_mode["clean_success"])   # injection resisted
    assert rep.accuracy["n"] == 16 and rep.accuracy["auc"] > 0.95


def test_bytes_and_unknown_input():
    data = DEMO_ARL.read_bytes()
    rep = pipeline.run(data, "upload.jsonl")          # default policy, bytes in
    assert rep.policy["origin"] == "built-in generic default" and len(rep.runs) == 16
    with pytest.raises(UnrecognizedFormatError) as e:
        pipeline.run(b"hello world\nthis is not a log\n", "notes.txt")
    assert "LOG_FORMAT.md" in str(e.value)


def test_heuristic_mode():
    rep = pipeline.run(DEMO_ARL, policy=POLICY, model=None)
    assert rep.model_info["method"] == "heuristic"
    assert all(r.prediction["method"] == "heuristic" for r in rep.runs)


# --------------------------------------------------------------------------- CLI

def test_cli_analyze_json_and_md(tmp_path):
    out = tmp_path / "r.json"
    cmd = [sys.executable, "-m", "arc", "analyze", str(DEMO_LOG), "--policy", str(POLICY), "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    d = json.loads(out.read_text())
    assert d["overview"]["n_runs"] == 16 and d["action_items"] and len(d["runs"]) == 16
    md = tmp_path / "r.md"
    subprocess.run(cmd[:-1] + [str(md)], check=True, cwd=ROOT, capture_output=True)
    assert md.read_text().startswith("# Agent Report Card")


def test_cli_unknown_file_exit_code(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("nothing to see\n")
    p = subprocess.run([sys.executable, "-m", "arc", "analyze", str(bad)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 2 and "Accepted formats" in p.stderr


# --------------------------------------------------------------------------- bundled model

def test_bundled_model_file():
    from arc import model as M
    path = ROOT / "arc/models/success_model.json"
    if not path.exists():
        pytest.skip("no bundled model")
    m = M.load_model(path)
    assert m["provenance"].get("source")
    assert m["metrics"]["n"] > 0 and m["metrics"]["auc"] > 0.7
    for f, c in zip(m["features"], m["coef"]):           # sign constraints hold
        s = M.EXPECTED_SIGNS.get(f)
        assert not s or c * s >= 0, f


# --------------------------------------------------------------------------- shipped data is reproducible

def test_shipped_demo_set_matches_generator(generator, tmp_path):
    """sample_logs/ is exactly what `python scripts/generate_sample_logs.py` writes (deterministic)."""
    info = generator.write_demo(tmp_path, seed=7)
    for name in ("demo_agent_runs.jsonl", "demo_agent_runs.log", "skydesk_policy.yaml", "shopdesk_policy.yaml",
                 "README.md"):
        assert (tmp_path / name).read_bytes() == (DEMO_ARL.parent / name).read_bytes(), name
    assert len(info["runs"]) == 16 and len({r["scenario"] for r in info["runs"]}) == 16
