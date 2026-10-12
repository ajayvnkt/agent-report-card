"""One call from "here is a log file" to "here is the report card".

    from arc.pipeline import run
    rep = run("sample_logs/demo_agent_runs.log", policy="sample_logs/skydesk_policy.yaml")
    rep.overview["n_runs"], rep.action_items[0].title, rep.accuracy["auc"]
    rep.to_json(); rep.to_markdown()

Steps: detect the format -> normalise to ARL events -> summarise each run (timeline, findings)
-> success probability with explanation factors -> prioritised action items -> autonomy ladder
-> cost -> accuracy block (only when the logs carry ground-truth outcomes).

CLI: ``python -m arc analyze <file> [--policy p.yaml] [--out report.json|report.md]``
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import model as _model
from .analyze import AnalysisResult, RunSummary, analyze
from .detect import load_any
from .policy import load_policy, policy_warnings
from .schema import Event

# cost per expected success is only shown when at least this many successes are expected; below that
# the division explodes (e.g. $0.07 / 0.000005 = $14,000 per success) and means nothing.
MIN_EXPECTED_SUCCESSES = 1.0
ACCURACY_MIN_RUNS = 10     # fewer labelled runs -> numbers shown but flagged as too few to trust


@dataclass
class Report:
    """Everything the app (or the CLI) shows for one upload."""
    source: dict                     # load_any report: format, format_label, confidence, n_runs, n_events, skipped_*, warnings
    policy: dict                     # {"name", "origin"}
    model_info: dict                 # {"method": "model"|"heuristic", "provenance", "features", ...}
    overview: dict                   # headline numbers (analyze.overview)
    runs: list[RunSummary]           # per run: timeline, findings, success_probability, prediction.factors
    action_items: list               # actions.ActionItem, highest priority first
    ladder: list[dict]               # autonomy ladder rows ([] when no task_type)
    cost: dict                       # totals + per agent/model, cost per (verified|expected) success
    accuracy: dict | None            # only when outcomes exist: AUC, Brier, confusion @0.5, calibration bins
    notes: list[str] = field(default_factory=list)    # plain-English caveats for the reader
    events: list[Event] = field(default_factory=list, repr=False)
    analysis: AnalysisResult | None = field(default=None, repr=False)

    # -- convenience --------------------------------------------------------
    def run(self, run_id: str) -> RunSummary:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        raise KeyError(run_id)

    def runs_table(self):
        return self.analysis.runs_table()

    def findings_table(self):
        return self.analysis.findings_table()

    @property
    def has_outcomes(self) -> bool:
        return self.accuracy is not None

    def to_dict(self, include_runs: bool = True) -> dict:
        d = {
            "source": self.source, "policy": self.policy, "model": self.model_info,
            "overview": self.overview, "cost": self.cost, "accuracy": self.accuracy, "notes": self.notes,
            "action_items": [a.to_dict() if hasattr(a, "to_dict") else a for a in self.action_items],
            "ladder": self.ladder,
        }
        if include_runs:
            d["runs"] = [_run_dict(r) for r in self.runs]
        return _clean(d)

    def to_json(self, include_runs: bool = True, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(include_runs), indent=indent, ensure_ascii=False, default=str)

    def to_markdown(self, max_items: int = 10, max_runs: int = 15) -> str:
        return _markdown(self, max_items, max_runs)


# --------------------------------------------------------------------------- main entry points

def run(path_or_bytes: Any, filename: str | None = None, policy: Any = None, *, model: Any = "default",
        fmt: str | None = None, ladder_thresholds: dict | None = None) -> Report:
    """Analyse one log file (path, bytes or Streamlit UploadedFile) in any supported format.

    policy: None (generic default), a path / YAML-JSON text / bytes / dict / Policy.
    model:  "default" (bundled arc/models/success_model.json if present, else heuristic),
            None (heuristic only), a model dict or a path.
    fmt:    force a format (arl|openai|pylog|otel|csv) instead of auto-detection.
    Raises UnrecognizedFormatError (ValueError) with the accepted formats when nothing matches.
    """
    events, src = load_any(path_or_bytes, filename=filename, fmt=fmt)
    return analyze_events(events, source=src, policy=policy, model=model, ladder_thresholds=ladder_thresholds)


def analyze_events(events: list, source: dict | None = None, policy: Any = None, *, model: Any = "default",
                   ladder_thresholds: dict | None = None) -> Report:
    """Same as :func:`run` for events you already have (ARL Event objects or dicts)."""
    pol = load_policy(policy)
    res = analyze(events, policy=pol, model=model, ladder_thresholds=ladder_thresholds)
    mdl = _model.resolve_model(model)
    notes: list[str] = list(policy_warnings(pol))
    info = _model_info(mdl)
    accuracy = _accuracy_block(res.runs, mdl, notes)
    if not any(r.task_type for r in res.runs):
        notes.append("No task_type in the logs, so the Autonomy Ladder is not shown. Add task_type to "
                     "run_start (or to any event) to get one rung per task.")
    if any(r.claim_source == "text" for r in res.runs):
        n = sum(r.claim_source == "text" for r in res.runs)
        notes.append(f"{n} run(s) did not log claimed_success, so whether the agent said \"done\" was read "
                     "from the final-answer text (keyword rules).")
    src = dict(source or {"format": "events", "format_label": "ARL events (in memory)", "confidence": 1.0,
                          "n_runs": len(res.runs), "n_events": sum(r.n_events for r in res.runs),
                          "skipped_lines": 0, "skipped_examples": [], "warnings": []})
    return Report(source=src, policy={"name": pol.name, "origin": _policy_origin(policy)},
                  model_info=info, overview=res.overview, runs=res.runs, action_items=res.action_items,
                  ladder=res.ladder, cost=cost_block(res.runs), accuracy=accuracy, notes=notes,
                  events=list(events), analysis=res)


# --------------------------------------------------------------------------- blocks

def _policy_origin(policy: Any) -> str:
    if policy is None:
        return "built-in generic default"
    if isinstance(policy, (str, Path)) and len(str(policy)) < 512 and "\n" not in str(policy):
        return str(policy)
    if isinstance(policy, dict):
        return "dict"
    return type(policy).__name__


def _model_info(mdl: dict | None) -> dict:
    if not mdl:
        return {"method": "heuristic", "rules": _model.HEURISTIC_RULES_TEXT,
                "note": "No trained model file found; using the transparent rule-of-thumb score."}
    prov = mdl.get("provenance") or {}
    m = mdl.get("metrics") or {}
    transfers = {k: v for k, v in (m.get("transfer") or {}).items() if isinstance(v, dict)}
    # the held-out set to show: the one named in the provenance, else "*_all", else the first listed
    tr_name = prov.get("transfer_primary") if prov.get("transfer_primary") in transfers else         next((k for k in transfers if k.endswith("_all")), next(iter(transfers), None))
    tr = transfers.get(tr_name) or {}
    return {
        "method": "model", "kind": mdl.get("kind"), "features": mdl.get("features"),
        "source": prov.get("source"), "trained_on_agents": prov.get("trained_on_agents"),
        "trained_at": prov.get("trained_at"), "selection": prov.get("selection"),
        "cv": prov.get("cv"), "cv_auc": m.get("auc"), "cv_brier": m.get("brier"),
        "evaluated_on": prov.get("evaluated_on"), "transfer_name": tr_name, "transfer_auc": tr.get("auc"), "transfer_brier": tr.get("brier"),
        "note": "Predicts from log signals only (failures, claims, policy findings) -- it cannot see whether "
                "the backend really did the work. See arc/models/MODEL_CARD.md.",
    }


def _accuracy_block(runs: list[RunSummary], mdl: dict | None, notes: list[str]) -> dict | None:
    lab = [r for r in runs if r.outcome is not None and r.success_probability is not None]
    if not lab:
        return None
    y = [int(r.outcome) for r in lab]
    p = [float(r.success_probability) for r in lab]
    ev = _model.evaluate(p, y)
    acc = {
        "n": ev["n"], "n_runs_total": len(runs), "base_rate": ev["base_rate"],
        "auc": ev["auc"], "brier": ev["brier"], "brier_baseline": ev["brier_baseline"],
        "brier_skill": (1 - ev["brier"] / ev["brier_baseline"]) if ev["brier_baseline"] else None,
        "accuracy_at_0_5": ev["accuracy_at_0_5"],
        "confusion_at_0_5": ev["confusion"],
        "mean_predicted": ev["mean_predicted"],
        "calibration": ev["calibration_bins"],
        "method": "model" if mdl else "heuristic",
        "claimed_but_failed": _claimed_but_failed(lab),
        "warnings": [],
    }
    if ev["auc"] is None:
        acc["warnings"].append("All labelled runs have the same outcome, so AUC is undefined.")
    if len(lab) < ACCURACY_MIN_RUNS:
        acc["warnings"].append(f"Only {len(lab)} runs have a verified outcome -- too few to trust these numbers.")
    trained_on = set(((mdl or {}).get("provenance") or {}).get("trained_on_agents") or [])
    seen = trained_on & {r.agent for r in lab if r.agent}
    if seen:
        acc["warnings"].append(
            f"The success model was trained on runs of {', '.join(sorted(seen))}, so these numbers are in-sample "
            "(optimistic). Use the cross-validated numbers in MODEL_CARD.md for this agent.")
    notes.extend(acc["warnings"])
    return acc


def _claimed_but_failed(runs: list[RunSummary]) -> dict:
    cl = [r for r in runs if r.claimed_success]
    bad = [r for r in cl if r.outcome == 0]
    caught = [r for r in bad if r.findings and any(f.severity in ("critical", "high") for f in r.findings)]
    return {"n_claimed": len(cl), "n_claimed_but_failed": len(bad),
            "rate": (len(bad) / len(cl)) if cl else None,
            "n_flagged_by_findings": len(caught),
            "example_runs": [r.run_id for r in bad[:10]]}


def cost_block(runs: list[RunSummary]) -> dict:
    """Total cost and cost per (verified or expected) success, overall and per agent/model."""
    def block(rs: list[RunSummary]) -> dict:
        n = len(rs)
        cost = sum(r.cost_usd for r in rs)
        probs = [r.success_probability for r in rs if r.success_probability is not None]
        lab = [r for r in rs if r.outcome is not None]
        wins = sum(r.outcome for r in lab)
        return {
            "n_runs": n, "total_usd": cost, "per_run_usd": cost / n if n else None,
            "tokens_in": sum(r.tokens_in for r in rs), "tokens_out": sum(r.tokens_out for r in rs),
            "expected_successes": sum(probs) if probs else None,
            "per_expected_success_usd": (cost / sum(probs)) if probs and sum(probs) >= MIN_EXPECTED_SUCCESSES else None,
            "verified_successes": wins if lab else None,
            "per_verified_success_usd": (sum(r.cost_usd for r in lab) / wins) if lab and wins else None,
            "estimated": any(r.cost_estimated for r in rs),
        }
    by: dict[tuple, list[RunSummary]] = defaultdict(list)
    for r in runs:
        by[(r.agent or "(unspecified)", r.model or "(unspecified)")].append(r)
    out = block(runs)
    out["by_agent_model"] = [{"agent": a, "model": m, **block(rs)} for (a, m), rs in sorted(by.items())]
    out["note"] = ("Cost uses cost_usd from the logs when present, otherwise tokens x the price table in "
                   "arc/analyze.py (PRICE_TABLE), otherwise ~4 characters per token.")
    return out


# --------------------------------------------------------------------------- serialisation helpers

def _run_dict(r: RunSummary) -> dict:
    d = r.to_dict()
    d["finding_codes"] = r.finding_codes
    return d


def _clean(v: Any) -> Any:
    """JSON-safe: NaN/inf -> None, tuples -> lists."""
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _usd(x: float | None) -> str:
    if x is None:
        return "n/a"
    if 0 < x < 0.0001:
        return "<$0.0001"
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def _markdown(rep: Report, max_items: int, max_runs: int) -> str:
    o, s, c = rep.overview, rep.source, rep.cost
    L: list[str] = ["# Agent Report Card", ""]
    L += [f"**File format:** {s.get('format_label')} (confidence {s.get('confidence', 0):.2f}) -- "
          f"{s.get('n_runs')} runs, {s.get('n_events')} events"
          + (f", {s.get('skipped_lines')} unreadable lines skipped" if s.get("skipped_lines") else "") + ".",
          f"**Policy:** {rep.policy['name']} ({rep.policy['origin']}).  "
          f"**Success score:** {'trained model' if rep.model_info['method'] == 'model' else 'rule-of-thumb heuristic'}.",
          ""]
    L += ["## Headline", "",
          f"- Runs: **{o.get('n_runs')}**; agent said \"done\" in **{_pct(o.get('claimed_success_rate'))}**.",
          f"- Mean predicted success: **{_pct(o.get('mean_success_probability'))}**"
          + (f"; verified success: **{_pct(o.get('actual_success_rate'))}** ({o.get('n_with_outcome')} runs with outcomes)."
             if o.get("actual_success_rate") is not None else " (no ground-truth outcomes in this file)."),
          f"- Runs with a critical finding: **{o.get('runs_with_critical')}**; "
          f"said done although something failed / never happened: **{o.get('false_claim_runs')}**.",
          f"- Cost: **{_usd(c.get('total_usd'))}** total, {_usd(c.get('per_run_usd'))} per run, "
          + (f"**{_usd(c.get('per_verified_success_usd'))} per verified success**."
             if c.get("per_verified_success_usd") is not None
             else f"{_usd(c.get('per_expected_success_usd'))} per expected success.")
          + (" (estimated)" if c.get("estimated") else ""), ""]
    if rep.action_items:
        L += ["## Fix these first", ""]
        for i, a in enumerate(rep.action_items[:max_items], 1):
            L += [f"### {i}. [{a.priority}] {a.title}", "", f"- **Why:** {a.why}", f"- **Fix:** {a.fix}",
                  f"- **Impact:** {a.expected_impact}",
                  f"- **Evidence runs:** {', '.join(a.evidence_runs[:6])}" + (" ..." if len(a.evidence_runs) > 6 else ""), ""]
    if rep.accuracy:
        A = rep.accuracy
        cm = A["confusion_at_0_5"]
        L += ["## How accurate is the success score on this file?", "",
              f"{A['n']} runs have a verified outcome (base rate {_pct(A['base_rate'])}).", "",
              "| AUC | Brier | Brier if always guessing the base rate | Accuracy at 0.5 |", "|---|---|---|---|",
              f"| {A['auc']:.3f} | {A['brier']:.3f} | {A['brier_baseline']:.3f} | {_pct(A['accuracy_at_0_5'])} |"
              if A["auc"] is not None else f"| n/a | {A['brier']:.3f} | {A['brier_baseline']:.3f} | {_pct(A['accuracy_at_0_5'])} |",
              "", f"Confusion at 0.5: {cm['tp']} true successes, {cm['tn']} true failures, "
              f"{cm['fp']} failures predicted as successes, {cm['fn']} successes predicted as failures.", ""]
        cb = A["claimed_but_failed"]
        if cb["n_claimed"]:
            L += [f"Of {cb['n_claimed']} runs where the agent said \"done\", **{cb['n_claimed_but_failed']}** actually failed; "
                  f"{cb['n_flagged_by_findings']} of those were flagged by a critical/high finding.", ""]
    if rep.ladder:
        L += ["## Autonomy Ladder", "", "| Task | Runs | Success | Rung | Why |", "|---|---|---|---|---|"]
        for row in rep.ladder:
            L.append(f"| {row.get('task_type')} | {row['n_runs']} | {_pct(row['success_used'])} ({row['success_source']}) "
                     f"| {row['rung']} | {'; '.join(row['reasons'])} |")
        L.append("")
    worst = sorted(rep.runs, key=lambda r: (r.success_probability if r.success_probability is not None else 1))[:max_runs]
    if worst:
        L += ["## Runs most likely to have failed", "", "| Run | Task | P(success) | Outcome | Findings |", "|---|---|---|---|---|"]
        for r in worst:
            oc = "-" if r.outcome is None else ("success" if r.outcome else "FAILED")
            L.append(f"| {r.run_id} | {r.task_type or '-'} | {_pct(r.success_probability)} | {oc} | {', '.join(r.finding_codes) or '-'} |")
        L.append("")
    if rep.notes or s.get("warnings"):
        L += ["## Notes", ""] + [f"- {n}" for n in rep.notes + list(s.get("warnings") or [])] + [""]
    return "\n".join(L)
