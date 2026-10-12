"""Success probability for each run -- no ground truth needed at prediction time.

Two scorers, same output shape (``Prediction``):

* **Trained model** -- L2 logistic regression on GENERIC features any agent log has
  (``FEATURES``). Trained with :func:`train` on runs that carry a ground-truth outcome,
  validated with GROUPED cross-validation (repeats of one task stay in one fold),
  calibrated (Platt or isotonic) on the out-of-fold predictions, and saved as plain
  JSON. :func:`predict` is pure Python: no numpy / scikit-learn needed at runtime.
* **Heuristic** -- transparent rule-of-thumb used when no model file exists
  (:func:`heuristic_score`). Starts at 85% and multiplies by a penalty per problem.

Each prediction lists its top contributing factors in plain English.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

MODEL_VERSION = 1
DEFAULT_MODEL_PATH = Path(__file__).with_name("models") / "success_model.json"

FEATURES: list[str] = [
    "n_steps", "n_tool_calls", "n_distinct_tools",
    "n_errors", "n_refused", "n_timeouts", "n_unanswered", "fail_rate",
    "last_tool_failed", "failure_after_last_success", "n_unrecovered_failures",
    "n_retries", "escalated", "has_final_answer", "claimed_success",
    "n_claimed_not_done", "false_success",
    "n_critical", "n_high", "n_medium", "n_low",
    "injection_present", "escalation_text_present", "n_money_calls_ok",
]

# Signals that mean the same thing in any domain ("a tool failed and was never fixed", "the agent
# said done"); drops raw counts whose scale depends on the agent's task (steps, tools, money calls).
GENERIC_FEATURES: list[str] = [
    "fail_rate", "last_tool_failed", "failure_after_last_success", "n_unrecovered_failures",
    "n_unanswered", "n_retries", "escalated", "has_final_answer", "claimed_success",
    "n_claimed_not_done", "false_success", "n_critical", "n_high", "n_medium", "n_low",
    "injection_present", "escalation_text_present",
]

# Direction we know in advance: more failures can never make success MORE likely. With correlated
# features a free fit can still give one of them the "wrong" sign (e.g. fail_rate > 0 because
# n_unrecovered_failures already took the blame), which would produce nonsense explanations like
# "share of failed tool calls: 50% -- raises the chance of success". train(signs=...) drops such
# features (coefficient 0) and refits until every constrained coefficient has its expected sign.
EXPECTED_SIGNS: dict[str, int] = {
    "n_errors": -1, "n_refused": -1, "n_timeouts": -1, "n_unanswered": -1, "fail_rate": -1,
    "last_tool_failed": -1, "failure_after_last_success": -1, "n_unrecovered_failures": -1,
    "n_retries": -1, "n_claimed_not_done": -1, "false_success": -1,
    "n_critical": -1, "n_high": -1, "n_medium": -1, "has_final_answer": +1,
}

FEATURE_LABELS: dict[str, str] = {
    "n_steps": "number of agent steps",
    "n_tool_calls": "number of tool calls",
    "n_distinct_tools": "number of different tools used",
    "n_errors": "tool errors",
    "n_refused": "tool calls refused",
    "n_timeouts": "tool time-outs",
    "n_unanswered": "tool calls with no result",
    "fail_rate": "share of tool calls that failed",
    "last_tool_failed": "the last tool call failed",
    "failure_after_last_success": "a failure came after the last success",
    "n_unrecovered_failures": "failures never fixed",
    "n_retries": "re-tries after a failure",
    "escalated": "handed over to a human",
    "has_final_answer": "gave a final answer",
    "claimed_success": "agent said the task was done",
    "n_claimed_not_done": "actions claimed but not done",
    "false_success": "said done after an unfixed failure",
    "n_critical": "critical findings",
    "n_high": "high-severity findings",
    "n_medium": "medium-severity findings",
    "n_low": "low-severity findings",
    "injection_present": "instruction-like text in the input",
    "escalation_text_present": "input needed a human (e.g. fraud, legal)",
    "n_money_calls_ok": "money actions that went through",
}

BINARY = {"last_tool_failed", "failure_after_last_success", "escalated", "has_final_answer",
          "claimed_success", "false_success", "injection_present", "escalation_text_present"}


# --------------------------------------------------------------------------- output type

@dataclass
class Prediction:
    probability: float
    method: str                      # "model" | "heuristic"
    raw_probability: float | None = None
    factors: list[dict] = field(default_factory=list)   # {feature,label,value,effect,text}

    def to_dict(self) -> dict:
        return asdict(self)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _fmt_value(feat: str, v: float) -> str:
    if feat in BINARY:
        return "yes" if v >= 0.5 else "no"
    if feat == "fail_rate":
        return f"{v:.0%}"
    return f"{v:g}"


def features_of(run: Any) -> dict[str, float]:
    """Feature dict from a RunSummary, a dict of features, or a dict with 'features'."""
    if hasattr(run, "features") and callable(run.features):
        return run.features()
    if isinstance(run, dict):
        if isinstance(run.get("features"), dict):
            return {k: float(v) for k, v in run["features"].items()}
        return {k: float(run[k]) for k in FEATURES if k in run and run[k] is not None}
    raise TypeError(f"cannot get features from {type(run).__name__}")


def outcome_of(run: Any) -> int | None:
    v = getattr(run, "outcome", None) if not isinstance(run, dict) else run.get("outcome", run.get("verified_success"))
    if isinstance(v, dict):
        v = v.get("verified_success")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return int(bool(int(float(v))))


# --------------------------------------------------------------------------- heuristic

HEURISTIC_BASE = 0.85
HEURISTIC_PENALTIES = {               # multiplicative: p *= (1 - penalty) per occurrence
    "n_critical": 0.45, "n_high": 0.25, "n_medium": 0.10, "n_low": 0.02,
    "last_tool_failed": 0.20, "n_unanswered": 0.15,
}
HEURISTIC_RULES_TEXT = [
    "Start at 85%.",
    "Each critical finding x0.55, each high x0.75, each medium x0.90, each low x0.98.",
    "Last tool call failed x0.80; each tool call without a result x0.85.",
    "No final answer x0.60; share of failed tool calls f: x(1 - 0.5*f).",
    "Clamp to 2%..98%.",
]


def heuristic_score(features: dict[str, float]) -> Prediction:
    """Transparent fallback score (rules in ``HEURISTIC_RULES_TEXT``)."""
    p = HEURISTIC_BASE
    factors = []
    for feat, pen in HEURISTIC_PENALTIES.items():
        n = int(round(features.get(feat, 0)))
        if n > 0:
            mult = (1 - pen) ** n
            p *= mult
            factors.append(_factor(feat, features.get(feat, 0), math.log(mult)))
    if features.get("has_final_answer", 1) < 0.5:
        p *= 0.60
        factors.append(_factor("has_final_answer", 0, math.log(0.60)))
    fr = features.get("fail_rate", 0)
    if fr > 0:
        p *= (1 - 0.5 * fr)
        factors.append(_factor("fail_rate", fr, math.log(1 - 0.5 * fr)))
    p = min(max(p, 0.02), 0.98)
    factors.sort(key=lambda f: f["effect"])
    if not factors:
        factors.append({"feature": None, "label": "no problems found", "value": None, "effect": 0.0,
                        "text": "No failures or policy findings -- starts at 85%."})
    return Prediction(probability=p, method="heuristic", raw_probability=p, factors=factors[:5])


def _factor(feat: str, value: float, effect: float) -> dict:
    label = FEATURE_LABELS.get(feat, feat)
    direction = "raises" if effect > 0 else "lowers"
    return {"feature": feat, "label": label, "value": value, "effect": round(effect, 4),
            "text": f"{label[0].upper()}{label[1:]}: {_fmt_value(feat, value)} -- {direction} the chance of success"}


# --------------------------------------------------------------------------- model predict (pure python)

def predict(model: dict | None, features: dict[str, float], top_k: int = 4) -> Prediction:
    """Probability of success for one run. ``model=None`` -> heuristic."""
    if not model:
        return heuristic_score(features)
    names = model["features"]
    z = model["intercept"]
    contribs = []
    for i, f in enumerate(names):
        x = float(features.get(f, 0.0))
        xs = (x - model["means"][i]) / (model["stds"][i] or 1.0)
        c = model["coef"][i] * xs
        z += c
        contribs.append((f, x, c))
    raw = _sigmoid(z)
    p = apply_calibration(model.get("calibration"), raw, z)
    contribs.sort(key=lambda t: -abs(t[2]))
    factors = [_factor(f, x, c) for f, x, c in contribs[:top_k] if abs(c) > 1e-6]
    return Prediction(probability=p, method="model", raw_probability=raw, factors=factors)


def apply_calibration(cal: dict | None, raw: float, z: float | None = None) -> float:
    if not cal or cal.get("method") in (None, "none"):
        return raw
    if cal["method"] == "platt":
        zz = z if z is not None else _logit(raw)
        return _sigmoid(cal["a"] * zz + cal["b"])
    if cal["method"] == "isotonic":
        xs, ys = cal["x"], cal["y"]
        if not xs:
            return raw
        if raw <= xs[0]:
            return ys[0]
        if raw >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if raw <= xs[i]:
                x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                return y0 if x1 == x0 else y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
    return raw


def predict_runs(runs: Sequence[Any], model: dict | None = None) -> list[Prediction]:
    """Predict every run; sets ``run.success_probability`` / ``run.prediction`` when possible."""
    out = []
    for r in runs:
        pr = predict(model, features_of(r))
        out.append(pr)
        if hasattr(r, "success_probability"):
            r.success_probability = pr.probability
            r.prediction = pr.to_dict()
    return out


# --------------------------------------------------------------------------- save / load

REQUIRED_KEYS = ("features", "means", "stds", "coef", "intercept")


def save_model(model: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return p


def load_model(source: str | Path | dict | bytes) -> dict:
    """Load and validate a model JSON (path, JSON text/bytes, or dict)."""
    if isinstance(source, dict):
        m = source
    else:
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        s = str(source)
        if s.lstrip().startswith("{"):
            m = json.loads(s)
        else:
            m = json.loads(Path(s).read_text(encoding="utf-8"))
    miss = [k for k in REQUIRED_KEYS if k not in m]
    if miss:
        raise ValueError(f"model file is missing keys: {', '.join(miss)}")
    n = len(m["features"])
    if not (len(m["means"]) == len(m["stds"]) == len(m["coef"]) == n):
        raise ValueError("model file: features/means/stds/coef lengths differ")
    return m


def resolve_model(model: Any = "default") -> dict | None:
    """'default' -> bundled file if present else None; None -> None; dict/path -> loaded."""
    if model is None:
        return None
    if isinstance(model, str) and model == "default":
        return load_model(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None
    return load_model(model)


# --------------------------------------------------------------------------- metrics

def auc(y: Sequence[int], p: Sequence[float]) -> float | None:
    """ROC AUC via rank statistic (ties get half credit). None if one class only."""
    pos = [pi for yi, pi in zip(y, p) if yi == 1]
    neg = [pi for yi, pi in zip(y, p) if yi == 0]
    if not pos or not neg:
        return None
    pairs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # average ranks
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rsum = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)


def calibration_bins(y: Sequence[int], p: Sequence[float], n_bins: int = 5) -> list[dict]:
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, pi in enumerate(p) if (lo <= pi < hi) or (b == n_bins - 1 and pi == 1.0)]
        bins.append({"bin": f"{lo:.0%}-{hi:.0%}", "lo": lo, "hi": hi, "n": len(idx),
                     "mean_predicted": (sum(p[i] for i in idx) / len(idx)) if idx else None,
                     "actual_rate": (sum(y[i] for i in idx) / len(idx)) if idx else None})
    return bins


def evaluate(probs: Sequence[float], y: Sequence[int], n_bins: int = 5) -> dict:
    """How well predicted probabilities match ground truth."""
    y = [int(v) for v in y]
    probs = [float(v) for v in probs]
    n = len(y)
    if not n:
        return {"n": 0}
    brier = sum((pi - yi) ** 2 for pi, yi in zip(probs, y)) / n
    base = sum(y) / n
    pred = [int(pi >= 0.5) for pi in probs]
    tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
    return {
        "n": n, "base_rate": base, "auc": auc(y, probs), "brier": brier,
        "brier_baseline": base * (1 - base),   # always predicting the base rate
        "accuracy_at_0_5": (tp + tn) / n,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "mean_predicted": sum(probs) / n,
        "calibration_bins": calibration_bins(y, probs, n_bins),
    }


# --------------------------------------------------------------------------- training (numpy)

def _fit_logreg(X, y, l2: float = 1.0, iters: int = 100):
    """L2-penalised logistic regression via Newton/IRLS (intercept unpenalised)."""
    import numpy as np
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    pen = np.full(d + 1, l2)
    pen[0] = 0.0
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) + pen * w
        W = p * (1 - p)
        H = (Xb * W[:, None]).T @ Xb + np.diag(pen) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w[1:], float(w[0])


def _fit_signed(X, y, l2: float, names: Sequence[str], signs: dict[str, int] | None):
    """_fit_logreg with sign constraints: features whose coefficient contradicts ``signs`` get
    coefficient 0 and the rest are refitted, until all constrained coefficients agree."""
    import numpy as np
    active = list(range(X.shape[1]))
    while True:
        coef_a, b = _fit_logreg(X[:, active], y, l2)
        coef = np.zeros(X.shape[1])
        coef[active] = coef_a
        if not signs:
            return coef, b
        bad = [i for i in active if signs.get(names[i]) and np.sign(coef[i]) not in (0, signs[names[i]])]
        if not bad:
            return coef, b
        worst = max(bad, key=lambda i: abs(coef[i]))     # drop one at a time, the most wrong first
        active.remove(worst)
        if not active:
            return np.zeros(X.shape[1]), float(np.log(y.mean() / (1 - y.mean())))


def _standardise(X):
    import numpy as np
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def _fit_platt(z, y) -> dict:
    import numpy as np
    coef, b = _fit_logreg(np.asarray(z, float).reshape(-1, 1), np.asarray(y, float), l2=1e-3)
    return {"method": "platt", "a": float(coef[0]), "b": float(b)}


def _fit_isotonic(p, y) -> dict:
    """Pool-adjacent-violators; returns a monotone lookup table."""
    pts = sorted(zip(p, y))
    blocks = [[pi, float(yi), 1.0] for pi, yi in pts]   # x, mean y, weight
    merged: list[list[float]] = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][1] > merged[-1][1]:
            x2, y2, w2 = merged.pop()
            x1, y1, w1 = merged.pop()
            w = w1 + w2
            merged.append([(x1 * w1 + x2 * w2) / w, (y1 * w1 + y2 * w2) / w, w])
    return {"method": "isotonic", "x": [round(m[0], 6) for m in merged],
            "y": [round(min(max(m[1], 0.01), 0.99), 6) for m in merged]}


def grouped_folds(groups: Sequence[str], n_splits: int = 5, seed: int = 0) -> list[list[int]]:
    """Assign whole groups to folds (largest groups first, to the smallest fold)."""
    by: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(str(g), []).append(i)
    keys = sorted(by)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: -len(by[k]))
    k = max(2, min(n_splits, len(keys)))
    folds: list[list[int]] = [[] for _ in range(k)]
    for key in keys:
        min(folds, key=len).extend(by[key])
    return [f for f in folds if f]


@dataclass
class TrainResult:
    model: dict          # JSON-serialisable model (save with save_model)
    cv: dict             # grouped-CV metrics on out-of-fold, calibrated predictions
    heuristic_cv: dict   # the fallback heuristic scored on the same runs, for comparison

    def to_dict(self) -> dict:
        return asdict(self)


def train(runs_with_outcome: Iterable[Any], groups: Sequence[str] | None = None, *,
          features: Sequence[str] = FEATURES, l2: float = 3.0, n_splits: int = 5,
          calibration: str = "platt", seed: int = 0, provenance: dict | None = None,
          signs: dict[str, int] | None = None) -> TrainResult:
    """Fit the success model.

    ``runs_with_outcome``: RunSummary objects (``.outcome`` set) or dicts with feature
    keys (or ``features``) plus ``outcome``/``verified_success``. Runs without an
    outcome are skipped. ``groups``: case/task id per run (default: ``run.group`` or
    run_id) -- all runs of a group are held out together in cross-validation.
    """
    import numpy as np

    rows, ys, gs = [], [], []
    runs = list(runs_with_outcome)
    if groups is not None and len(groups) != len(runs):
        raise ValueError("groups must have one entry per run")
    for i, r in enumerate(runs):
        y = outcome_of(r)
        if y is None:
            continue
        f = features_of(r)
        rows.append([float(f.get(k, 0.0)) for k in features])
        ys.append(y)
        if groups is not None:
            gs.append(str(groups[i]))
        else:
            g = getattr(r, "group", None) if not isinstance(r, dict) else r.get("group")
            rid = getattr(r, "run_id", None) if not isinstance(r, dict) else r.get("run_id")
            gs.append(str(g or rid or i))
    if len(ys) < 4 or len(set(ys)) < 2:
        raise ValueError("need at least 4 runs with both successful and failed outcomes to train")
    X, Y = np.asarray(rows, float), np.asarray(ys, float)

    # grouped CV -> out-of-fold raw scores
    folds = grouped_folds(gs, n_splits, seed) if len(set(gs)) >= 2 else []
    oof_z = np.full(len(Y), np.nan)
    for test in folds:
        tr = np.setdiff1d(np.arange(len(Y)), test)
        if len(set(Y[tr])) < 2:
            continue
        mu, sd = _standardise(X[tr])
        coef, b = _fit_signed((X[tr] - mu) / sd, Y[tr], l2, list(features), signs)
        oof_z[test] = ((X[test] - mu) / sd) @ coef + b
    have = ~np.isnan(oof_z)
    cal: dict = {"method": "none"}
    cv: dict = {"n_folds": len(folds), "n_groups": len(set(gs))}
    if have.sum() >= 4 and len(set(Y[have])) == 2:
        zz, yy = oof_z[have], Y[have]
        raw = 1 / (1 + np.exp(-zz))
        if calibration == "isotonic":
            cal = _fit_isotonic(raw.tolist(), yy.tolist())
        elif calibration == "platt":
            cal = _fit_platt(zz, yy)
        calp = [apply_calibration(cal, float(r_), float(z_)) for r_, z_ in zip(raw, zz)]
        cv.update(evaluate(calp, yy.astype(int).tolist()))
        cv["auc_uncalibrated"] = auc(yy.astype(int).tolist(), raw.tolist())
        cv["note"] = ("calibration is fitted on these same out-of-fold scores, so calibrated "
                      "Brier/bins are slightly optimistic; AUC is unaffected by Platt scaling")
    # final model on all data
    mu, sd = _standardise(X)
    coef, b = _fit_signed((X - mu) / sd, Y, l2, list(features), signs)
    heur = [heuristic_score(dict(zip(features, row))).probability for row in X.tolist()]
    model = {
        "version": MODEL_VERSION,
        "kind": "logistic_regression_l2",
        "features": list(features),
        "means": [float(v) for v in mu], "stds": [float(v) for v in sd],
        "coef": [float(v) for v in coef], "intercept": float(b),
        "l2": l2, "calibration": cal,
        "sign_constraints": dict(signs) if signs else None,
        "dropped_by_sign": [f for f, c in zip(features, coef) if signs and signs.get(f) and abs(c) < 1e-12],
        "provenance": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_runs": int(len(Y)), "n_groups": len(set(gs)), "base_rate": float(Y.mean()),
            **(provenance or {}),
        },
        "metrics": {k: v for k, v in cv.items() if k != "calibration_bins"},
        "calibration_bins": cv.get("calibration_bins"),
    }
    return TrainResult(model=model, cv=cv, heuristic_cv=evaluate(heur, Y.astype(int).tolist()))


def describe_model(model: dict | None) -> list[dict]:
    """Coefficient table in plain English (largest effects first)."""
    if not model:
        return [{"rule": r} for r in HEURISTIC_RULES_TEXT]
    rows = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "coef_per_std": c,
             "direction": "more likely to succeed" if c > 0 else "less likely to succeed"}
            for f, c in zip(model["features"], model["coef"])]
    return sorted(rows, key=lambda r: -abs(r["coef_per_std"]))


# --------------------------------------------------------------------------- CLI

def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    import argparse
    from .analyze import summarize_runs
    from .detect import load_any
    ap = argparse.ArgumentParser(description="Train the success model on logs (any supported format) with outcomes.")
    ap.add_argument("cmd", choices=["train"], nargs="?", default="train")
    ap.add_argument("logs", help="log file with ground-truth outcomes (run_end.outcome.verified_success)")
    ap.add_argument("--policy")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--features", default="generic", choices=["generic", "full"])
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--calibration", default="platt", choices=["platt", "isotonic", "none"])
    ap.add_argument("--source", default="", help="provenance note, e.g. dataset name + licence")
    a = ap.parse_args(argv)
    events, _ = load_any(a.logs)
    runs = summarize_runs(events, a.policy)
    res = train(runs, features=GENERIC_FEATURES if a.features == "generic" else FEATURES, l2=a.l2,
                calibration=a.calibration, signs=EXPECTED_SIGNS,
                provenance={"source_file": Path(a.logs).name, "source": a.source,
                            "trained_on_agents": sorted({r.agent for r in runs if r.agent})})
    save_model(res.model, a.out)
    print(json.dumps({"saved": a.out, "cv": {k: v for k, v in res.cv.items() if k != "calibration_bins"},
                      "heuristic_auc": res.heuristic_cv.get("auc")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
