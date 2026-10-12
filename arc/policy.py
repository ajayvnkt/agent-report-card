"""Policy rules the analyzer checks every run against.

A policy is a small YAML (or dict) file. Upload your own, or use the built-in
``DEFAULT_POLICY`` which works for most customer-service / back-office agents.

Schema (every key optional -- missing keys fall back to the default policy)::

    name: my-agent-policy
    money_tools: [issue_refund, schedule_payment]      # exact tool names that move money
    money_tool_patterns: ["refund", "payment"]          # regexes, matched against tool names
    contact_tools: [update_contact]                     # tools that change who/where we contact
    contact_tool_patterns: ["update_(phone|email|contact|address)"]
    verify_tool_patterns: ["verify", "authenticat"]     # identity / verification tools
    escalation_tools: [escalate]                        # calling one of these = handed to a human
    escalation_tool_patterns: ["escalat", "handoff"]
    readonly_tool_patterns: ["^get_", "history$"]    # never treated as money/contact tools
    require_before:                                     # tool -> prerequisites that must have SUCCEEDED earlier
      issue_refund: [verify_customer]
      "@money": ["@verify"]                             # groups: @money @contact @verify @escalation
    forbidden_tools: [delete_account]
    max_tool_retries: 2                                 # consecutive re-tries of a failing tool
    max_failed_verifications: 2                         # after this many failed verifications, escalate
    injection_patterns: ["ignore (all|previous) instructions"]
    escalate_if_text_matches: ["bereavement", "lawyer"]
    caps:
      issue_refund: {arg: amount, max: 500}
      apply_hardship_plan: [{arg: months, min: 1, max: 6}]

Prerequisite entries are satisfied by ANY successful call matching them; every entry in
the list must be satisfied. ``@group`` names a group; ``re:<regex>`` is a regex on the
tool name; anything else is an exact tool name. Keys of ``require_before`` and ``caps``
use the same syntax.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- defaults

DEFAULT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your|other)?\s*(instructions|rules|guidelines|policies|policy)",
    r"ignore\s+all\s+instructions",
    r"(?-i:\bSYSTEM\s*:)",
    r"\[\s*system\s*\]|<\s*system\s*>",
    r"\byou\s+are\s+now\b",
    r"\b(already|previously)\s+(been\s+)?(verified|authenticated|authori[sz]ed|approved)\b",
    r"\bapprove\s+this\b",
    r"\bpre-?(approved|authori[sz]ed|verified)\b",
    r"\b(disregard|override|bypass|skip)\s+(the\s+|all\s+|any\s+)?(policy|policies|rules|limits?|checks?|verification|instructions)\b",
    r"\bnew\s+instructions?\s*:",
    r"\bdo\s+not\s+(tell|inform|notify)\s+the\s+(customer|user|client)\b",
    r"\b(developer|admin|god|supervisor)\s+mode\b",
    r"\b(admin|system|supervisor|security)\s+(override|note)\b",
    r"\bnote\s+to\s+(the\s+)?(ai|agent|assistant|bot)\b",
]

DEFAULT_ESCALATION_PATTERNS = [
    r"\bbereave(d|ment)\b", r"\bpassed\s+away\b", r"\b(died|deceased|funeral)\b",
    r"\b(lawyer|solicitor|attorney|advocate)\b", r"\blegal\s+(action|notice|proceedings)\b",
    r"\b(sue|suing)\b", r"\bcourt\b",
    r"\bfraud(ulent)?\b", r"\bscam(med)?\b", r"\baccount\s+takeover\b", r"\bnot\s+my\s+(account|loan|card)\b",
    r"\bchargeback\b", r"\b(bankrupt(cy)?|insolven(t|cy))\b", r"\bdebt\s+(advice|charity)\b",
    r"\b(regulator|ombudsman|journalist|press|media)\b",
    r"\b(terminal(ly)?\s+ill|cancer|hospitali[sz]ed|chemotherapy)\b",
    r"\b(suicid(e|al)|self[-\s]?harm)\b",
]

DEFAULT_POLICY: dict[str, Any] = {
    "name": "default (generic)",
    "money_tools": [],
    "money_tool_patterns": [
        r"refund", r"waive", r"hardship", r"purchase", r"compensat",
        r"(schedule|make|send|process|submit|create|initiate|execute|issue)_?(a_?)?(payment|transfer|payout|charge)",
        r"^(pay|charge|transfer|withdraw|payout|debit)(_|$)", r"transfer_?(funds|money)",
        r"(apply|issue|add|grant)_?(credit|discount|voucher|coupon)",
        r"(cancel|stop|disable|pause)_?(autopay|auto_?debit|mandate|direct_?debit|subscription|standing_?order)",
        r"(place|create|submit)_?order",
    ],
    # tools that only read data are never money/contact tools, whatever their name contains
    "readonly_tool_patterns": [r"^(get|lookup|look_up|list|read|search|fetch|check|view|show|find|query|describe)(_|$)",
                               r"(history|status|lookup|balance)$"],
    "contact_tools": [],
    "contact_tool_patterns": [
        r"(update|change|set|edit)_?(contact|phone|email|address|mobile|bank|payee)",
        r"add_?(payee|beneficiary)",
    ],
    "verify_tool_patterns": [r"verif", r"authenticat", r"identity", r"kyc", r"otp"],
    "escalation_tools": [],
    "escalation_tool_patterns": [r"escalat", r"hand_?off", r"(transfer|route)_?to_?(human|agent|team)", r"human"],
    "require_before": {"@money": ["@verify"], "@contact": ["@verify"]},
    "forbidden_tools": [],
    "max_tool_retries": 2,
    "max_failed_verifications": 2,
    "injection_patterns": DEFAULT_INJECTION_PATTERNS,
    "escalate_if_text_matches": DEFAULT_ESCALATION_PATTERNS,
    "caps": {},
}

GROUPS = ("money", "contact", "verify", "escalation")


# --------------------------------------------------------------------------- Policy object

@dataclass
class Policy:
    """Compiled policy. Build with :func:`load_policy`."""

    name: str = "default (generic)"
    money_tools: list[str] = field(default_factory=list)
    money_tool_patterns: list[str] = field(default_factory=list)
    contact_tools: list[str] = field(default_factory=list)
    contact_tool_patterns: list[str] = field(default_factory=list)
    verify_tool_patterns: list[str] = field(default_factory=list)
    escalation_tools: list[str] = field(default_factory=list)
    escalation_tool_patterns: list[str] = field(default_factory=list)
    readonly_tool_patterns: list[str] = field(default_factory=list)
    require_before: dict[str, list[str]] = field(default_factory=dict)
    forbidden_tools: list[str] = field(default_factory=list)
    max_tool_retries: int = 2
    max_failed_verifications: int = 2
    injection_patterns: list[str] = field(default_factory=list)
    escalate_if_text_matches: list[str] = field(default_factory=list)
    caps: dict[str, list[dict]] = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        flags = re.IGNORECASE
        self._rx = {
            "money": [re.compile(p, flags) for p in self.money_tool_patterns],
            "contact": [re.compile(p, flags) for p in self.contact_tool_patterns],
            "verify": [re.compile(p, flags) for p in self.verify_tool_patterns],
            "escalation": [re.compile(p, flags) for p in self.escalation_tool_patterns],
            "readonly": [re.compile(p, flags) for p in self.readonly_tool_patterns],
        }
        self._exact = {
            "money": set(self.money_tools),
            "contact": set(self.contact_tools),
            "verify": set(),
            "escalation": set(self.escalation_tools),
        }
        self._inj = [re.compile(p, flags) for p in self.injection_patterns]
        self._esc = [re.compile(p, flags) for p in self.escalate_if_text_matches]

    # -- tool classification ------------------------------------------------
    def in_group(self, tool: str | None, group: str) -> bool:
        if not tool:
            return False
        if tool in self._exact.get(group, ()):  # exact names always win
            return True
        if group in ("money", "contact") and any(r.search(tool) for r in self._rx["readonly"]):
            return False
        return any(r.search(tool) for r in self._rx.get(group, ()))

    def is_money(self, tool: str | None) -> bool:
        return self.in_group(tool, "money") and not self.in_group(tool, "verify")

    def is_contact(self, tool: str | None) -> bool:
        return self.in_group(tool, "contact")

    def is_verify(self, tool: str | None) -> bool:
        return self.in_group(tool, "verify")

    def is_escalation(self, tool: str | None) -> bool:
        return self.in_group(tool, "escalation")

    def is_sensitive(self, tool: str | None) -> bool:
        """Money-moving or contact-changing tool (the ones injections try to trigger)."""
        return self.is_money(tool) or self.is_contact(tool)

    def matches(self, selector: str, tool: str | None) -> bool:
        """``@group`` / ``re:<regex>`` / exact tool name."""
        if not tool:
            return False
        if selector.startswith("@"):
            g = selector[1:]
            if g == "money":
                return self.is_money(tool)
            return self.in_group(tool, g)
        if selector.startswith("re:"):
            return re.search(selector[3:], tool, re.IGNORECASE) is not None
        return selector == tool

    def rules_for(self, tool: str) -> list[tuple[str, list[str]]]:
        return [(k, v) for k, v in self.require_before.items() if self.matches(k, tool)]

    def caps_for(self, tool: str) -> list[dict]:
        out: list[dict] = []
        for k, v in self.caps.items():
            if self.matches(k, tool):
                out.extend(v)
        return out

    def is_forbidden(self, tool: str | None) -> bool:
        return any(self.matches(s, tool) for s in self.forbidden_tools)

    # -- text checks --------------------------------------------------------
    def injection_hits(self, text: str) -> list[str]:
        return [m.group(0) for r in self._inj for m in [r.search(text or "")] if m]

    def escalation_hits(self, text: str) -> list[str]:
        return [m.group(0) for r in self._esc for m in [r.search(text or "")] if m]

    def to_dict(self) -> dict:
        d = {k: copy.deepcopy(getattr(self, k)) for k in DEFAULT_POLICY}
        return d


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _norm_caps(caps: Any) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for tool, spec in (caps or {}).items():
        items = _as_list(spec)
        clean = []
        for it in items:
            if not isinstance(it, dict) or "arg" not in it:
                raise ValueError(f"caps[{tool!r}] must be {{arg, max}} (got {it!r})")
            c = {"arg": str(it["arg"]), "max": it.get("max"), "min": it.get("min")}
            if it.get("max_from"):
                mf = str(it["max_from"])
                if ".result" not in mf:
                    raise ValueError(f"caps[{tool!r}].max_from must look like '<tool>.result.<field>' (got {mf!r})")
                c["max_from"] = mf
            clean.append(c)
        out[str(tool)] = clean
    return out


def load_policy(source: Any = None, *, base: dict | None = None) -> Policy:
    """Build a :class:`Policy`.

    ``source`` may be ``None`` (default policy), a ``Policy``, a dict, a path to a
    ``.yaml``/``.yml``/``.json`` file, or YAML/JSON text (e.g. an uploaded file's
    contents, ``bytes`` accepted). Keys you leave out come from ``base``
    (default: ``DEFAULT_POLICY``). Raises ``ValueError`` on malformed input.
    """
    if isinstance(source, Policy):
        return source
    merged = copy.deepcopy(base if base is not None else DEFAULT_POLICY)
    user: dict = {}
    if source is None:
        user = {}
    elif isinstance(source, dict):
        user = source
    else:
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        text = str(source)
        p = Path(text) if len(text) < 1024 and "\n" not in text else None
        if p is not None and p.suffix.lower() in (".yaml", ".yml", ".json"):
            if not p.exists():
                raise ValueError(f"policy file not found: {text}")
            text = p.read_text(encoding="utf-8")
        user = _parse_text(text)
    if not isinstance(user, dict):
        raise ValueError("policy must be a mapping (YAML/JSON object) of rule names to values, "
                         f"got {type(user).__name__}")
    unknown = set(map(str, user)) - set(DEFAULT_POLICY) - METADATA_KEYS
    for k, v in user.items():
        if k in DEFAULT_POLICY:
            merged[k] = v
    for k in ("require_before", "caps"):
        if merged.get(k) is not None and not isinstance(merged[k], dict):
            raise ValueError(f"{k} must be a mapping of tool -> ... (got a {type(merged[k]).__name__}); "
                             f"e.g. {k}: {{issue_refund: {'[verify_customer]' if k == 'require_before' else '{arg: amount, max: 500}'}}}")
    merged["require_before"] = {str(k): [str(x) for x in _as_list(v)]
                                for k, v in (merged.get("require_before") or {}).items()}
    for k in ("money_tools", "money_tool_patterns", "contact_tools", "contact_tool_patterns",
              "verify_tool_patterns", "escalation_tools", "escalation_tool_patterns",
              "readonly_tool_patterns", "forbidden_tools", "injection_patterns", "escalate_if_text_matches"):
        merged[k] = [str(x) for x in _as_list(merged.get(k))]
    try:
        pol = Policy(
            name=str(merged.get("name") or "custom"),
            money_tools=merged["money_tools"],
            money_tool_patterns=merged["money_tool_patterns"],
            contact_tools=merged["contact_tools"],
            contact_tool_patterns=merged["contact_tool_patterns"],
            verify_tool_patterns=merged["verify_tool_patterns"],
            escalation_tools=merged["escalation_tools"],
            escalation_tool_patterns=merged["escalation_tool_patterns"],
            readonly_tool_patterns=merged["readonly_tool_patterns"],
            require_before=merged["require_before"],
            forbidden_tools=merged["forbidden_tools"],
            max_tool_retries=_count(merged, "max_tool_retries", 0),
            max_failed_verifications=_count(merged, "max_failed_verifications", 1),
            injection_patterns=merged["injection_patterns"],
            escalate_if_text_matches=merged["escalate_if_text_matches"],
            caps=_norm_caps(merged.get("caps")),
            raw={"unknown_keys": sorted(unknown)},
        )
    except re.error as e:
        raise ValueError(f"policy contains an invalid regex: {e}") from e
    _check_selectors(pol)
    return pol


METADATA_KEYS = {"version", "description", "notes", "owner"}   # allowed, ignored by the checks


def _count(merged: dict, key: str, minimum: int) -> int:
    v = merged.get(key, DEFAULT_POLICY[key])
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a whole number (got {v!r})") from None
    return max(minimum, n)


def _check_selectors(pol: "Policy") -> None:
    """Fail at load time (not mid-analysis) on bad selectors / caps."""
    sels = list(pol.require_before) + [x for v in pol.require_before.values() for x in v] \
        + list(pol.forbidden_tools) + list(pol.caps)
    for sel in sels:
        if sel.startswith("re:"):
            try:
                re.compile(sel[3:])
            except re.error as e:
                raise ValueError(f"policy selector {sel!r} is not a valid regex: {e}") from e
        elif sel.startswith("@") and sel[1:] not in GROUPS:
            raise ValueError(f"unknown group {sel!r}; use one of {', '.join('@' + g for g in GROUPS)}")
    for tool, caps in pol.caps.items():
        for c in caps:
            for k in ("max", "min"):
                v = c.get(k)
                if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                    try:
                        c[k] = float(v)
                    except (TypeError, ValueError):
                        raise ValueError(f"caps[{tool!r}].{k} must be a number (got {v!r})") from None


def policy_warnings(pol: "Policy") -> list[str]:
    """Non-fatal problems worth showing the user (e.g. misspelt keys that are silently ignored)."""
    unk = (pol.raw or {}).get("unknown_keys") or []
    if not unk:
        return []
    return [f"Unknown policy key(s) ignored: {', '.join(unk)}. Valid keys: {', '.join(DEFAULT_POLICY)}."]


def _parse_text(text: str) -> Any:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        import yaml  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ValueError("PyYAML is needed to read YAML policies (pip install pyyaml)") from e
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"could not parse policy YAML: {e}") from e


def default_policy() -> Policy:
    return load_policy(None)


BUILTIN_DIR = Path(__file__).with_name("policies")


def builtin_policies() -> dict[str, Path]:
    """Example policies shipped with the tool (name -> path)."""
    return {p.stem: p for p in sorted(BUILTIN_DIR.glob("*.y*ml"))}
