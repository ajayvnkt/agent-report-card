"""Did the agent tell the user the task was done? -- one keyword heuristic for the whole package.

Used by the adapters' normalisation (``schema.finalize_events``) when a final answer has no
``claimed_success`` field, and by the analyzer. Rules:

* hand-off phrases ("escalated", "handed over", "a colleague will") -> NOT claimed;
* failure phrases ("could not", "unable to", "failed", "declined", "error", ...) count against;
  apologies alone ("Sorry about the delay") do not -- they are politeness, not failure;
* success words ("done", "issued", "has been", "rebooked", "resent", ...) count for, unless
  negated within the 4 preceding words ("has not been issued", "could not be processed");
* claimed = at least one un-negated success word AND more success than failure phrases.
"""
from __future__ import annotations

import re


_HANDOFF = re.compile(
    r"\bescalat\w*|\bhand(ed|ing)?\s+(it\s+|this\s+|the\s+case\s+|you\s+)?(over|off)\b|"
    r"\btransferr?(ed|ing)?\s+(you\s+|it\s+|this\s+)?to\s+(a\s+|the\s+|our\s+)?(human|colleague|specialist|agent|team|supervisor)|"
    r"\b(a|our)\s+(human|colleague|specialist|supervisor|team\s+member)\s+will\b|\bsomeone\s+will\b|"
    r"\bpending\s+(review|approval)\b|\bawaiting\s+(review|approval)\b|\brequires?\s+(manual|human)\b",
    re.IGNORECASE)
_FAIL = re.compile(
    r"\bcould\s*n[o']t\b|\bcouldn't\b|\bcan(?:'t|not)\b|\bunable\s+to\b|\bfail(ed|ure|s)?\b|\b(was|were)\s*n[o']t\s+able\b|"
    r"\bwasn't\s+able\b|\bnot\s+(possible|permitted|allowed|eligible)\b|\bdeclined\b|\brefused\b|\bdenied\b|"
    r"\brejected\b|\bunsuccessful\b|\berror\b|\bdid\s*n[o']t\s+go\s+through\b",
    re.IGNORECASE)
_SUCCESS = re.compile(
    r"\b(done|completed?|successful(ly)?|succeeded|issued|scheduled|cancel+ed|refunded|updated|processed|"
    r"waived|applied|confirmed|resolved|booked|raised|sent|submitted|verified|approved|arranged|set\s+up|"
    r"all\s+set|taken\s+care\s+of|has\s+been|have\s+been|went\s+through|re-?sent|e-?mailed|re-?booked|"
    r"created|changed|added|reset|fixed|charged|paid|placed|closed|handled|sorted\s+out)\b",
    re.IGNORECASE)
_NEGATOR = re.compile(r"\b(not|never|no|nothing|none|neither|nor|unable\s+to|failed\s+to|without)\b|n't\b",
                      re.IGNORECASE)


def infer_claimed_success(text: str | None) -> tuple[bool, str]:
    """Keyword heuristic (see module docstring). Returns ``(claimed, reason)``."""
    if not text or not str(text).strip():
        return False, "empty final answer"
    t = str(text)
    if _HANDOFF.search(t):
        return False, f"hand-off phrase: '{_HANDOFF.search(t).group(0)}'"
    fails = [m.group(0) for m in _FAIL.finditer(t)]
    pos: list[str] = []
    for m in _SUCCESS.finditer(t):
        window = " ".join(re.findall(r"[\w']+", t[max(0, m.start() - 60):m.start()])[-4:])
        if _NEGATOR.search(window):
            fails.append(f"not {m.group(0)}")
        else:
            pos.append(m.group(0))
    if pos and len(pos) > len(fails):
        return True, f"success words {pos[:3]}" + (f" outweigh failure words {fails[:3]}" if fails else "")
    if fails:
        return False, f"failure words {fails[:3]}"
    return False, "no success words"


def claimed_success_or_none(text: str | None) -> bool | None:
    """Like :func:`infer_claimed_success` but ``None`` when the text gives no signal either way
    (empty, or neither success nor failure/hand-off words, e.g. "Your balance is 42")."""
    claimed, reason = infer_claimed_success(text)
    if reason in ("empty final answer", "no success words"):
        return None
    return claimed
