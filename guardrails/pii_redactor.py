"""
ISDO Lab C9 — PII Redaction Middleware
Masks PII before any ticket data is sent to Claude.
Patterns covered: names (via spaCy NER, with a regex fallback when spaCy
isn't installed), login usernames (AD-style domain\\user and labeled
username/login fields), email addresses, employee IDs, IP addresses, and
phone numbers.

Usage:
    from guardrails.pii_redactor import redact, restore

    clean_text, mapping = redact(raw_text)
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)
"""

import re
import json
from datetime import datetime

# Try to import spaCy — graceful fallback if not installed
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("⚠  spaCy not available — using regex-only PII detection.")

# ── REGEX PATTERNS (whole match is masked) ────────────────────────────────────

PATTERNS = {
    "EMAIL":       r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
    "AD_LOGIN":    r'\b[A-Za-z][A-Za-z0-9\-]{1,20}\\[A-Za-z][A-Za-z0-9._\-]{2,20}\b',  # DOMAIN\jsmith
    "IP_ADDRESS":  r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "EMPLOYEE_ID": r'\b(?:EMP|ZEN|EMP-|ZEN-)\d{3,6}\b',
    "PHONE":       r'\b(?:\+91[\-\s]?)?\d{10}\b|\b\d{3}[\-\s]\d{3}[\-\s]\d{4}\b',
    "TICKET_REF":  r'\b(?:INC|REQ|CHG)\d{7}\b',   # keep ticket refs — not PII
}

# ── LABELED PATTERNS (only the captured value is masked, not the label) ───────
# Catches usernames/logins that only make sense next to a label, e.g.
# "Username: jsmith23", "login: j.doe", "user id: ZEN\\jsmith"

LABELED_PATTERNS = {
    "USERNAME": r'\b(?:username|user\s*id|user|login|logon|account\s*id)\s*[:\-]\s*([A-Za-z0-9._\-\\]{3,30})\b',
}

# ── NAME FALLBACK (used only when spaCy is unavailable) ───────────────────────
# Matches two-or-three-word Title Case sequences (optionally with an
# apostrophe, e.g. "D'Souza"). Not as accurate as NER, but means names
# aren't left completely unredacted when spaCy isn't installed.

_NAME_WORD = r"[A-Z][a-z]*(?:['’][A-Z][a-z]+)?"   # handles "John", "D'Souza", "O'Brien"
NAME_FALLBACK_PATTERN = rf"\b{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,2}}\b"

# Common false positives the name fallback would otherwise catch
# (product/system names, title-case phrases that aren't people)
NAME_FALLBACK_STOPWORDS = {
    "vpn", "sap", "erp", "it", "hr", "service desk", "password reset",
    "no pii", "contact", "ticket", "error", "access",
}

# Leading words that are often capitalized (start of sentence, a role/action
# word) but aren't part of the name that follows — e.g. "User John Smith"
# should redact to [NAME_1] = "John Smith", not "User John Smith".
_NAME_LEADING_STRIP = (
    "user", "contact", "contractor", "employee", "requester", "reporter",
    "assignee", "owner", "approver", "manager", "caller",
)

# ── AUDIT LOGGER ──────────────────────────────────────────────────────────────

audit_log = []

def _audit(action, detail):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "module": "PIIRedactor",
        "action": action,
        "detail": detail
    }
    audit_log.append(entry)
    return entry

# ── REDACTION FUNCTION ────────────────────────────────────────────────────────

def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns:
      - clean_text: text with PII replaced by tokens like [EMAIL_1], [NAME_1]
      - mapping: dict to restore original values later

    Example:
      clean, m = redact("Contact john.doe@corp.com or call 9876543210")
      # clean  = "Contact [EMAIL_1] or call [PHONE_1]"
      # m      = {"[EMAIL_1]": "john.doe@corp.com", "[PHONE_1]": "9876543210"}
    """
    mapping = {}
    counters = {}
    clean = text

    # Step 1: Named Entity Recognition (spaCy) — catches PERSON names
    if SPACY_AVAILABLE:
        doc = nlp(text)
        for ent in doc.ents:
            # Guard against a known spaCy false-positive: short ALL-CAPS acronyms
            # (PII, SLA, KB, VPN...) occasionally get tagged PERSON. Real names
            # in ticket text are essentially never written fully uppercase, so
            # this is a safe filter, not a real name being skipped.
            if ent.label_ == "PERSON" and not ent.text.isupper() and ent.text not in mapping.values():
                counters["NAME"] = counters.get("NAME", 0) + 1
                token = f"[NAME_{counters['NAME']}]"
                mapping[token] = ent.text
                clean = clean.replace(ent.text, token)
    else:
        # Step 1b: regex fallback for names when spaCy isn't installed.
        # Without this, names were being left completely unredacted.
        for match in re.finditer(NAME_FALLBACK_PATTERN, clean):
            matched = match.group(0)
            # Strip a leading role/action word that isn't part of the name
            # itself, e.g. "User John Smith" -> "John Smith"
            lowered = matched.lower()
            for lead in _NAME_LEADING_STRIP:
                if lowered.startswith(lead + " "):
                    matched = matched[len(lead) + 1:]
                    lowered = matched.lower()
                    break
            if not matched or lowered in NAME_FALLBACK_STOPWORDS:
                continue
            if matched in mapping.values():
                continue
            counters["NAME"] = counters.get("NAME", 0) + 1
            token = f"[NAME_{counters['NAME']}]"
            mapping[token] = matched
            clean = clean.replace(matched, token)

    # Step 2: Labeled patterns — mask only the captured value, keep the label
    # (e.g. "Username: jsmith23" -> "Username: [USERNAME_1]")
    for label, pattern in LABELED_PATTERNS.items():
        for match in re.finditer(pattern, clean, re.IGNORECASE):
            value = match.group(1)
            if value.startswith("[") and value.endswith("]"):
                continue  # already replaced
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            if token not in mapping:
                mapping[token] = value
            # Replace only the value portion, not the whole labeled phrase
            clean = clean[:match.start(1)] + token + clean[match.end(1):]

    # Step 3: Regex patterns (whole match masked)
    for label, pattern in PATTERNS.items():
        if label == "TICKET_REF":
            continue  # Preserve ticket numbers — not PII
        for match in re.finditer(pattern, clean, re.IGNORECASE):
            matched = match.group(0)
            # Skip if already replaced
            if matched.startswith("[") and matched.endswith("]"):
                continue
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            if token not in mapping:
                mapping[token] = matched
            clean = clean.replace(matched, token, 1)

    pii_count = len(mapping)
    if pii_count > 0:
        _audit("redact", f"{pii_count} PII item(s) masked: {list(mapping.keys())}")
    else:
        _audit("redact", "No PII detected")

    return clean, mapping

def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored

def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# ── AUDIT TRAIL LOGGER ────────────────────────────────────────────────────────

class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval."""

    def __init__(self, log_file: str = "logs/audit_trail.jsonl"):
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": rationale[:200] if rationale else "",
            "approval_status": approval_status
        }
        self.entries.append(entry)

        # Append to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'='*55}")
        print(f"FULL AUDIT TRAIL ({len(self.entries)} entries)")
        print(f"{'='*55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("PII REDACTION DEMO")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
        "Login issue for Username: jsmith23 on domain account ZEN\\jsmith. Cannot authenticate.",
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 — VPN failure after password change", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK — 210 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to user — auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: logs/demo_audit.jsonl")