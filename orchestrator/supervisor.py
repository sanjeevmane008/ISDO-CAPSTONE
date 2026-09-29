"""
ISDO Lab C6 — LangGraph Orchestrator
Wires Triage, Resolution, SLA, HITL, and Communication agents into a single
LangGraph StateGraph. Every node reads and writes the shared TicketState.

Flow:
  triage -> resolution -> sla -> (conditional) -> [hitl ->] communication

Requires: Labs C1/C4 KB data at data/kb/*.md.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Optional

import anthropic
import chromadb
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── SHARED STATE ──────────────────────────────────────────────────────────────
# Every field is optional at graph start — each agent only writes the fields it owns.

class TicketState(TypedDict, total=False):
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    kb_article: Optional[str]
    resolution_text: str
    auto_resolve: bool
    confidence: str
    sla_breach_risk: str
    escalation_required: bool
    hitl_required: bool
    hitl_approved: bool
    user_message: str
    final_status: str
    audit_log: List[dict]

def log(state: TicketState, agent: str, action: str, detail: str) -> None:
    """Every node appends here — this is the audit trail Lab C9 persists to disk."""
    state.setdefault("audit_log", []).append({
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "detail": detail
    })
    print(f"  [AUDIT] {agent}: {action}")

# ── KB SETUP (same chunking logic as Lab C1 / C4) ────────────────────────────

KB_DIR = Path("data/kb")

def chunk_article(text: str, filename: str) -> list[dict]:
    chunks, current_lines, current_heading = [], [], "Introduction"
    for line in text.split("\n"):
        if line.startswith("## ") and current_lines:
            chunks.append({"content": "\n".join(current_lines).strip(),
                            "heading": current_heading, "filename": filename})
            current_lines, current_heading = [], line[3:].strip()
        current_lines.append(line)
    if current_lines:
        chunks.append({"content": "\n".join(current_lines).strip(),
                        "heading": current_heading, "filename": filename})
    return chunks

def build_kb():
    db = chromadb.Client()
    try:
        kb = db.get_collection("isdo_kb")
        print("KB already loaded.")
        return kb
    except Exception:
        pass

    kb = db.create_collection("isdo_kb")
    docs, ids, metas = [], [], []
    chunk_id = 0
    for md_file in sorted(KB_DIR.glob("*.md")):
        for chunk in chunk_article(md_file.read_text(), md_file.name):
            docs.append(chunk["content"])
            ids.append(f"kb_{chunk_id}")
            metas.append({"filename": md_file.name, "heading": chunk["heading"]})
            chunk_id += 1
    if docs:
        kb.add(documents=docs, ids=ids, metadatas=metas)
        print(f"KB loaded: {len(docs)} chunks")
    return kb

KB = build_kb()

def search_kb(query: str):
    """Return (filename, confidence_score, full_article_text) for the best match, or (None, 0.0, '')."""
    raw = KB.query(query_texts=[query], n_results=10)
    best_per_file = {}
    for i, _doc in enumerate(raw["documents"][0]):
        meta = raw["metadatas"][0][i] if raw["metadatas"] else {}
        distance = raw["distances"][0][i] if raw.get("distances") else 1.0
        fname = meta.get("filename", f"unknown_{i}")
        if fname not in best_per_file or distance < best_per_file[fname]:
            best_per_file[fname] = distance

    if not best_per_file:
        return None, 0.0, ""

    fname, distance = min(best_per_file.items(), key=lambda kv: kv[1])
    confidence_score = max(0, 1 - distance)
    full_path = KB_DIR / fname
    full_text = full_path.read_text() if full_path.exists() else ""
    return fname, confidence_score, full_text

# ── SLA THRESHOLDS (same as Lab C5) ──────────────────────────────────────────

SLA_HOURS = {"P1": 1, "P2": 4, "P3": 8, "P4": 24}
SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)  # fixed 'now' for reproducible demo results

def calc_sla_risk(sla_due: str, priority: str):
    due_dt = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    minutes_remaining = int((due_dt - SIMULATED_NOW).total_seconds() / 60)
    total_minutes = SLA_HOURS.get(priority, 8) * 60

    if minutes_remaining < 0:
        risk = "BREACHED"
    elif minutes_remaining < total_minutes * 0.2:
        risk = "CRITICAL"
    elif minutes_remaining < total_minutes * 0.5:
        risk = "AT_RISK"
    else:
        risk = "ON_TRACK"
    return risk, minutes_remaining

# ── TOOL SCHEMAS ──────────────────────────────────────────────────────────────

CLASSIFY_TOOL = {
    "name": "classify_ticket",
    "description": "Classify an IT support ticket. Returns category, priority, assignment_group, and whether PII was detected.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["Network", "Application", "Hardware", "Access", "Email", "Server", "Software"]
            },
            "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            "assignment_group": {"type": "string", "description": "Team to assign the ticket to"},
            "pii_detected": {"type": "boolean"},
            "reasoning": {"type": "string", "description": "One sentence explaining the classification"}
        },
        "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"]
    }
}

DRAFT_RESOLUTION_TOOL = {
    "name": "draft_resolution",
    "description": "Draft a resolution message based strictly on the supplied KB article content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resolution_text": {
                "type": "string",
                "description": "3-4 plain-English steps drawn directly from the KB article"
            },
            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "auto_resolve": {"type": "boolean"}
        },
        "required": ["resolution_text", "confidence", "auto_resolve"]
    }
}

def call_tool_forced(system_prompt: str, user_content: str, tool_schema: dict) -> dict:
    """Single forced tool call. A StateGraph node needs one deterministic result,
    not the multi-round ReAct loop used in the standalone C3-C5 agent scripts —
    tool_choice pins the model to this exact tool on the first turn."""
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=600,
        output_config={"effort": "low"},
        system=system_prompt,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": tool_schema["name"]},
        messages=[{"role": "user", "content": user_content}]
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return {}

# ── NODES ─────────────────────────────────────────────────────────────────────

TRIAGE_SYSTEM = """You are the ISDO Triage Agent for Zensar's IT Service Desk.
Classify the ticket using the classify_ticket tool.

Priority rules:
- P1: Service down, many users affected, or security breach
- P2: Significant impact, single department or function affected
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment)"""

def triage_node(state: TicketState) -> TicketState:
    print(f"\n▶ TRIAGE AGENT — {state['ticket_number']}")
    result = call_tool_forced(
        TRIAGE_SYSTEM,
        f"Ticket: {state['ticket_number']}\nSummary: {state['short_description']}\nDetails: {state['description']}",
        CLASSIFY_TOOL
    )
    state["triage_category"] = result.get("category")
    state["triage_priority"] = result.get("priority")
    state["triage_assignment_group"] = result.get("assignment_group")
    state["pii_detected"] = bool(result.get("pii_detected", False))

    print(f"  Category: {state['triage_category']}   Priority: {state['triage_priority']}")
    print(f"  Assign To: {state['triage_assignment_group']}   PII: {state['pii_detected']}")
    log(state, "TriageAgent", "classify_ticket", f"{state['triage_category']} / {state['triage_priority']}")
    return state

RESOLUTION_SYSTEM = """You are the ISDO Resolution Agent for Zensar's IT Service Desk.
Use the draft_resolution tool to write a resolution using ONLY the KB article content provided —
never invent steps that aren't in the article.

Confidence rules:
- HIGH (article fully covers the issue): auto_resolve=True only if priority is P3 or P4
- MEDIUM (partial match): auto_resolve=False
- LOW (no relevant KB content provided): auto_resolve=False, say escalation is needed"""

def resolution_node(state: TicketState) -> TicketState:
    print(f"\n▶ RESOLUTION AGENT — searching KB")
    query = f"{state['short_description']} {state['description']}"
    fname, score, article_text = search_kb(query)
    log(state, "ResolutionAgent", "search_kb", f"{fname or 'no match'} ({score:.0%})")

    if fname is None:
        state["kb_article"] = None
        state["resolution_text"] = "No matching KB article found — escalate to L2 for manual review."
        state["confidence"] = "LOW"
        state["auto_resolve"] = False
    else:
        priority = state.get("triage_priority") or state.get("priority")
        user_content = (
            f"Ticket: {state['ticket_number']} (Priority: {priority})\n"
            f"Issue: {state['short_description']}\n{state['description']}\n\n"
            f"KB Article ({fname}):\n{article_text}\n\n"
            f"Similarity score: {score:.2f}  (>0.6=HIGH, >0.35=MEDIUM, else LOW)"
        )
        result = call_tool_forced(RESOLUTION_SYSTEM, user_content, DRAFT_RESOLUTION_TOOL)
        state["kb_article"] = fname
        state["resolution_text"] = result.get("resolution_text", "")
        state["confidence"] = result.get("confidence", "LOW")
        state["auto_resolve"] = bool(result.get("auto_resolve", False))

    print(f"  KB Article: {state['kb_article']}")
    print(f"  Confidence: {state['confidence']} ({score:.0%})  |  Auto-resolve: {state['auto_resolve']}")
    return state

def sla_node(state: TicketState) -> TicketState:
    print(f"\n▶ SLA AGENT — checking deadline")
    priority = state.get("triage_priority") or state["priority"]
    risk, minutes_remaining = calc_sla_risk(state["sla_due"], priority)

    state["sla_breach_risk"] = risk
    state["escalation_required"] = risk in ("CRITICAL", "BREACHED")
    state["hitl_required"] = state["escalation_required"] and priority == "P1"

    print(f"  SLA Risk: {risk}  |  Minutes remaining: {minutes_remaining}")
    log(state, "SLAAgent", "get_sla_status", f"{risk} ({minutes_remaining} min remaining)")
    return state

def hitl_node(state: TicketState) -> TicketState:
    print(f"\n▶ HITL GATE — human approval required")
    print(f"  Ticket:  {state['ticket_number']}")
    print(f"  Reason:  Priority {state.get('triage_priority')} with SLA risk {state['sla_breach_risk']}")
    decision = input("  Approve escalation? [y/n]: ").strip().lower()
    state["hitl_approved"] = (decision == "y")
    log(state, "HITLNode", "approve_escalation", "APPROVED" if state["hitl_approved"] else "REJECTED")
    return state

def communication_node(state: TicketState) -> TicketState:
    print(f"\n▶ COMMUNICATION AGENT")
    ticket = state["ticket_number"]

    if state.get("auto_resolve"):
        state["user_message"] = (
            f"Dear User, regarding {ticket}: we found a known fix for your issue. "
            f"Please follow these steps:\n{state.get('resolution_text', '')}"
        )
        state["final_status"] = "RESOLVED"
    elif state.get("hitl_required") and state.get("hitl_approved"):
        state["user_message"] = (
            f"Dear User, regarding {ticket}: your ticket has been escalated to our senior "
            f"support team due to its priority. You will receive an update shortly."
        )
        state["final_status"] = "ESCALATED"
    elif state.get("hitl_required") and not state.get("hitl_approved"):
        state["user_message"] = (
            f"Dear User, regarding {ticket}: your ticket remains under review by our support team."
        )
        state["final_status"] = "ESCALATION_REJECTED"
    else:
        state["user_message"] = (
            f"Dear User, regarding {ticket}: your ticket has been assigned to "
            f"{state.get('triage_assignment_group', 'our support team')} for further action."
        )
        state["final_status"] = "ASSIGNED"

    print(f"  USER MESSAGE: {state['user_message'][:80]}...")
    print(f"  ✅ FINAL STATUS: {state['final_status']}")
    log(state, "CommunicationAgent", "draft_message", state["final_status"])
    return state

# ── CONDITIONAL ROUTING ───────────────────────────────────────────────────────

def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"

# ── BUILD THE GRAPH ───────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("triage", triage_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("sla", sla_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("communication", communication_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "resolution")
    graph.add_edge("resolution", "sla")
    graph.add_conditional_edges("sla", route_after_sla, {
        "hitl": "hitl",
        "communication": "communication"
    })
    graph.add_edge("hitl", "communication")
    graph.add_edge("communication", END)

    return graph.compile()

# ── RUN ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()

    test_tickets = [
        {
            "ticket_number": "INC0001001",
            "short_description": "VPN not connecting after password change",
            "description": "User reports VPN client fails to connect after AD password was reset. Error: authentication failed.",
            "category": "Network",
            "priority": "P2",
            "sla_due": "2024-01-15 14:00:00",
        },
        {
            "ticket_number": "INC0001002",
            "short_description": "Cannot access ERP system - login error",
            "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today.",
            "category": "Application",
            "priority": "P1",
            "sla_due": "2024-01-15 11:00:00",
        },
    ]

    for ticket in test_tickets:
        print(f"\n{'═'*58}")
        print(f"PROCESSING TICKET: {ticket['ticket_number']}")
        print(f"{'═'*58}")

        initial_state: TicketState = {**ticket, "audit_log": []}
        final_state = app.invoke(initial_state)

        print(f"\n--- AUDIT LOG for {ticket['ticket_number']} ---")
        for entry in final_state["audit_log"]:
            print(f"  [{entry['timestamp']}] {entry['agent']} -> {entry['action']}: {entry['detail']}")