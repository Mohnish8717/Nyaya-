"""Finite-state intake workflows.

Reversed from the workflow schema in the `@vapi-ai/web` types: a workflow is a
list of nodes where a `conversation` node is active until the model exits it,
a `tool` node runs one call and continues, and variables are collected along
the way. The part that matters for a legal product is *who decides the path*:

  * the model talks, and may ask to advance;
  * this file decides whether that transition exists, whether the facts the
    step requires are actually present, and which reviewed procedure the step
    ends at.

So a model cannot invent a step, skip consent, or reach a document-preparation
step with no documents. A step that is not satisfied simply asks its question
again, in different words.

Workflows are also safety-subordinate: an active emergency aborts the workflow
rather than being filed as an intake answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "workflows.json"

NODE_TYPES = {"conversation", "tool", "end"}
TOOL_NODES = {"get_procedure", "search_law", "search_judgments"}
ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")


def _validate(kb: Any) -> dict[str, Any]:
    if not isinstance(kb, dict) or not isinstance(kb.get("workflows"), list):
        raise ValueError("Workflow file must be an object with a workflows array.")
    seen_ids: set[str] = set()
    for workflow in kb["workflows"]:
        wid = str(workflow.get("id", ""))
        if not ID_RE.fullmatch(wid) or wid in seen_ids:
            raise ValueError(f"Invalid or duplicate workflow id: {wid!r}")
        seen_ids.add(wid)
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(f"Workflow {wid} has no nodes.")
        node_ids = {str(node.get("id")) for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError(f"Workflow {wid} has duplicate node ids.")
        if workflow.get("start") not in node_ids:
            raise ValueError(f"Workflow {wid} start node is missing.")
        for node in nodes:
            nid = str(node.get("id"))
            ntype = node.get("type")
            if ntype not in NODE_TYPES:
                raise ValueError(f"Workflow {wid} node {nid} has invalid type {ntype!r}")
            if ntype == "conversation":
                if not str(node.get("ask", "")).strip():
                    raise ValueError(f"Workflow {wid} node {nid} needs an ask.")
                if not isinstance(node.get("requires", []), list):
                    raise ValueError(f"Workflow {wid} node {nid} requires must be a list.")
            if ntype == "tool":
                if node.get("tool") not in TOOL_NODES:
                    raise ValueError(f"Workflow {wid} node {nid} calls an unbounded tool.")
                if node.get("next") not in node_ids:
                    raise ValueError(f"Workflow {wid} tool node {nid} has no next.")
            if ntype == "conversation" and not node.get("transitions") and node.get("next") not in node_ids:
                raise ValueError(f"Workflow {wid} node {nid} has no next or transitions.")
            for transition in node.get("transitions", []) or []:
                if transition.get("to") not in node_ids:
                    raise ValueError(f"Workflow {wid} node {nid} transitions to an unknown node.")
    return kb


def load(path: Path = PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return _validate(json.load(handle))


KB = load()
WORKFLOWS: list[dict[str, Any]] = KB["workflows"]
BY_ID: dict[str, dict[str, Any]] = {workflow["id"]: workflow for workflow in WORKFLOWS}
BY_PROCEDURE: dict[str, dict[str, Any]] = {
    workflow["procedure_id"]: workflow for workflow in WORKFLOWS if workflow.get("procedure_id")
}


def reload(path: Path = PATH) -> None:
    global KB, WORKFLOWS, BY_ID, BY_PROCEDURE
    KB = load(path)
    WORKFLOWS = KB["workflows"]
    BY_ID = {workflow["id"]: workflow for workflow in WORKFLOWS}
    BY_PROCEDURE = {
        workflow["procedure_id"]: workflow for workflow in WORKFLOWS if workflow.get("procedure_id")
    }


def node_of(workflow: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    return next((node for node in workflow["nodes"] if node["id"] == node_id), None)


def for_procedure(procedure_id: str | None) -> dict[str, Any] | None:
    return BY_PROCEDURE.get(str(procedure_id or ""))


def start(workflow_id: str, turn: int = 0) -> dict[str, Any] | None:
    workflow = BY_ID.get(workflow_id)
    if not workflow:
        return None
    return {
        "id": workflow_id,
        "node": workflow["start"],
        "turn": turn,
        "status": "active",
        "history": [workflow["start"]],
        "collected": {},
        # Which step question was last put to the caller, so the same step can
        # be re-asked in different words instead of being repeated verbatim.
        "asked_node": "",
        "shown": False,
    }


def load_state(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or raw.get("id") not in BY_ID:
        return None
    state = {
        "id": str(raw["id"]),
        "node": str(raw.get("node") or BY_ID[raw["id"]]["start"]),
        "turn": int(raw.get("turn", 0) or 0),
        "status": str(raw.get("status") or "active"),
        "history": [str(item) for item in raw.get("history", []) if isinstance(item, str)][-20:],
        "collected": {str(k): str(v) for k, v in (raw.get("collected") or {}).items()},
        "asked_node": str(raw.get("asked_node") or ""),
        "shown": bool(raw.get("shown")),
    }
    if node_of(BY_ID[state["id"]], state["node"]) is None:
        return None
    return state


def _missing(node: dict[str, Any] | None, ledger: Any, matter: str = "") -> list[str]:
    if not node:
        return []
    return [slot for slot in node.get("requires", []) if not ledger.known_for(slot, matter)]


def _condition_matches(condition: Any, ledger: Any, matter: str = "") -> bool:
    if not isinstance(condition, dict):
        return True
    slot = str(condition.get("slot") or "")
    value = ledger.value_for(slot, matter)
    if not value:
        return False
    lowered = value.lower()
    if "contains" in condition and isinstance(condition["contains"], list):
        return any(str(term).lower() in lowered for term in condition["contains"])
    if "equals" in condition:
        return lowered.strip() == str(condition["equals"]).lower().strip()
    return True


def _allowed_targets(workflow: dict[str, Any], node: dict[str, Any], ledger: Any, matter: str = "") -> list[str]:
    """Transitions whose condition holds, in declared order (first wins)."""
    targets: list[str] = []
    for transition in node.get("transitions", []) or []:
        if _condition_matches(transition.get("when"), ledger, matter):
            targets.append(transition["to"])
    if not targets and node.get("next"):
        targets.append(node["next"])
    return targets


def can_advance(state: dict[str, Any], ledger: Any, to: str | None = None) -> tuple[bool, str]:
    workflow = BY_ID[state["id"]]
    node = node_of(workflow, state["node"])
    if not node:
        return False, "unknown_node"
    if state.get("status") != "active":
        return False, f"workflow_{state.get('status')}"
    matter = state["id"]
    missing = _missing(node, ledger, matter)
    if missing:
        return False, "missing_" + ",".join(missing)
    if to:
        if to not in _allowed_targets(workflow, node, ledger, matter):
            return False, "transition_not_allowed"
        return True, "ok"
    return True, "ok"


def advance(
    state: dict[str, Any],
    ledger: Any,
    to: str | None = None,
    turn: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Advance as far as the facts allow. Returns (state, events).

    Events are `advanced`, `tool_call`, `completed`, `refused` or `aborted`.
    A refused transition changes nothing: the current step asks again rather
    than pretending the caller answered.
    """
    events: list[dict[str, Any]] = []
    if state.get("status") != "active":
        return state, events
    workflow = BY_ID.get(state["id"])
    if not workflow:
        return state, events

    if to:
        ok, reason = can_advance(state, ledger, to)
        if not ok:
            events.append({"event": "refused", "reason": reason, "to": to})
            return state, events
        state["node"] = to
        state["history"].append(to)
        state["turn"] = turn
        events.append({"event": "advanced", "to": to, "requested": True})

    for _ in range(len(workflow["nodes"]) + 2):
        node = node_of(workflow, state["node"])
        if node is None:
            break
        if node["type"] == "end":
            state["status"] = "completed"
            events.append({"event": "completed", "procedure_id": workflow.get("procedure_id")})
            break
        if node["type"] == "tool":
            events.append({
                "event": "tool_call",
                "node": node["id"],
                "tool": node["tool"],
                "arguments": node.get("arguments", {}),
            })
            state["node"] = node["next"]
            state["history"].append(node["next"])
            state["turn"] = turn
            continue
        # conversation node: leave it only when its required facts are present
        matter = state["id"]
        missing = _missing(node, ledger, matter)
        if missing:
            break
        targets = _allowed_targets(workflow, node, ledger, matter)
        if not targets:
            break
        state["node"] = targets[0]
        state["history"].append(targets[0])
        state["turn"] = turn
        events.append({"event": "advanced", "to": targets[0], "requested": False})

    # `collected` is a view of the ledger for every slot this workflow can use,
    # rewritten each turn so it cannot drift from what the caller actually said.
    state["collected"] = {
        slot: ledger.value_for(slot, state["id"])
        for slot in {s for item in workflow["nodes"] for s in item.get("requires", [])}
        if ledger.known_for(slot, state["id"])
    }
    return state, events


def abort(state: dict[str, Any] | None, reason: str) -> dict[str, Any] | None:
    """Stop an *active* intake. A finished intake stays finished.

    Safety takes over an intake that is still running; it must not rewrite the
    record of one that already ended, or the session would claim the caller's
    earlier matter was interrupted when it was not.
    """
    if not state or state.get("status") != "active":
        return state
    state["status"] = "aborted"
    state["abort_reason"] = reason
    return state


def is_closed(state: dict[str, Any] | None) -> bool:
    return bool(state) and state.get("status") in {"completed", "aborted"}


def question(state: dict[str, Any] | None, ledger: Any) -> tuple[str | None, str | None]:
    """The current step's question and the first fact it still needs."""
    if not state or state.get("status") != "active":
        return None, None
    workflow = BY_ID.get(state["id"])
    node = node_of(workflow, state["node"]) if workflow else None
    if not node or node["type"] != "conversation":
        return None, None
    missing = _missing(node, ledger, state["id"])
    if not missing:
        return None, None
    return missing[0], str(node["ask"])


def public(state: dict[str, Any] | None, ledger: Any) -> dict[str, Any] | None:
    if not state:
        return None
    workflow = BY_ID.get(state["id"])
    if not workflow:
        return None
    node = node_of(workflow, state["node"])
    steps = [item["id"] for item in workflow["nodes"] if item["type"] == "conversation"]
    # Count questions actually reached, not positions in the file: a branch
    # skips nodes, and reporting 4/4 while the household question was never
    # asked would misstate what happened on the call.
    visited = [node_id for node_id in state.get("history", []) if node_id in steps]
    index = max(1, len(visited))
    total = len(steps) if state.get("status") == "active" else max(index, len(visited))
    slot, ask = question(state, ledger)
    return {
        "id": state["id"],
        "label": workflow["label"],
        "step": index,
        "total": total,
        "skipped": [node_id for node_id in steps if node_id not in visited] if state.get("status") == "completed" else [],
        "node": state["node"],
        "status": state.get("status"),
        "ask": ask,
        "needs": slot,
        "procedure_id": workflow.get("procedure_id"),
        "collected": {
            key: ledger.value_for(key, state["id"])
            for key in sorted({s for n in workflow["nodes"] for s in n.get("requires", [])})
            if ledger.known_for(key, state["id"])
        },
    }
