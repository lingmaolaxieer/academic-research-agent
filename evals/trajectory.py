"""Deterministic workflow-trajectory checks independent of answer quality judges."""
from __future__ import annotations


PATHS = {
    "quick": {"input_guard", "router", "quick_plan", "researcher", "synthesizer", "output_guard"},
    "deep": {"input_guard", "router", "planner", "researcher", "critic", "synthesizer", "output_guard"},
}


def evaluate_trajectory(events: list[dict], route: str) -> dict:
    """Check node coverage, tool grounding, balanced starts/ends and errors."""
    expected = PATHS[route]
    started = [event.get("name") for event in events if event.get("kind") == "node_start"]
    finished = [event.get("name") for event in events if event.get("kind") == "node_end"]
    errors = [event for event in events if event.get("kind") in {"node_error", "model_error"}]
    missing = sorted(expected - set(started))
    unclosed = sorted(name for name in set(started) if started.count(name) > finished.count(name))
    used_tool = any(event.get("kind") == "tool_start" for event in events)
    return {
        "route": route,
        "valid": not missing and not unclosed and not errors and used_tool,
        "missing_nodes": missing,
        "unclosed_nodes": unclosed,
        "error_count": len(errors),
        "used_retrieval_tool": used_tool,
    }
