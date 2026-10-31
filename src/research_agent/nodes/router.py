"""Cost-aware deterministic routing before expensive planning."""
from __future__ import annotations

import re


DEEP_SIGNALS = {
    "compare", "comparison", "survey", "review", "trade-off", "tradeoff",
    "evaluate", "analyze", "systematic", "综述", "比较", "对比", "评估",
    "分析", "调研", "研究现状", "优缺点", "异同",
}


def router_node(state):
    requested = state.get("research_depth", "deep")
    if requested in {"quick", "deep"}:
        route = requested
    elif state.get("require_approval"):
        route = "deep"
    else:
        query = state.get("query", "").strip().lower()
        terms = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", query)
        route = "deep" if len(terms) > 40 or any(signal in query for signal in DEEP_SIGNALS) else "quick"
    return {"route": route}


def quick_plan_node(state):
    """A single bounded retrieval task; no planner-model call is needed."""
    return {"subtasks": [state["query"]], "iterations": 0, "gaps": []}
