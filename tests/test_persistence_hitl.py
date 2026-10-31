"""
These exercise OUR code (the approval gate, the routing, the checkpointer wiring)
with stub planner/researcher nodes, so they make NO real LLM calls and need no
API key. They run in normal CI (not gated behind the `eval` marker).
这些测试使用 stub planner/researcher 节点测试我们的代码（审批门、路由和
checkpointer wiring），因此不会调用真实 LLM，也不需要 API key。
它们在普通 CI 中运行（不受 `eval` 标记门控）。
"""

from __future__ import annotations

import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from research_agent.config import settings
from research_agent.graph import build_graph, route_after_approval
from research_agent.nodes.approval import approval_node
from research_agent.persistence import sqlite_saver
from research_agent.state import ResearchState


@pytest.fixture
def hitl_on():
    """Turn the approval gate on for a test, then restore the default (off).
    为测试开启审批门，然后恢复默认值（关闭）。"""
    original = settings.hitl_approval_enabled
    settings.hitl_approval_enabled = True
    yield
    settings.hitl_approval_enabled = original


# A stub graph: planner → approval → researcher, with NO LLM calls. The planner
# returns ['edited!'] when it sees human feedback (a rejected plan), else ['a','b'].
# 一个 stub 图：planner → approval → researcher，不调用 LLM。planner 看到人类反馈
# （计划被拒绝）时返回 ['edited!']，否则返回 ['a','b']。
def _stub_planner(state):
    feedback = state.get("plan_feedback")
    return {"subtasks": ["edited!"] if feedback else ["a", "b"], "plan_feedback": ""}


def _stub_researcher(state):
    return {"findings": []}


def _build_stub():
    g = StateGraph(ResearchState)
    g.add_node("planner", _stub_planner)
    g.add_node("approval", approval_node)
    g.add_node("researcher", _stub_researcher)
    g.add_edge(START, "planner")
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_edge("researcher", END)
    return g


# ── Pass-through behaviour (directly callable: returns before interrupt()) ──────
# ── 直通行为（可直接调用：在 interrupt() 之前返回） ───────────────────────────
def test_gate_disabled_is_passthrough():
    """With HITL off, the node never interrupts and never marks a rejection.
    HITL 关闭时，节点不会中断，也不会标记拒绝。

    Force the flag off rather than reading the ambient default — a developer may
    have HITL_APPROVAL_ENABLED=true in their .env.
    强制关闭该标志，而不是读取环境默认值——开发者可能在 .env 中设置
    HITL_APPROVAL_ENABLED=true。
    """
    original = settings.hitl_approval_enabled
    settings.hitl_approval_enabled = False
    try:
        assert approval_node({"subtasks": ["a"]}) == {"_plan_rejected": False}
    finally:
        settings.hitl_approval_enabled = original


def test_gate_skipped_on_critic_loop(hitl_on):
    """Even with HITL on, a critic-loop replan (iterations > 0) is not re-gated.
    即使 HITL 开启，critic 循环的重新规划（iterations > 0）也不会再次经过审批门。"""
    out = approval_node({"subtasks": ["a"], "iterations": 1})
    assert out == {"_plan_rejected": False}


# ── Full pause/resume machinery (needs a checkpointer) ─────────────────────────
# ── 完整暂停/恢复机制（需要 checkpointer） ────────────────────────────────────
def test_approve(hitl_on):
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "approve"}}

        paused = app.invoke({"query": "q"}, config=cfg)
        assert paused["__interrupt__"][0].value["proposed_subtasks"] == ["a", "b"]

        done = app.invoke(Command(resume={"action": "approve"}), config=cfg)
        assert done["subtasks"] == ["a", "b"]  # ran as proposed 按拟议方案运行
        assert "__interrupt__" not in done


def test_edit(hitl_on):
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "edit"}}

        app.invoke({"query": "q"}, config=cfg)
        done = app.invoke(
            Command(resume={"action": "edit", "subtasks": ["x", "y", "z"]}), config=cfg
        )
        assert done["subtasks"] == ["x", "y", "z"]  # human's list won 采用了人类提供的列表


def test_reject_replans_and_repauses(hitl_on):
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "reject"}}

        app.invoke({"query": "q"}, config=cfg)
        repaused = app.invoke(
            Command(resume={"action": "reject", "feedback": "too broad"}), config=cfg
        )
        # Looped back to the planner (which saw the feedback) and paused again.
        # 回到看到了反馈的 planner，并再次暂停。
        assert repaused["__interrupt__"][0].value["proposed_subtasks"] == ["edited!"]

        done = app.invoke(Command(resume={"action": "approve"}), config=cfg)
        assert done["subtasks"] == ["edited!"]


def test_abort_ends_the_run(hitl_on):
    """Abort stops the graph for good: it routes to END (no fan-out, no re-pause)
    and leaves a terminal 'aborted' report rather than a dangling pause.
    中止会永久停止图：路由到 END（不进行 fan-out，也不再次暂停），
    并留下终止的“aborted”报告，而不是悬挂的暂停。"""
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "abort"}}

        app.invoke({"query": "q"}, config=cfg)
        done = app.invoke(
            Command(resume={"action": "abort", "reason": "not worth it"}), config=cfg
        )
        assert "__interrupt__" not in done            # not paused — it ended 未暂停——已结束
        assert done["_aborted"] is True
        assert "not worth it" in done["report"]["summary"]
        assert done.get("findings", []) == []         # researcher never ran researcher 从未运行

        # The run is terminal: re-invoking the same thread does not resurrect it.
        # 运行已终止：再次调用同一 thread 不会使其复活。
        state = app.get_state(cfg)
        assert state.next == ()                        # no pending nodes 没有待处理节点


def test_run_survives_restart(hitl_on, tmp_path):
    """A paused run is durable: a brand-new connection resumes the same thread.
    暂停的运行具有持久性：全新连接可以恢复同一个 thread。"""
    db = str(tmp_path / "checkpoints.db")
    cfg: RunnableConfig = {"configurable": {"thread_id": "survive"}}

    with sqlite_saver(db) as saver:  # "process 1": start + pause, then close “进程 1”：启动并暂停，然后关闭
        app = _build_stub().compile(checkpointer=saver)
        assert app.invoke({"query": "q"}, config=cfg).get("__interrupt__")

    with sqlite_saver(db) as saver:  # "process 2": fresh connection to same file “进程 2”：连接同一文件的新连接
        app = _build_stub().compile(checkpointer=saver)
        done = app.invoke(Command(resume={"action": "approve"}), config=cfg)
        assert done["subtasks"] == ["a", "b"]


# ── The real graph still compiles both ways (no regression for evals/CI) ───────
# ── 真实图仍可用两种方式编译（对 evals/CI 无回归） ───────────────────────────
def test_real_graph_builds_with_and_without_checkpointer():
    assert "approval" in build_graph().get_graph().nodes
    with sqlite_saver(":memory:") as saver:
        assert build_graph(checkpointer=saver).checkpointer is not None
