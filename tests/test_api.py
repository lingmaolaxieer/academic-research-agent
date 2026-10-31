"""API tests for the FastAPI wrapper. FastAPI 封装器的 API 测试。

Like the persistence and HITL tests, these exercise our code (the endpoints, the
background runner, and status reporting) against a stub graph—no real LLM calls
or API key are needed, so they run in normal CI.
与持久化和人在环测试一样，这些测试针对的是我们的代码（端点、后台运行器和
状态报告），使用的是存根图——不调用真实的 LLM，也不需要 API 密钥，因此可以
在普通 CI 中运行。

The stub graph is compiled with an in-memory SQLite checkpointer and injected via
``create_api(graph=...)``, so the app never opens the real ``checkpoints.db``.
存根图使用内存 SQLite 检查点编译，并通过 ``create_api(graph=...)`` 注入，
因此应用不会打开真实的 ``checkpoints.db``。
通过 ``create_api(graph=...)`` 注入，因此应用不会打开真实的 ``checkpoints.db``。
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import START, END, StateGraph

from research_agent.api import create_api
from research_agent.config import settings
from research_agent.graph import route_after_approval
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


def _report_node(state):
    """Stand in for the whole graph: emit a report-shaped result and call count.
    代替整个图：生成报告形状的结果和调用次数。"""
    return {
        "report": {
            "summary": f"Report for: {state['query']}",
            "sections": ["section one"],
            "citations": ["https://example.com"],
        },
        "llm_calls": 3,
    }


def _build_stub(saver):
    g = StateGraph(ResearchState)
    g.add_node("run", _report_node)
    g.add_edge(START, "run")
    g.add_edge("run", END)
    return g.compile(checkpointer=saver)


def _poll_until(client, thread_id, targets, timeout=5.0):
    """Background runs change status asynchronously; poll GET until one of
    ``targets`` is reached (for example, {"completed", "error"} or {"interrupted"}).
    后台运行会异步改变状态；轮询 GET 请求，直到达到 ``targets``（例如
    {"completed", "error"} 或 {"interrupted"}）。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/research/{thread_id}").json()
        if body["status"] in targets:
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {thread_id} did not reach {targets} within {timeout}s")


def _poll_until_done(client, thread_id, timeout=5.0):
    return _poll_until(client, thread_id, {"completed", "error"}, timeout)


# A HITL stub mirroring the real graph's shape: planner → approval (the real
# gate) → researcher (fans out, one branch per subtask) → synthesizer. No LLM
# calls. The planner returns ['edited!'] after seeing feedback (a rejected plan),
# otherwise ['a', 'b']. The researcher writes the reduced `findings` channel
# (so parallel branches do not clash), and the synthesizer writes `report` once—
# exactly why route_after_approval needs this fan-out structure.
# 这个人在环存根模拟真实图的结构：规划器 → 审批（真正的审批门）
# → 研究器（扇出，每个子任务对应一个分支）→ 综合器。不调用 LLM。
# 规划器看到反馈（被拒绝的计划）后返回 ['edited!']，否则返回 ['a', 'b']。
# 研究器写入经过约简的 `findings` 通道（避免并行分支冲突），综合器只写入
# 一次 `report`——这正是 route_after_approval 需要这种扇出结构的原因。
def _hitl_planner(state):
    feedback = state.get("plan_feedback")
    return {"subtasks": ["edited!"] if feedback else ["a", "b"], "plan_feedback": ""}


def _hitl_researcher(state):
    return {"findings": []}


def _hitl_synth(state):
    return {"report": {"summary": "done: " + ",".join(state["subtasks"]), "sections": [], "citations": []}}


def _build_hitl_stub(saver):
    g = StateGraph(ResearchState)
    g.add_node("planner", _hitl_planner)
    g.add_node("approval", approval_node)
    g.add_node("researcher", _hitl_researcher)
    g.add_node("synthesizer", _hitl_synth)
    g.add_edge(START, "planner")
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile(checkpointer=saver)


def test_health():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.get("/health").json() == {"status": "ok"}


def test_research_lifecycle():
    """POST queues a run (202 + thread_id); GET eventually returns its report.
    POST 将运行排队（202 + thread_id）；GET 最终返回其报告。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            resp = client.post("/research", json={"query": "what is RAG?"})
            assert resp.status_code == 202
            thread_id = resp.json()["thread_id"]
            assert resp.json()["status"] == "running"

            done = _poll_until_done(client, thread_id)
            assert done["status"] == "completed"
            assert done["report"]["summary"] == "Report for: what is RAG?"
            assert done["report"]["citations"] == ["https://example.com"]
            assert done["llm_calls"] == 3


def test_query_validation_rejects_empty_and_oversized():
    """The request schema caps query size (cost/abuse guard)—both an empty query
    and one over the limit are rejected with 422 before any run is queued.
    请求模式限制查询大小（防止成本失控和滥用）——空查询和超出限制的查询
    都会在排队运行前以 422 被拒绝。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.post("/research", json={"query": ""}).status_code == 422
            assert client.post("/research", json={"query": "x" * 2001}).status_code == 422
            # A query at the boundary is accepted. 达到边界的查询会被接受。
            assert client.post("/research", json={"query": "x" * 2000}).status_code == 202


def test_unknown_thread_is_404():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.get("/research/does-not-exist").status_code == 404


def test_failed_run_reports_error():
    """If the graph raises, the run surfaces as status='error', not a hang.
    如果图抛出异常，运行会显示为 status='error'，而不是一直挂起。"""

    def _boom(state):
        raise RuntimeError("planner exploded")

    with sqlite_saver(":memory:") as saver:
        g = StateGraph(ResearchState)
        g.add_node("run", _boom)
        g.add_edge(START, "run")
        g.add_edge("run", END)
        app = g.compile(checkpointer=saver)

        with TestClient(create_api(graph=app)) as client:
            thread_id = client.post("/research", json={"query": "x"}).json()["thread_id"]
            done = _poll_until_done(client, thread_id)
            assert done["status"] == "error"
            assert "RuntimeError" in done["error"]
            assert "planner exploded" not in done["error"]


# ── Human-in-the-loop over HTTP ───────────────────────────────────────────────
# ── 通过 HTTP 实现人在环 ─────────────────────────────────────────────────────
def test_hitl_pauses_then_resumes_approve(hitl_on):
    """With HITL on, POST pauses at the gate; resume(approve) runs the plan.
    开启 HITL 后，POST 会在审批门暂停；resume(approve) 执行计划。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]

            paused = _poll_until(client, thread_id, {"interrupted"})
            assert paused["status"] == "interrupted"
            assert paused["report"] is None
            assert paused["interrupt"]["proposed_subtasks"] == ["a", "b"]

            resume = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            assert resume.status_code == 202

            done = _poll_until_done(client, thread_id)
            assert done["status"] == "completed"
            assert done["report"]["summary"] == "done: a,b"  # Ran the proposed plan. 执行了拟议计划。


def test_hitl_resume_edit_uses_human_subtasks(hitl_on):
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})

            client.post(
                f"/research/{thread_id}/resume",
                json={"action": "edit", "subtasks": ["x", "y", "z"]},
            )
            done = _poll_until_done(client, thread_id)
            assert done["report"]["summary"] == "done: x,y,z"  # The human's list won. 采用了人类提供的列表。


def test_hitl_reject_repauses_then_approve(hitl_on):
    """A reject re-plans and pauses again, so the client resumes a second time.
    reject 会重新规划并再次暂停，因此客户端需要第二次恢复。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})

            client.post(
                f"/research/{thread_id}/resume",
                json={"action": "reject", "feedback": "too broad"},
            )
            repaused = _poll_until(client, thread_id, {"interrupted"})
            # Looped back to the planner, which saw the feedback, and paused again.
            # 回到看到了反馈的规划器，并再次暂停。
            assert repaused["interrupt"]["proposed_subtasks"] == ["edited!"]

            client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            done = _poll_until_done(client, thread_id)
            assert done["report"]["summary"] == "done: edited!"


def test_resume_rejected_when_not_interrupted(hitl_on):
    """Resuming a finished run is a 409, not a silent no-op.
    恢复已完成的运行会返回 409，而不是静默地无操作。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})
            client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            _poll_until_done(client, thread_id)

            again = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            assert again.status_code == 409


def test_resume_unknown_thread_is_404():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            resp = client.post("/research/nope/resume", json={"action": "approve"})
            assert resp.status_code == 404
