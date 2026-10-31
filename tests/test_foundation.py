"""Offline regression tests for actual state transitions and failure boundaries."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from research_agent.api import create_api
from research_agent.api_service import _read_status
from research_agent.config import settings
from research_agent.graph import route_after_critic, route_after_output_guard
from research_agent.jobs import JobRegistry
from research_agent.persistence import sqlite_saver
from research_agent.state import ResearchState
from research_agent.structured import StructuredOutputError, invoke_structured


class Answer(BaseModel):
    text: str


class FakeModel:
    def __init__(self, results):
        self.results = iter(results)
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        assert schema is Answer
        assert kwargs == {"method": "function_calling", "include_raw": True}
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def invalid():
    return {"parsed": None, "raw": SimpleNamespace(response_metadata={"finish_reason": "length"})}


def test_invalid_structure_is_retried_and_counted():
    model = FakeModel([invalid(), {"parsed": Answer(text="ok")}])
    result = invoke_structured(model, Answer, "question", attempts=2)
    assert result.value.text == "ok"
    assert result.calls == 2
    assert len(model.prompts) == 2


def test_invalid_structure_has_bounded_failure():
    model = FakeModel([invalid(), invalid()])
    with pytest.raises(StructuredOutputError) as error:
        invoke_structured(model, Answer, "private input", attempts=2)
    assert error.value.attempts == 2
    assert error.value.finish_reason == "length"
    assert "private input" not in str(error.value)


def test_transport_failure_is_not_retried_by_structure_helper():
    model = FakeModel([PermissionError("unauthorized")])
    with pytest.raises(PermissionError):
        invoke_structured(model, Answer, "question", attempts=3)
    assert len(model.prompts) == 1


def test_sufficient_flag_reaches_real_conditional_edge(monkeypatch):
    monkeypatch.setattr(settings, "max_research_iterations", 5)
    graph = StateGraph(ResearchState)
    graph.add_node("critic", lambda s: {"_sufficient": True, "iterations": 1})
    graph.add_node("synthesizer", lambda s: {"gaps": ["finished"]})
    graph.add_node("planner", lambda s: {"gaps": ["incorrect loop"]})
    graph.add_edge(START, "critic")
    graph.add_conditional_edges("critic", route_after_critic, ["planner", "synthesizer"])
    graph.add_edge("planner", END)
    graph.add_edge("synthesizer", END)
    result = graph.compile().invoke({"query": "q"})
    assert result["gaps"] == ["finished"]


def test_repair_flag_is_preserved_and_loop_stops():
    graph = StateGraph(ResearchState)
    graph.add_node("output_guard", lambda s: {"_needs_repair": s.get("repairs", 0) == 0})
    graph.add_node("synthesizer", lambda s: {"repairs": s.get("repairs", 0) + 1})
    graph.add_edge(START, "output_guard")
    graph.add_conditional_edges("output_guard", route_after_output_guard, ["synthesizer", END])
    graph.add_edge("synthesizer", "output_guard")
    result = graph.compile().invoke({"query": "q"})
    assert result["repairs"] == 1
    assert result["_needs_repair"] is False


def test_job_errors_and_pauses_survive_restart(tmp_path):
    path = str(tmp_path / "jobs.db")
    jobs = JobRegistry(path)
    jobs.start("failed")
    jobs.fail("failed", "synthetic failure")
    jobs.start("paused")
    jobs.interrupt("paused", {"question": "approve?"})
    jobs.start("crashed")
    jobs.close()
    jobs = JobRegistry(path)
    try:
        assert jobs.recover_stale() == 1
        assert jobs.get("failed")["error"] == "synthetic failure"
        assert jobs.get("paused")["interrupt"] == {"question": "approve?"}
        assert jobs.get("crashed")["status"] == "error"
    finally:
        jobs.close()


def test_resume_is_atomic_across_database_connections(tmp_path):
    path = str(tmp_path / "jobs.db")
    first, second = JobRegistry(path), JobRegistry(path)
    try:
        first.start("task")
        first.interrupt("task", {"type": "plan_approval"})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(store.claim_resume, "task") for store in (first, second)]
            assert sorted(f.result() for f in futures) == [False, True]
    finally:
        first.close()
        second.close()


def test_legacy_failed_checkpoint_is_not_a_human_pause():
    snapshot = SimpleNamespace(values={"query": "q"}, next=("researcher",), tasks=())
    graph = SimpleNamespace(get_state=lambda config: snapshot)
    jobs = JobRegistry()
    try:
        result = _read_status(graph, jobs, "legacy")
        assert result.status == "error"
        assert result.interrupt is None
    finally:
        jobs.close()


@pytest.mark.parametrize("decision", [
    {"action": "unknown"}, {"action": "edit"},
    {"action": "edit", "subtasks": []},
    {"action": "edit", "subtasks": ["   "]},
    {"action": "edit", "subtasks": ["x"] * 6},
    {"action": "reject"},
    {"action": "approve", "subtasks": ["unexpected"]},
])
def test_bad_decision_rejected_before_execution(decision):
    graph = SimpleNamespace()
    with TestClient(create_api(graph=graph)) as client:
        response = client.post("/research/any/resume", json=decision)
        assert response.status_code == 422


def test_whitespace_query_rejected():
    with TestClient(create_api(graph=SimpleNamespace())) as client:
        assert client.post("/research", json={"query": "   "}).status_code == 422


def test_failure_and_partial_report_stay_failed_after_api_restart(tmp_path):
    def report(state):
        return {"report": {"summary": "partial", "sections": [], "citations": []}}

    def fail(state):
        raise ValueError("repair failed")

    graph = StateGraph(ResearchState)
    graph.add_node("report", report)
    graph.add_node("fail", fail)
    graph.add_edge(START, "report")
    graph.add_edge("report", "fail")
    graph.add_edge("fail", END)
    db = str(tmp_path / "jobs.db")
    with sqlite_saver(str(tmp_path / "checkpoints.db")) as saver:
        compiled = graph.compile(checkpointer=saver)
        with TestClient(create_api(graph=compiled, job_db_path=db)) as client:
            task = client.post("/research", json={"query": "q"}).json()["thread_id"]
        # Shutdown drains work, so querying a new API instance observes the terminal result.
        with TestClient(create_api(graph=compiled, job_db_path=db)) as client:
            result = client.get(f"/research/{task}").json()
            assert result["status"] == "error"
            assert result["report"]["summary"] == "partial"
            assert "ValueError" in result["error"]
            assert "repair failed" not in result["error"]
            assert client.post(f"/research/{task}/resume", json={"action": "approve"}).status_code == 409
