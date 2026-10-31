"""Offline checks for production adapters without requiring external services."""
from types import SimpleNamespace
import threading
import time

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from research_agent.api import create_api
from research_agent.config import settings
from research_agent.graph import route_after_researcher
from research_agent.jobs import IdempotencyConflict, JobRegistry, LeaseLost
from research_agent.nodes.router import router_node
from research_agent.outbox import dispatch_once
from research_agent.rerank import rerank, warm_reranker
from research_agent.state import ResearchState
from evals.trajectory import evaluate_trajectory


def test_job_events_and_fenced_worker_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "worker_lease_seconds", 60)
    jobs = JobRegistry(str(tmp_path / "jobs.db"))
    try:
        jobs.start("task", "alice", "queued")
        token = jobs.claim_execution("task", "worker-a")
        assert token and token.startswith("worker-a:")
        assert jobs.claim_execution("task", "worker-b") is None
        assert jobs.renew_lease("task", token)
        jobs.append_event("task", "progress", {"step": "planner"})
        jobs.finish("task")
        events = jobs.events("task")
        assert [event["kind"] for event in events] == [
            "queued", "running", "progress", "completed"
        ]
        assert [event["seq"] for event in events] == sorted(event["seq"] for event in events)
    finally:
        jobs.close()


def test_cross_encoder_reranks_retrieval_candidates(monkeypatch):
    monkeypatch.setattr(settings, "reranker_model", "offline-fixture")
    evidence = [
        {"evidence_id": "ev_a", "text": "weak", "score": .9},
        {"evidence_id": "ev_b", "text": "strong", "score": .1},
    ]
    model = SimpleNamespace(predict=lambda pairs: [0.1, 0.9])
    result = rerank("query", evidence, model=model, top_k=1)
    assert result[0]["evidence_id"] == "ev_b"
    assert result[0]["score"] == 0.9


def test_reranker_can_be_preloaded(monkeypatch):
    from research_agent import rerank as rerank_module

    loaded = []
    monkeypatch.setattr(settings, "reranker_model", "offline-fixture")
    monkeypatch.setattr(rerank_module, "_cross_encoder", loaded.append)
    warm_reranker()
    assert loaded == ["offline-fixture"]


def test_cost_router_selects_quick_and_deep_paths():
    assert router_node({"query": "什么是 RAG？", "research_depth": "auto"})["route"] == "quick"
    assert router_node({"query": "比较 RAG 与微调的优缺点", "research_depth": "auto"})["route"] == "deep"
    assert router_node({"query": "short", "research_depth": "auto",
                        "require_approval": True})["route"] == "deep"
    assert router_node({"query": "compare", "research_depth": "quick"})["route"] == "quick"
    assert route_after_researcher({"route": "quick"}) == "synthesizer"
    assert route_after_researcher({"route": "deep"}) == "critic"


def test_trajectory_evaluation_checks_path_and_tool_grounding():
    names = ["input_guard", "router", "quick_plan", "researcher", "synthesizer", "output_guard"]
    events = ([{"kind": "node_start", "name": name} for name in names]
              + [{"kind": "node_end", "name": name} for name in names]
              + [{"kind": "tool_start", "name": "search_documents"}])
    result = evaluate_trajectory(events, "quick")
    assert result["valid"]
    broken = evaluate_trajectory(events[:-2], "quick")
    assert not broken["valid"]


class FakeOutbox:
    def __init__(self):
        self.sent = []
        self.retried = []

    def claim_outbox(self):
        return [{"id": 1, "message_id": "msg", "thread_id": "thread",
                 "payload": {"inputs": {"query": "q"}, "resume": None}, "attempts": 1,
                 "publish_token": "claim-one"}]

    def mark_outbox_sent(self, identifier, publish_token):
        self.sent.append((identifier, publish_token))

    def retry_outbox(self, identifier, attempts, publish_token):
        self.retried.append((identifier, attempts, publish_token))


def test_outbox_publishes_stable_message_id(monkeypatch):
    observed = {}

    def send_task(name, **kwargs):
        observed.update(name=name, **kwargs)

    monkeypatch.setattr("research_agent.outbox.celery_app.send_task", send_task)
    store = FakeOutbox()
    assert dispatch_once(store) == 1
    assert observed["name"] == "research.execute"
    assert observed["task_id"] == "msg"
    assert observed["kwargs"]["thread_id"] == "thread"
    assert store.sent == [(1, "claim-one")]
    assert store.retried == []


def test_sse_replays_persisted_task_events(tmp_path):
    graph = StateGraph(ResearchState)
    graph.add_node("done", lambda state: {"report": {
        "summary": "done", "sections": [], "citations": [], "limitations": []
    }})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    app = create_api(
        graph.compile(checkpointer=InMemorySaver()),
        job_db_path=str(tmp_path / "jobs.db"),
    )
    with TestClient(app) as client:
        thread_id = client.post("/research", json={"query": "status"}).json()["thread_id"]
        for _ in range(100):
            if client.get(f"/research/{thread_id}").json()["status"] == "completed":
                break
            time.sleep(.01)
        response = client.get(f"/research/{thread_id}/events")
        assert response.status_code == 200
        assert "event: running" in response.text
        assert "event: completed" in response.text
        first_id = next(line.split(": ", 1)[1] for line in response.text.splitlines()
                        if line.startswith("id: "))
        replay = client.get(f"/research/{thread_id}/events", headers={"Last-Event-ID": first_id})
        assert f"id: {first_id}\n" not in replay.text


def test_expired_worker_cannot_overwrite_new_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "worker_lease_seconds", 60)
    jobs = JobRegistry(str(tmp_path / "jobs.db"))
    try:
        jobs.start("task", "alice", "queued")
        old_token = jobs.claim_execution("task", "old-worker")
        with jobs._lock, jobs._db:
            jobs._db.execute("UPDATE research_jobs SET lease_until=0 WHERE thread_id='task'")
        new_token = jobs.claim_execution("task", "new-worker")
        assert new_token and new_token != old_token
        assert not jobs.finish("task", old_token)
        with pytest.raises(LeaseLost):
            jobs.append_event("task", "progress", token=old_token)
        with pytest.raises(LeaseLost):
            jobs.set_metrics("task", {"model_calls": 99}, token=old_token)
        assert jobs.finish("task", new_token)
        assert jobs.get("task")["status"] == "completed"
        assert jobs.get("task")["metrics"] == {}
        assert [event["kind"] for event in jobs.events("task")].count("completed") == 1
    finally:
        jobs.close()


def test_idempotency_key_scoped_to_owner_and_request(tmp_path):
    jobs = JobRegistry(str(tmp_path / "jobs.db"))
    try:
        assert jobs.start_idempotent("task-a", "alice", "running", "key", "hash-a") == (
            "task-a", True
        )
        assert jobs.start_idempotent("unused", "alice", "running", "key", "hash-a") == (
            "task-a", False
        )
        with pytest.raises(IdempotencyConflict):
            jobs.start_idempotent("wrong", "alice", "running", "key", "hash-b")
        assert jobs.start_idempotent("task-b", "bob", "running", "key", "hash-b") == (
            "task-b", True
        )
        assert len(jobs.list_jobs("alice")) == 1
        assert len(jobs.list_jobs("bob")) == 1
    finally:
        jobs.close()


def test_api_reuses_idempotent_request(tmp_path):
    graph = StateGraph(ResearchState)
    graph.add_node("done", lambda state: {"report": {
        "summary": "done", "sections": [], "citations": [], "limitations": []
    }})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    app = create_api(graph.compile(checkpointer=InMemorySaver()),
                     job_db_path=str(tmp_path / "jobs.db"))
    with TestClient(app) as client:
        headers = {"Idempotency-Key": "same-request"}
        first = client.post("/research", json={"query": "one"}, headers=headers)
        again = client.post("/research", json={"query": "one"}, headers=headers)
        assert first.status_code == again.status_code == 202
        assert first.json()["thread_id"] == again.json()["thread_id"]
        assert len(client.get("/research").json()) == 1
        conflict = client.post("/research", json={"query": "two"}, headers=headers)
        assert conflict.status_code == 409
        assert client.post("/research", json={"query": "one"},
                           headers={"Idempotency-Key": "invalid key"}).status_code == 400


def test_cancel_running_job_prevents_late_completion(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow_node(state):
        entered.set()
        assert release.wait(timeout=5)
        return {"report": {"summary": "late", "sections": [],
                           "citations": [], "limitations": []}}

    graph = StateGraph(ResearchState)
    graph.add_node("slow", slow_node)
    graph.add_edge(START, "slow")
    graph.add_edge("slow", END)
    app = create_api(graph.compile(checkpointer=InMemorySaver()),
                     job_db_path=str(tmp_path / "jobs.db"))
    with TestClient(app) as client:
        thread_id = client.post("/research", json={"query": "slow"}).json()["thread_id"]
        try:
            assert entered.wait(timeout=5)
            cancelled = client.post(f"/research/{thread_id}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert client.post(f"/research/{thread_id}/cancel").status_code == 200
        finally:
            release.set()
        for _ in range(100):
            if client.get(f"/research/{thread_id}").json()["status"] == "cancelled":
                break
            time.sleep(.01)
        status = client.get(f"/research/{thread_id}").json()
        assert status["status"] == "cancelled"
        assert status["report"] is None
        events = client.get(f"/research/{thread_id}/events").text
        assert "event: cancelled" in events
        assert "event: completed" not in events
    jobs = JobRegistry(str(tmp_path / "jobs.db"))
    try:
        assert jobs.get(thread_id)["status"] == "cancelled"
        assert "completed" not in [event["kind"] for event in jobs.events(thread_id)]
    finally:
        jobs.close()


def test_cancel_requires_task_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", {"alice": "alice-secret", "bob": "bob-secret"})
    monkeypatch.setattr(settings, "allow_anonymous", False)
    graph = StateGraph(ResearchState)
    graph.add_node("done", lambda state: {"report": {
        "summary": "done", "sections": [], "citations": [], "limitations": []
    }})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    app = create_api(graph.compile(checkpointer=InMemorySaver()),
                     job_db_path=str(tmp_path / "jobs.db"))
    with TestClient(app) as client:
        created = client.post("/research", json={"query": "q"},
                              headers={"X-API-Key": "alice-secret"})
        thread_id = created.json()["thread_id"]
        assert client.post(f"/research/{thread_id}/cancel",
                           headers={"X-API-Key": "bob-secret"}).status_code == 404
