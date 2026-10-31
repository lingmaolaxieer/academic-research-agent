"""Offline integration: real graph, tools, persistence and MCP; synthetic model only."""
import asyncio
import re
from types import SimpleNamespace

import pytest
import httpx
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.api import create_api
from research_agent.config import settings
from research_agent.evidence import Evidence, context_for, evidence_id, merge_evidence
from research_agent.graph import build_graph
from research_agent.knowledge import KnowledgeStore, chunks, parse_document
from research_agent.mcp_client import call_knowledge
from research_agent.nodes.guardrails import CitationValidationError, output_guard_node
from research_agent.nodes.synthesizer import synthesizer_node
from research_agent.observation import BudgetExceeded, RunObserver


@pytest.fixture
def knowledge(tmp_path, monkeypatch):
    path = str(tmp_path / "knowledge.db")
    monkeypatch.setattr(settings, "knowledge_db_path", path)
    monkeypatch.setattr(settings, "retrieval_mode", "bm25")
    monkeypatch.setattr(settings, "knowledge_transport", "local")
    monkeypatch.setattr(settings, "api_keys", {})
    monkeypatch.setattr(settings, "allow_anonymous", True)
    monkeypatch.setattr(settings, "requests_per_minute", 1000)
    return KnowledgeStore(path)


def test_retrieval_owner_filter_dedupe_delete(knowledge):
    a = knowledge.add("alice", "paper.md", b"Visible thermal fusion combines complementary sensors.")
    b = knowledge.add("bob", "secret.md", b"Visible thermal fusion bob confidential results.")
    assert knowledge.add("alice", "paper.md", b"Visible thermal fusion combines complementary sensors.")["duplicate"]
    results = knowledge.search("alice", "thermal fusion")
    assert len(results) == 1
    assert results[0]["document_id"] == a["document_id"]
    assert knowledge.search("alice", "thermal", document_ids=[b["document_id"]]) == []
    assert knowledge.get_chunk("bob", results[0]["chunk_id"]) is None
    assert not knowledge.delete("bob", a["document_id"])
    assert knowledge.delete("alice", a["document_id"])
    assert knowledge.search("alice", "thermal") == []


def test_chinese_search_and_preferences_isolation(knowledge):
    knowledge.add("a", "中文.md", "可见光与热红外融合可以用于目标检测。".encode())
    assert knowledge.search("a", "目标检测")
    knowledge.preferences("a", "中文回答")
    assert knowledge.preferences("b") == ""
    assert knowledge.preferences("a") == "中文回答"
    assert knowledge.preferences("a", "") == ""


def test_document_bounds_and_chunk_overlap():
    assert list(chunks("0123456789", 6, 2)) == [(0, "012345"), (4, "456789")]
    with pytest.raises(ValueError):
        list(chunks("x", 2, 2))
    for name, content in [("x.exe", b"x"), ("x.txt", b""), ("x.pdf", b"invalid")]:
        with pytest.raises(ValueError):
            parse_document(name, content)


class Embeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "thermal" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, query):
        return [1.0, 0.0]


def test_dense_hybrid_and_embedding_change(knowledge, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "embedding_model", "synthetic-test-only")
    knowledge.embedder = Embeddings()
    item = knowledge.add("a", "a.md", b"thermal image fusion")
    knowledge.add("a", "b.md", b"language translation")
    assert knowledge.search("a", "thermal")[0]["document_id"] == item["document_id"]
    assert knowledge.search("a", "infrared", mode="dense")[0]["document_id"] == item["document_id"]
    monkeypatch.setattr(settings, "embedding_model", "changed")
    with pytest.raises(ValueError, match="re-ingest"):
        knowledge.search("a", "thermal")


def sample_evidence():
    text, source = "Thermal fusion study", "document://test#page=1"
    return Evidence(evidence_id=evidence_id(source, text), source_type="document",
                    title="test", source=source, text=text).model_dump()


def test_context_budget_and_citation_failure():
    item = sample_evidence()
    assert context_for([item], 1) == ("", [])
    assert context_for([item], 5000)[1] == [item["evidence_id"]]
    state = {"report": {"summary": "Claim [ev_fake]", "sections": [], "citations": ["ev_fake"]},
             "available_citations": [item["evidence_id"]]}
    assert output_guard_node(state)["_needs_repair"]
    with pytest.raises(CitationValidationError):
        output_guard_node({**state, "repairs": settings.max_output_repairs})
    state["report"] = {"summary": f"Claim [{item['evidence_id']}]", "sections": [],
                       "citations": [item["evidence_id"]]}
    assert not output_guard_node(state)["_needs_repair"]


def test_output_guard_rejects_unlinked_evidence_index():
    evidence_id = "ev_1234567890abcdef1234"
    state = {
        "report": {
            "summary": "Supported claim without an inline marker.",
            "sections": [],
            "citations": [evidence_id],
            "limitations": [],
        },
        "available_citations": [evidence_id],
        "repairs": settings.max_output_repairs,
    }
    with pytest.raises(CitationValidationError):
        output_guard_node(state)


def test_output_guard_checks_claim_locations_and_limitation_citations():
    first, second = "ev_aaaaaaaaaaaaaaaaaaaa", "ev_bbbbbbbbbbbbbbbbbbbb"
    state = {
        "report": {
            "summary": "A factual claim without a marker.",
            "sections": ["A heading", "Another heading"],
            "citations": [first],
            "limitations": [f"Evidence gap [{first}]"],
        },
        "available_citations": [first, second],
    }
    repair = output_guard_node(state)
    assert repair["_needs_repair"]
    assert "Summary must cite retrieved evidence" in repair["guardrail_flags"]
    assert "Report sections contain no cited findings" in repair["guardrail_flags"]
    state["report"] = {
        "summary": f"Supported summary [{first}]",
        "sections": [f"Supported detail [{second}]"],
        "citations": [first, second],
        "limitations": [f"Evidence gap [{first}]"],
    }
    assert not output_guard_node(state)["_needs_repair"]


def test_output_guard_derives_citation_list_from_valid_inline_ids():
    first, second = "ev_aaaaaaaaaaaaaaaaaaaa", "ev_bbbbbbbbbbbbbbbbbbbb"
    report = {
        "summary": f"First fact [{first}] and second fact [{second}]",
        "sections": [f"Both are discussed [{first}] [{second}]"],
        "citations": [first],
        "limitations": [],
    }
    result = output_guard_node({"report": report, "available_citations": [first, second]})
    assert not result["_needs_repair"]
    assert result["report"]["citations"] == [first, second]
    assert report["citations"] == [first]
    report["summary"] = "Invented source [ev_unknown]"
    assert output_guard_node({"report": report, "available_citations": [first, second]})[
        "_needs_repair"
    ]


def test_empty_evidence_never_uses_model():
    result = synthesizer_node({"query": "q"})
    assert result["report"]["citations"] == []
    assert result["report"]["limitations"]
    assert output_guard_node(result)["guardrail_flags"] == ["insufficient_evidence"]


def test_real_stdio_mcp_discovery_and_owner_scope(knowledge):
    document = knowledge.add("alice", "a.md", b"thermal fusion")
    result = asyncio.run(call_knowledge("alice", "search_documents", {"query": "thermal"}))
    assert result["evidence"][0]["document_id"] == document["document_id"]
    result = asyncio.run(call_knowledge("bob", "search_documents", {"query": "thermal"}))
    assert result["evidence"] == []


class ScriptedChat(BaseChatModel):
    """Synthetic responses with real LangChain tool parsing and callbacks."""
    plan_count: int = 1
    @property
    def _llm_type(self):
        return "offline-scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        name = kwargs["tools"][0]["function"]["name"]
        if name == "InputVerdict":
            args = {"allowed": True, "reason": "academic"}
        elif name == "Plan":
            args = {"subtasks": ["thermal fusion"] * self.plan_count}
        elif name == "Assessment":
            args = {"sufficient": True, "missing": []}
        elif name == "ResearchReport":
            source = re.search(r'"evidence_id": "(ev_[a-f0-9]+)"', str(messages[-1].content)).group(1)
            args = {"summary": f"Thermal fusion uses complementary sensors [{source}]",
                    "sections": [], "citations": [source], "limitations": ["Synthetic fixture"]}
        elif isinstance(messages[-1], ToolMessage):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Retrieved evidence."))])
        else:
            args = {"query": "thermal fusion"}
        message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_fixture"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def scripted(monkeypatch):
    from research_agent.nodes import planner, critic, synthesizer, guardrails, source_researcher
    model = ScriptedChat()
    for module in (planner, critic, synthesizer, guardrails, source_researcher):
        monkeypatch.setattr(module, "get_model", lambda role: model)
    monkeypatch.setattr(settings, "hitl_approval_enabled", False)
    monkeypatch.setattr(settings, "guardrail_input_enabled", True)
    monkeypatch.setattr(settings, "guardrail_output_enabled", True)
    return model


def test_real_graph_tool_loop_citations_and_callbacks(knowledge, scripted):
    knowledge.add("alice", "a.md", b"Thermal fusion combines complementary sensors.")
    observer = RunObserver(20, 60)
    graph = build_graph()
    result = graph.invoke({"query": "thermal fusion?", "owner": "alice", "mode": "local"},
                          config={"callbacks": [observer]})
    assert result["report"]["citations"] == [result["evidence"][0]["evidence_id"]]
    assert result["guardrail_flags"] == []
    assert observer.snapshot()["model_calls"] == 6
    assert observer.snapshot()["tool_calls"] == 1


def test_shared_callback_budget_stops_real_graph(knowledge, scripted):
    knowledge.add("alice", "a.md", b"thermal fusion")
    observer = RunObserver(2, 60)
    with pytest.raises(BudgetExceeded):
        build_graph().invoke({"query": "thermal", "owner": "alice", "mode": "local"},
                             config={"callbacks": [observer]})
    assert observer.snapshot()["model_calls"] == 2


def test_api_auth_documents_tasks_and_resume(knowledge, scripted, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "api_keys", {"alice": "test-alice", "bob": "test-bob"})
    monkeypatch.setattr(settings, "allow_anonymous", False)
    app = create_api(build_graph(checkpointer=InMemorySaver()), job_db_path=str(tmp_path / "jobs.db"))
    alice, bob = {"X-API-Key": "test-alice"}, {"X-API-Key": "test-bob"}
    with TestClient(app) as client:
        assert client.get("/documents").status_code == 401
        document = client.post("/documents", headers=alice,
                    files={"file": ("a.md", b"thermal fusion complementary sensors", "text/markdown")}).json()
        assert client.get("/documents", headers=bob).json() == []
        denied = client.post("/research", headers=bob,
                    json={"query": "thermal", "document_ids": [document["document_id"]]})
        assert denied.status_code == 404
        task = client.post("/research", headers=alice,
                           json={"query": "thermal fusion", "require_approval": True}).json()["thread_id"]
    with TestClient(app) as client:
        state = client.get(f"/research/{task}", headers=alice).json()
        assert state["status"] == "interrupted"
        assert client.get(f"/research/{task}", headers=bob).status_code == 404
        assert client.post(f"/research/{task}/resume", headers=bob,
                           json={"action": "approve"}).status_code == 404
        assert client.post(f"/research/{task}/resume", headers=alice,
                           json={"action": "approve"}).status_code == 202
    with TestClient(app) as client:
        state = client.get(f"/research/{task}", headers=alice).json()
        assert state["status"] == "completed"
        assert state["report"]["citations"]
        assert state["metrics"]["model_calls"] == 6


def test_rate_limit_is_principal_scoped(knowledge, monkeypatch):
    monkeypatch.setattr(settings, "requests_per_minute", 1)
    with TestClient(create_api(graph=SimpleNamespace())) as client:
        assert client.get("/documents").status_code == 200
        assert client.get("/documents").status_code == 429
        assert client.get("/health").status_code == 200


def test_parallel_research_merges_duplicate_sources(knowledge, scripted):
    scripted.plan_count = 2
    knowledge.add("alice", "a.md", b"Thermal fusion combines complementary sensors.")
    observer = RunObserver(20, 60)
    result = build_graph().invoke({"query": "thermal", "owner": "alice", "mode": "local"},
                                 config={"callbacks": [observer], "max_concurrency": 2})
    assert len(result["findings"]) == 2
    assert len(result["evidence"]) == 1
    assert observer.snapshot()["model_calls"] == 8
    assert any(e["kind"] == "node_start" and e["name"] == "planner"
               for e in observer.snapshot()["events"])


def test_api_body_limit_and_home(knowledge, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 1000)
    with TestClient(create_api(graph=SimpleNamespace())) as client:
        assert "学术资料调研助手" in client.get("/").text
        response = client.post("/documents", content=b"x" * 70000,
                               headers={"content-type": "application/octet-stream"})
        assert response.status_code == 413
        response = client.post("/documents", files={"file": ("a.md", b"x" * 1001)})
        assert response.status_code == 400


def test_web_agent_captures_tool_sources(knowledge, scripted, monkeypatch):
    from research_agent.nodes import source_researcher
    fake = SimpleNamespace(invoke=lambda args: {"results": [
        {"url": "https://example.org/paper", "title": "Fixture", "content": "Thermal fusion"},
        {"url": "file:///private", "content": "must not be retained"},
    ]})
    monkeypatch.setattr(source_researcher, "_get_search", lambda: fake)
    result = source_researcher.researcher_node({"subtasks": ["thermal"], "mode": "web"})
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["source"] == "https://example.org/paper"
    assert result["evidence"][0]["source_type"] == "web"


class RepeatingToolChat(BaseChatModel):
    """A misbehaving provider that never voluntarily stops calling a tool."""

    @property
    def _llm_type(self):
        return "offline-repeating-tool"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        name = kwargs["tools"][0]["function"]["name"]
        message = AIMessage(
            content="",
            tool_calls=[{"name": name, "args": {"query": "thermal"}, "id": "repeat"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class FailingResearchChat(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-failing-researcher"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("synthetic provider failure")


def test_researcher_bounds_repeating_tool_agent(knowledge, monkeypatch):
    from research_agent.nodes import source_researcher

    calls = {"count": 0}

    def search(_args):
        calls["count"] += 1
        return {"results": [
            {"url": "https://example.org/paper", "title": "Fixture", "content": "Thermal"}
        ]}

    monkeypatch.setattr(source_researcher, "get_model", lambda _role: RepeatingToolChat())
    monkeypatch.setattr(source_researcher, "_get_search", lambda: SimpleNamespace(invoke=search))
    result = source_researcher.researcher_node({"subtasks": ["thermal"], "mode": "web"})

    assert result["llm_calls"] == settings.max_researcher_model_calls
    assert calls["count"] <= 2
    assert len(result["evidence"]) == 1


def test_researcher_reports_a_failed_provider_call(knowledge, monkeypatch):
    from research_agent.nodes import source_researcher

    monkeypatch.setattr(source_researcher, "get_model", lambda _role: FailingResearchChat())
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        source_researcher.researcher_node({"subtasks": ["thermal"], "mode": "web"})


def test_embedding_client_sends_strings_to_compatible_provider(monkeypatch):
    import langchain_openai
    from research_agent.knowledge import embedding_client

    observed = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(settings, "embedding_model", "compatible-embedding")
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", FakeEmbeddings)
    assert isinstance(embedding_client(), FakeEmbeddings)
    assert observed["check_embedding_ctx_length"] is False


def test_retrieval_fixture_metrics_are_reproducible():
    from evals.retrieval import evaluate
    result = evaluate(3)
    assert result["cases"] == 14
    assert result["document_recall_at_k"] >= 0.9
    assert result["empty_query_accuracy"] == 1.0


def test_public_search_has_timeout_and_response_contract(monkeypatch):
    from research_agent.tools.search import SearchClient
    original = httpx.Client
    observed = {}

    def handler(request):
        assert request.url == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer synthetic-key"
        observed["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json={"results": []})

    monkeypatch.setenv("TAVILY_API_KEY", "synthetic-key")
    monkeypatch.setattr(httpx, "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    assert SearchClient().invoke({"query": "thermal"}) == {"results": []}
    assert observed["timeout"]["connect"] == 10.0
    assert observed["timeout"]["read"] == 30.0


def test_default_api_lifespan_without_provider_calls(knowledge, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "job_db_path", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(settings, "checkpoint_db_path", str(tmp_path / "state" / "checkpoint.db"))
    with TestClient(create_api()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/documents").json() == []


def test_hybrid_context_preserves_web_evidence_and_parallel_order(monkeypatch):
    monkeypatch.setattr(settings, "max_evidence", 6)
    local = [Evidence(evidence_id=f"ev_local{i}", source_type="document", document_id="paper",
                      title="paper", source=f"document://paper#{i}", text="local " * 10,
                      score=1.0).model_dump() for i in range(10)]
    web = Evidence(evidence_id="ev_web", source_type="web", title="web",
                   source="https://example.org", text="web " * 10).model_dump()
    first = merge_evidence(merge_evidence([], local), [web])
    second = merge_evidence(merge_evidence([], [web]), local)
    assert first == second
    _, visible = context_for(first, 800)
    assert "ev_web" in visible
