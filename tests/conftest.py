"""Keep offline tests independent from a developer's deployment `.env` file."""

import pytest

from research_agent.config import settings


@pytest.fixture(autouse=True)
def local_test_profile(monkeypatch):
    monkeypatch.setattr(settings, "runtime_profile", "local")
    monkeypatch.setattr(settings, "task_backend", "thread")
    monkeypatch.setattr(settings, "rate_limit_backend", "memory")
    monkeypatch.setattr(settings, "api_keys", {})
    monkeypatch.setattr(settings, "allow_anonymous", True)
    monkeypatch.setattr(settings, "knowledge_transport", "local")
    monkeypatch.setattr(settings, "retrieval_mode", "bm25")
    monkeypatch.setattr(settings, "embedding_model", "")
    monkeypatch.setattr(settings, "reranker_model", "")
