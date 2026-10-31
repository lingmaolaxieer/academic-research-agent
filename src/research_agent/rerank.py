"""Optional Cross-Encoder reranking after broad hybrid retrieval."""
from __future__ import annotations

from functools import lru_cache

from research_agent.config import settings


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - depends on optional production extra
        raise RuntimeError(
            "RERANKER_MODEL requires the project's 'rerank' optional dependency"
        ) from exc
    return CrossEncoder(model_name)


def warm_reranker() -> None:
    """Load configured weights before the worker accepts its first research task."""
    if settings.reranker_model:
        _cross_encoder(settings.reranker_model)


def rerank(query: str, evidence: list[dict], model=None, top_k: int | None = None) -> list[dict]:
    """Score query/excerpt pairs and return a deterministic best-first list.

    Retrieval remains usable when no model is configured. Once configured, a
    missing model is a configuration error rather than a silent quality change.
    """
    if not evidence or not settings.reranker_model and model is None:
        return evidence[:top_k] if top_k is not None else evidence
    encoder = model or _cross_encoder(settings.reranker_model)
    candidates = evidence[: settings.reranker_candidates]
    scores = encoder.predict([(query, item["text"]) for item in candidates])
    if len(scores) != len(candidates):
        raise ValueError("Reranker response size mismatch")
    ranked = []
    for item, score in zip(candidates, scores):
        updated = dict(item)
        updated["score"] = float(score)
        ranked.append(updated)
    ranked.sort(key=lambda item: (-item["score"], item["evidence_id"]))
    return ranked[:top_k] if top_k is not None else ranked
