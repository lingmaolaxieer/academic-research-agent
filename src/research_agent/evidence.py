"""Validated source excerpts; report references use IDs, never invented URLs."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Literal

from pydantic import BaseModel, Field
from research_agent.config import settings


class Evidence(BaseModel):
    evidence_id: str
    source_type: Literal["document", "web"]
    title: str
    source: str
    text: str
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    score: float = 0.0


def evidence_id(source: str, text: str) -> str:
    return "ev_" + hashlib.sha256((source + "\0" + text).encode()).hexdigest()[:20]


def diversify(evidence):
    """Round-robin sources so many chunks from one document do not hide all others."""
    groups = {}
    for item in sorted(evidence, key=lambda x: (-x.get("score", 0), x["evidence_id"])):
        group = item.get("document_id") or item["source"]
        groups.setdefault(group, deque()).append(item)
    queues = deque(groups.values())
    while queues:
        queue = queues.popleft()
        yield queue.popleft()
        if queue:
            queues.append(queue)


def merge_evidence(existing: list[dict], incoming: list[dict]) -> list[dict]:
    # Deterministic ordering despite parallel branches. Bound total context separately.
    by_id = {item["evidence_id"]: item for item in existing}
    for item in incoming:
        checked = Evidence.model_validate(item).model_dump()
        key = checked["evidence_id"]
        old = by_id.get(key)
        if old is None or checked["score"] > old.get("score", 0):
            by_id[key] = checked
    # Fixed per-type quotas make top-k reduction independent of branch arrival
    # order and prevent document scores from evicting every web excerpt.
    ordered = sorted(by_id.values(), key=lambda x: (-x["score"], x["evidence_id"]))
    local_limit = (settings.max_evidence + 1) // 2
    web_limit = settings.max_evidence // 2
    return ([item for item in ordered if item["source_type"] == "document"][:local_limit]
            + [item for item in ordered if item["source_type"] == "web"][:web_limit])


def context_for(evidence: list[dict], limit: int) -> tuple[str, list[str]]:
    """Only complete excerpts within the context budget are visible/citable."""
    blocks, ids = [], []
    used = 0
    for item in diversify(evidence):
        block = json.dumps(item, ensure_ascii=False)
        if used + len(block) + 1 > limit:
            continue
        blocks.append(block)
        ids.append(item["evidence_id"])
        used += len(block) + 1
    return "\n".join(blocks), ids


class ResearchReport(BaseModel):
    summary: str = Field(
        description="A concise answer with inline [ev_ID] citations after factual claims"
    )
    sections: list[str] = Field(
        description="Substantive explanatory paragraphs, not bare headings; each paragraph cites "
                    "its evidence inline as [ev_ID]"
    )
    citations: list[str] = Field(
        description="Unique evidence_id values actually used inline in summary, sections "
                    "or limitations; never list uncited IDs"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Brief evidence gaps and uncertainty, with inline citations when factual"
    )
