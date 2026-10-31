"""Single-process API: background execution, durable status and real HITL.

202 means accepted, not successful. Drain workers before closing checkpoints.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from research_agent.config import settings
from research_agent.graph import build_graph
from research_agent.execution import run_graph
from research_agent.jobs import (
    IdempotencyConflict, JobRegistry, QueueCapacityReached, create_job_registry,
)
from research_agent.persistence import checkpoint_saver
from research_agent.document_api import install_document_api
from research_agent.knowledge import create_knowledge_store
from research_agent.logging_setup import configure_logging
from research_agent.security import install_security
from research_agent.telemetry import HTTP_DURATION, HTTP_REQUESTS
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ResearchRequest(BaseModel):
    query: NonemptyText
    mode: Literal["auto", "local", "web", "hybrid"] = "auto"
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    require_approval: bool | None = None
    depth: Literal["auto", "quick", "deep"] = "auto"


class ResearchAccepted(BaseModel):
    thread_id: str
    status: str


class Report(BaseModel):
    summary: str
    sections: list[str]
    citations: list[str]
    limitations: list[str] = Field(default_factory=list)


class ResearchStatus(BaseModel):
    thread_id: str
    status: str
    report: Report | None = None
    guardrail_flags: list[str] = Field(default_factory=list)
    llm_calls: int | None = None
    error: str | None = None
    interrupt: dict | None = None
    evidence: list[dict] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    source_errors: list[str] = Field(default_factory=list)


class ResumeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["approve", "edit", "reject", "abort"]
    subtasks: list[NonemptyText] | None = Field(
        default=None, min_length=1, max_length=settings.max_research_subtasks
    )
    feedback: NonemptyText | None = None
    reason: NonemptyText | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "edit" and not self.subtasks:
            raise ValueError(
                f"edit requires 1-{settings.max_research_subtasks} nonempty subtasks"
            )
        if self.action == "reject" and not self.feedback:
            raise ValueError("reject requires feedback")
        allowed = {"approve": set(), "edit": {"subtasks"}, "reject": {"feedback"}, "abort": {"reason"}}
        for name in ("subtasks", "feedback", "reason"):
            if getattr(self, name) is not None and name not in allowed[self.action]:
                raise ValueError(f"{name} is not valid for {self.action}")
        return self


def _run_config(thread_id: str) -> RunnableConfig:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.graph_recursion_limit,
        "max_concurrency": settings.max_workers,
        "run_name": "academic-research",
        "metadata": {
            "model_planner": settings.model_planner,
            "model_researcher": settings.model_researcher,
            "model_synthesizer": settings.model_synthesizer,
            "model_critic": settings.model_critic,
        },
    }


def _pending_interrupt(snapshot) -> dict | None:
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0].value
    return None


def _read_status(graph, registry: JobRegistry, thread_id: str) -> ResearchStatus:
    snapshot = graph.get_state(_run_config(thread_id))
    values = snapshot.values or {}
    job = registry.get(thread_id)
    if job is not None:
        status, error = job["status"], job["error"]
        payload = job.get("interrupt")
    elif values:
        payload = _pending_interrupt(snapshot)
        error = None
        if payload is not None:
            status = "interrupted"
        elif not snapshot.next:
            status = "completed"
        else:
            status = "error"
            error = "Execution stopped without a pending human approval; automatic replay is disabled."
    else:
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")
    report = values.get("report") if status != "cancelled" else None
    return ResearchStatus(
        thread_id=thread_id, status=status,
        report=Report.model_validate(report) if report else None,
        guardrail_flags=values.get("guardrail_flags", []),
        llm_calls=values.get("llm_calls"), error=error, interrupt=payload,
        evidence=values.get("evidence", []), metrics=job.get("metrics", {}) if job else {},
        source_errors=values.get("source_errors", []),
    )


def create_api(graph=None, *, job_db_path: str | None = None) -> FastAPI:
    path = job_db_path if job_db_path is not None else (
        ":memory:" if graph is not None else settings.job_db_path
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_format)
        if not settings.allow_anonymous and not settings.api_keys:
            raise ValueError("Configure API_KEYS when anonymous access is disabled")
        if any(not key for key in settings.api_keys.values()) or len(set(settings.api_keys.values())) != len(settings.api_keys):
            raise ValueError("API keys must be nonempty and unique per owner")
        if settings.task_backend == "celery" and settings.runtime_profile != "production":
            raise ValueError("Celery task backend requires RUNTIME_PROFILE=production")
        app.state.knowledge_path = settings.knowledge_db_path
        app.state.registry = create_job_registry(path)
        app.state.registry.recover_stale()
        app.state.use_celery = settings.task_backend == "celery" and graph is None
        app.state.executor = None
        app.state.capacity = None
        if not app.state.use_celery:
            app.state.executor = ThreadPoolExecutor(
                max_workers=settings.max_workers, thread_name_prefix="research",
            )
            app.state.capacity = threading.BoundedSemaphore(settings.max_pending_jobs)
        try:
            if graph is not None:
                app.state.graph = graph
                try:
                    yield
                finally:
                    if app.state.executor:
                        app.state.executor.shutdown(wait=True)
            else:
                with checkpoint_saver() as saver:
                    app.state.graph = build_graph(checkpointer=saver)
                    try:
                        yield
                    finally:
                        if app.state.executor:
                            app.state.executor.shutdown(wait=True)
        finally:
            app.state.registry.close()

    app = FastAPI(
        title="Academic Research Agent", version="1.0.0",
        summary="Evidence-grounded research with durable multi-agent orchestration.",
        lifespan=lifespan,
    )
    install_security(app)
    install_document_api(app)

    @app.middleware("http")
    async def observe_http(request, call_next):
        started = time.monotonic()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", request.url.path)
        HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, route).observe(time.monotonic() - started)
        return response

    def authorize(request, thread_id):
        job = request.app.state.registry.get(thread_id)
        if job is not None and job["owner"] == request.state.owner:
            return
        if job is None and not settings.api_keys and settings.allow_anonymous:
            return  # Local-only compatibility for older checkpoint files.
        raise HTTPException(404, "Research task not found")

    def reserve(request):
        if request.app.state.use_celery:
            return
        if not request.app.state.capacity.acquire(blocking=False):
            raise HTTPException(429, "Research capacity reached; retry later.")

    def release(request):
        if request.app.state.capacity is not None:
            request.app.state.capacity.release()

    def submit(request, thread_id, inputs=None, resume=None):
        if request.app.state.use_celery:
            return  # Transactional outbox dispatcher publishes the durable message.
        try:
            future = request.app.state.executor.submit(
                run_graph, request.app.state.graph, request.app.state.registry,
                thread_id, Command(resume=resume) if resume is not None else inputs,
                _run_config(thread_id),
            )
        except Exception:
            release(request)
            request.app.state.registry.fail(thread_id, "Task submission failed.")
            raise HTTPException(503, "Unable to submit research task.") from None
        future.add_done_callback(lambda _: release(request))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict:
        if settings.runtime_profile == "production":
            try:
                with request.app.state.registry.connect() as db, db.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                import redis
                redis.Redis.from_url(settings.redis_url).ping()
            except Exception:
                raise HTTPException(503, "Required storage service is unavailable") from None
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post("/research", response_model=ResearchAccepted, status_code=202)
    def start_research(req: ResearchRequest, request: Request) -> ResearchAccepted:
        request_key = request.headers.get("Idempotency-Key")
        if request_key is not None and not re.fullmatch(r"[A-Za-z0-9._~-]{1,128}", request_key):
            raise HTTPException(400, "Invalid Idempotency-Key")
        request_hash = hashlib.sha256(
            json.dumps(req.model_dump(exclude_none=True), sort_keys=True,
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        registry = request.app.state.registry
        if request_key is not None:
            existing = registry.lookup_idempotency(request.state.owner, request_key)
            if existing:
                if existing[1] != request_hash:
                    raise HTTPException(409, "Idempotency-Key was used with another request")
                job = registry.get(existing[0])
                return ResearchAccepted(thread_id=existing[0], status=job["status"])
        knowledge = create_knowledge_store(request.app.state.knowledge_path)
        documents = knowledge.list_documents(request.state.owner)
        owned = {doc["document_id"] for doc in documents}
        if set(req.document_ids) - owned:
            raise HTTPException(404, "Selected document not found")
        mode = req.mode if req.mode != "auto" else ("local" if documents else "web")
        if mode == "local" and not documents:
            raise HTTPException(400, "Upload a document before local research")
        inputs = {"query": req.query, "owner": request.state.owner, "mode": mode,
                  "document_ids": req.document_ids,
                  "research_depth": req.depth,
                  "preferences": knowledge.preferences(request.state.owner)}
        if req.require_approval is not None:
            inputs["require_approval"] = req.require_approval
        reserve(request)
        thread_id = uuid.uuid4().hex
        try:
            initial = "queued" if request.app.state.use_celery else "running"
            outbox = None
            if request.app.state.use_celery:
                outbox = {"message_id": uuid.uuid4().hex, "inputs": inputs, "resume": None}
            if request_key is not None:
                thread_id, created = registry.start_idempotent(
                    thread_id, request.state.owner, initial, request_key,
                    request_hash, outbox=outbox,
                )
                if not created:
                    release(request)
                    job = registry.get(thread_id)
                    return ResearchAccepted(thread_id=thread_id, status=job["status"])
            else:
                registry.start(thread_id, request.state.owner, initial, outbox=outbox)
        except QueueCapacityReached:
            release(request)
            raise HTTPException(429, "Research capacity reached; retry later.") from None
        except IdempotencyConflict:
            release(request)
            raise HTTPException(409, "Idempotency-Key was used with another request") from None
        except Exception:
            release(request)
            raise
        submit(request, thread_id, inputs)
        return ResearchAccepted(thread_id=thread_id, status=initial)

    @app.get("/research")
    def list_research(request: Request):
        return request.app.state.registry.list_jobs(request.state.owner)

    @app.get("/research/{thread_id}", response_model=ResearchStatus)
    def get_research(thread_id: str, request: Request) -> ResearchStatus:
        authorize(request, thread_id)
        return _read_status(request.app.state.graph, request.app.state.registry, thread_id)

    @app.get("/research/{thread_id}/events", include_in_schema=True)
    async def research_events(thread_id: str, request: Request, after: int = 0):
        authorize(request, thread_id)
        header = request.headers.get("last-event-id")
        if header:
            try:
                after = max(after, int(header))
            except ValueError:
                raise HTTPException(400, "Invalid Last-Event-ID") from None

        async def stream():
            cursor = after
            idle = 0
            while not await request.is_disconnected():
                rows = await asyncio.to_thread(
                    request.app.state.registry.events, thread_id, cursor, 100
                )
                if rows:
                    idle = 0
                    for row in rows:
                        cursor = int(row["seq"])
                        data = json.dumps(row["payload"], ensure_ascii=False, default=str)
                        yield f"id: {cursor}\nevent: {row['kind']}\ndata: {data}\n\n"
                else:
                    idle += 1
                    job = await asyncio.to_thread(request.app.state.registry.get, thread_id)
                    if job and job["status"] in {"completed", "error", "interrupted", "cancelled"}:
                        break
                    if idle % 15 == 0:
                        yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    @app.post("/research/{thread_id}/resume", response_model=ResearchAccepted, status_code=202)
    def resume_research(thread_id: str, decision: ResumeDecision, request: Request):
        authorize(request, thread_id)
        current_graph, registry = request.app.state.graph, request.app.state.registry
        job = registry.get(thread_id)
        snapshot = current_graph.get_state(_run_config(thread_id))
        if job is None and not snapshot.values:
            raise HTTPException(404, f"Unknown thread_id: {thread_id}")
        payload = _pending_interrupt(snapshot)
        if payload is None or (job is not None and job["status"] != "interrupted"):
            raise HTTPException(409, "Run is not waiting for human approval.")
        if job is None:
            try:
                registry.start(thread_id)
            except sqlite3.IntegrityError:
                pass
            else:
                registry.interrupt(thread_id, payload)
        reserve(request)
        try:
            target = "queued" if request.app.state.use_celery else "running"
            outbox = None
            if request.app.state.use_celery:
                outbox = {"message_id": uuid.uuid4().hex, "inputs": None,
                          "resume": decision.model_dump(exclude_none=True)}
            claimed = registry.claim_resume(thread_id, target, outbox=outbox)
        except Exception:
            release(request)
            raise
        if not claimed:
            release(request)
            raise HTTPException(409, "Run already resumed by another request.")
        submit(request, thread_id, resume=decision.model_dump(exclude_none=True))
        return ResearchAccepted(thread_id=thread_id, status=target)

    @app.post("/research/{thread_id}/cancel", response_model=ResearchAccepted)
    def cancel_research(thread_id: str, request: Request):
        authorize(request, thread_id)
        registry = request.app.state.registry
        if registry.cancel(thread_id):
            return ResearchAccepted(thread_id=thread_id, status="cancelled")
        job = registry.get(thread_id)
        if job and job["status"] == "cancelled":
            return ResearchAccepted(thread_id=thread_id, status="cancelled")
        raise HTTPException(409, "Research task is already terminal")

    return app


api = create_api()
