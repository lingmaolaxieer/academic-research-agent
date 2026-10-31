"""Celery entrypoints for durable, horizontally scalable research execution."""
from __future__ import annotations

import logging
import socket

from celery import Celery, signals
from prometheus_client import start_http_server
from langgraph.types import Command

from research_agent.config import settings
from research_agent.execution import run_graph
from research_agent.graph import build_graph
from research_agent.jobs import create_job_registry
from research_agent.logging_setup import configure_logging
from research_agent.persistence import checkpoint_saver
from research_agent.rerank import warm_reranker

logger = logging.getLogger(__name__)
celery_app = Celery("academic_research", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_default_queue=settings.celery_queue,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


@signals.after_setup_logger.connect
def setup_worker_logging(**kwargs):
    configure_logging(settings.log_format)


@signals.worker_ready.connect
def start_worker_metrics(**kwargs):
    start_http_server(settings.worker_metrics_port)
    try:
        warm_reranker()
        if settings.reranker_model:
            logger.info("reranker_warmed model=%s", settings.reranker_model)
    except Exception:
        logger.exception("reranker_warmup_failed")


def run_config(thread_id: str) -> dict:
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


@celery_app.task(name="research.execute", bind=True)
def execute_research(self, thread_id: str, inputs: dict | None = None,
                     resume: dict | None = None, recover: bool = False):
    registry = create_job_registry()
    token = registry.claim_execution(thread_id, f"{socket.gethostname()}:{self.request.id}")
    if token is None:
        job = registry.get(thread_id)
        registry.close()
        if job and job["status"] == "running":
            raise self.retry(
                countdown=max(10, settings.worker_lease_seconds // 3), max_retries=24
            )
        return {"thread_id": thread_id, "status": "duplicate_ignored"}
    try:
        with checkpoint_saver() as saver:
            graph = build_graph(checkpointer=saver)
            if recover:
                snapshot = graph.get_state(run_config(thread_id))
                if snapshot.values:
                    pending_approval = any(
                        getattr(task, "interrupts", ())
                        for task in (getattr(snapshot, "tasks", ()) or ())
                    )
                    payload = Command(resume=resume) if pending_approval and resume else None
                elif inputs is not None:
                    payload = inputs
                else:
                    raise RuntimeError("Recovery checkpoint is missing")
            else:
                payload = Command(resume=resume) if resume is not None else inputs
            run_graph(graph, registry, thread_id, payload, run_config(thread_id), token)
        return {"thread_id": thread_id, "status": (registry.get(thread_id) or {}).get("status")}
    except Exception as exc:
        from research_agent.execution import public_error
        registry.fail(thread_id, public_error(exc), token)
        return {"thread_id": thread_id, "status": "error"}
    finally:
        registry.close()
