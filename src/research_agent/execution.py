"""Shared graph execution used by the local executor and Celery workers."""
from __future__ import annotations

import logging
import threading

from research_agent.config import settings
from research_agent.jobs import LeaseLost
from research_agent.observation import RunObserver, RunCancelled
from research_agent.telemetry import RUN_DURATION, RUNS

logger = logging.getLogger(__name__)


def public_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: research failed; check service configuration and trace metadata."


class LeaseHeartbeat:
    def __init__(self, registry, thread_id: str, token: str | None):
        self.registry, self.thread_id, self.token = registry, thread_id, token
        self.stop_event = threading.Event()
        self.lost = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.token:
            interval = max(10, settings.worker_lease_seconds // 3)
            self.thread = threading.Thread(target=self._run, args=(interval,), daemon=True)
            self.thread.start()
        return self

    def _run(self, interval):
        while not self.stop_event.wait(interval):
            if not self.registry.renew_lease(self.thread_id, self.token):
                self.lost.set()
                logger.error("worker_lease_lost thread_id=%s", self.thread_id)
                return

    def __exit__(self, *args):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)


def run_graph(graph, registry, thread_id: str, inputs, run_config: dict,
              lease_token: str | None = None) -> None:
    logger.info("research_started thread_id=%s", thread_id)
    prior = registry.get(thread_id) or {}
    observer = RunObserver(
        settings.max_model_calls,
        settings.task_timeout_seconds,
        prior.get("metrics", {}).get("model_calls", 0),
        event_sink=lambda kind, event: registry.append_event(
            thread_id, kind, event, lease_token
        ),
        cancel_check=lambda: registry.is_cancelled(thread_id),
    )
    config = dict(run_config)
    config["callbacks"] = [observer]
    try:
        with LeaseHeartbeat(registry, thread_id, lease_token) as heartbeat:
            result = graph.invoke(inputs, config=config)
        if heartbeat.lost.is_set():
            raise LeaseLost("Task execution lease was lost during graph execution")
        if registry.is_cancelled(thread_id):
            RUNS.labels(result="cancelled").inc()
            return
        registry.assert_execution(thread_id, lease_token)
        snapshot = observer.snapshot()
        registry.set_metrics(thread_id, snapshot, lease_token)
        RUN_DURATION.observe(snapshot["elapsed_s"])
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            if registry.interrupt(thread_id, interrupts[0].value, lease_token):
                RUNS.labels(result="interrupted").inc()
        else:
            if registry.finish(thread_id, lease_token):
                RUNS.labels(result="completed").inc()
    except RunCancelled:
        RUNS.labels(result="cancelled").inc()
        logger.info("research_cancelled thread_id=%s", thread_id)
    except LeaseLost:
        logger.info("research_stale_worker_ignored thread_id=%s", thread_id)
    except Exception as exc:
        if registry.is_cancelled(thread_id):
            RUNS.labels(result="cancelled").inc()
            return
        try:
            registry.assert_execution(thread_id, lease_token)
        except LeaseLost:
            logger.info("research_stale_worker_ignored thread_id=%s", thread_id)
            return
        snapshot = observer.snapshot()
        registry.set_metrics(thread_id, snapshot, lease_token)
        RUN_DURATION.observe(snapshot["elapsed_s"])
        if registry.fail(thread_id, public_error(exc), lease_token):
            RUNS.labels(result="error").inc()
        logger.warning("research_failed thread_id=%s error_type=%s", thread_id, type(exc).__name__)
