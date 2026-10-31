"""Transactional-outbox dispatcher from PostgreSQL to the Celery broker."""
from __future__ import annotations

import logging
import signal
import time
from prometheus_client import start_http_server

from research_agent.jobs import PostgresJobRegistry
from research_agent.logging_setup import configure_logging
from research_agent.tasks import celery_app
from research_agent.config import settings
from research_agent.telemetry import OUTBOX

logger = logging.getLogger(__name__)
stopping = False


def _stop(*args):
    global stopping
    stopping = True


def dispatch_once(registry: PostgresJobRegistry) -> int:
    rows = registry.claim_outbox()
    for row in rows:
        payload = row["payload"]
        try:
            celery_app.send_task(
                "research.execute",
                kwargs={
                    "thread_id": row["thread_id"],
                    "inputs": payload.get("inputs"),
                    "resume": payload.get("resume"),
                    "recover": payload.get("recover", False),
                },
                task_id=row["message_id"],
                queue=settings.celery_queue,
            )
            registry.mark_outbox_sent(row["id"], row["publish_token"])
            OUTBOX.labels(result="published").inc()
        except Exception as exc:
            registry.retry_outbox(row["id"], row["attempts"], row["publish_token"])
            OUTBOX.labels(result="retry").inc()
            logger.warning("outbox_publish_failed error_type=%s", type(exc).__name__)
    return len(rows)


def main():
    configure_logging(settings.log_format)
    start_http_server(settings.outbox_metrics_port)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    registry = PostgresJobRegistry(settings.database_url)
    try:
        last_recovery = 0.0
        while not stopping:
            if time.monotonic() - last_recovery >= 10:
                registry.recover_stale()
                last_recovery = time.monotonic()
            if dispatch_once(registry) == 0:
                time.sleep(1)
    finally:
        registry.close()


if __name__ == "__main__":
    main()
