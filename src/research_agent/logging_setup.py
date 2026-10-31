"""Structured logs with stable fields and no prompt or credential payloads."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(mode: str) -> None:
    if mode != "json":
        return
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
        root.setLevel(logging.INFO)
    formatter = JsonFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
