"""Low-cardinality Prometheus metrics for the API and graph workers."""
from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "research_http_requests_total", "HTTP requests", ("method", "route", "status")
)
HTTP_DURATION = Histogram(
    "research_http_request_duration_seconds", "HTTP request latency", ("method", "route")
)
MODEL_CALLS = Counter("research_model_calls_total", "Chat model calls")
TOOL_CALLS = Counter("research_tool_calls_total", "Agent tool calls", ("tool",))
RUNS = Counter("research_runs_total", "Research runs by terminal result", ("result",))
RUN_DURATION = Histogram("research_run_duration_seconds", "Active graph execution time")
OUTBOX = Counter("research_outbox_messages_total", "Outbox publish outcomes", ("result",))
