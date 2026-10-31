"""Per-run callbacks enforce shared call/deadline limits across parallel nodes."""
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler
from research_agent.telemetry import MODEL_CALLS, TOOL_CALLS


class BudgetExceeded(RuntimeError):
    pass


class RunCancelled(RuntimeError):
    pass


class RunObserver(BaseCallbackHandler):
    raise_error = True
    run_inline = True

    def __init__(self, max_calls: int, timeout: int, initial_calls: int = 0,
                 event_sink=None, cancel_check=None):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.deadline = self.started + timeout
        self.max_calls = max_calls
        self.calls = initial_calls
        self.tool_calls = 0
        self.events = []
        self.node_runs = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_records = 0
        self.event_sink = event_sink
        self.cancel_check = cancel_check

    def check(self):
        if self.cancel_check and self.cancel_check():
            raise RunCancelled("Research task was cancelled")
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("Research active-time deadline exceeded")

    def _model_start(self):
        with self.lock:
            self.check()
            if self.calls >= self.max_calls:
                raise BudgetExceeded("Research model-call budget exceeded")
            self.calls += 1
            MODEL_CALLS.inc()
            self._event("model_start")

    def _event(self, kind, **extra):
        event = {"kind": kind, "elapsed_s": round(time.monotonic() - self.started, 3), **extra}
        if len(self.events) < 200:
            self.events.append(event)
        if self.event_sink:
            self.event_sink(kind, event)

    def on_llm_start(self, *args, **kwargs):
        self._model_start()

    def on_chat_model_start(self, *args, **kwargs):
        self._model_start()

    def on_tool_start(self, serialized, *args, **kwargs):
        with self.lock:
            self.check()
            self.tool_calls += 1
            name = (serialized or {}).get("name", "unknown")
            TOOL_CALLS.labels(tool=name).inc()
            self._event("tool_start", name=name)

    def on_llm_error(self, error, **kwargs):
        with self.lock:
            self._event("model_error", error_type=type(error).__name__)

    def on_llm_end(self, response, **kwargs):
        with self.lock:
            for generations in response.generations:
                for generation in generations:
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                    if usage:
                        self.input_tokens += usage.get("input_tokens", 0)
                        self.output_tokens += usage.get("output_tokens", 0)
                        self.usage_records += 1

    def on_chain_start(self, serialized, inputs, *, run_id, name=None, **kwargs):
        if name in {"input_guard", "router", "quick_plan", "planner", "approval",
                    "researcher", "critic", "synthesizer", "output_guard"}:
            with self.lock:
                self.check()
                self.node_runs[run_id] = name
                self._event("node_start", name=name)

    def _chain_finished(self, run_id, error=None):
        with self.lock:
            name = self.node_runs.pop(run_id, None)
            if name:
                self._event("node_error" if error else "node_end", name=name,
                            **({"error_type": type(error).__name__} if error else {}))

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._chain_finished(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._chain_finished(run_id, error)

    def snapshot(self):
        with self.lock:
            return {"model_calls": self.calls, "tool_calls": self.tool_calls,
                    "elapsed_s": round(time.monotonic() - self.started, 3), "events": list(self.events),
                    "segment_input_tokens": self.input_tokens if self.usage_records else None,
                    "segment_output_tokens": self.output_tokens if self.usage_records else None,
                    "usage_records": self.usage_records}
