"""Robustness + output checks (no graph dependency). 稳健性与输出检查（不依赖图）。

Kept separate from the guardrail *nodes* (``nodes/guardrails.py``) so these pure
helpers stay trivially testable:
这些纯辅助函数易于测试：

- ``RETRY_POLICY``      — a LangGraph retry policy that backs off on transient
                          failures *including 429 rate limits*.
- ``LLMCallCounter``    — a callback that counts model calls, so the researcher
                          sub-agent can report its real cost to the budget guard.
- ``verify_citations``  — the hard citation gate: which cited URLs are NOT backed
                          by a gathered source.
- ``scan_content``      — content checks: leaked system prompts and PII.
- ``scan_content``      —— 内容检查：泄露的系统提示词和 PII。
"""

from __future__ import annotations

import re

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import RetryPolicy

from research_agent.config import settings


# ── Robustness: retry policy ─────────────────────────────────────────────────
# ── 稳健性：重试策略 ─────────────────────────────────────────────────────────
def _is_transient(exc: Exception) -> bool:
    """Retry on rate limits (429), server errors (5xx), and connection drops.
    针对速率限制（429）、服务器错误（5xx）和连接断开进行重试。

    The provider SDKs don't share a base class, so we match structurally:
    anything carrying a 429 / 5xx status code, plus any error whose name screams
    "rate limit" or "timeout". Deterministic bugs (ValueError, TypeError, …) are
    deliberately *not* retried — retrying them just wastes money.
    服务商 SDK 没有共享的基类，因此我们按结构匹配：任何携带 429/5xx 状态码的错误，
    以及名称明显表示“速率限制”或“超时”的错误。确定性错误（ValueError、TypeError 等）
    会被有意*不重试*——重试只会浪费成本。
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "timeout" in name


# One policy, reused on every LLM node in the graph. `initial_interval` doubles
# by `backoff_factor` each attempt → 1s, 2s, 4s, 8s, exactly the schedule the
# 图中的每个 LLM 节点都复用同一策略。`initial_interval` 按 `backoff_factor`
# 在每次尝试时翻倍 → 1 秒、2 秒、4 秒、8 秒，正好是该计划的时间表。
RETRY_POLICY = RetryPolicy(
    max_attempts=settings.retry_max_attempts,
    initial_interval=settings.retry_initial_interval,
    backoff_factor=settings.retry_backoff_factor,
    retry_on=_is_transient,
)


# ── Budget: count real LLM calls ─────────────────────────────────────────────
# ── 预算：统计真实 LLM 调用 ─────────────────────────────────────────────────
class LLMCallCounter(BaseCallbackHandler):
    """Counts model invocations within a single ``invoke``.
    统计单次 ``invoke`` 中的模型调用次数。

    Plain LLMs fire ``on_llm_start``; chat models fire ``on_chat_model_start`` —
    a given call triggers exactly one of them, so counting both is safe and never
    double-counts. Used by the researcher node, whose internal tool-loop makes an
    unknown number of calls, to report its true cost to the budget guard.
    普通 LLM 会触发 ``on_llm_start``；聊天模型会触发 ``on_chat_model_start``——
    每次调用只会触发其中一个，因此统计二者是安全的，不会重复计数。该回调供
    researcher 节点使用，其内部工具循环的调用次数未知，可将真实成本报告给预算守卫。
    """

    def __init__(self) -> None:
        self.count = 0

    def on_llm_start(self, *args, **kwargs) -> None:
        self.count += 1

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.count += 1


# ── Output guardrail: citation verification ──────────────────────────────────
# ── 输出守卫：引用验证 ───────────────────────────────────────────────────────
def _findings_text(findings) -> str:
    """Flatten accumulated findings (strings or Finding objects) into one blob.
    将累积的发现（字符串或 Finding 对象）展平为一个文本块。"""
    parts = []
    for f in findings or []:
        parts.append(f if isinstance(f, str) else str(getattr(f, "model_dump", lambda: f)()))
    return "\n\n".join(parts)


def verify_citations(findings, report) -> tuple[float, list[str]]:
    """Hard citation gate.
    严格的引用门槛。

    Returns ``(coverage, unsupported)`` where ``coverage`` is the fraction of the
    report's cited URLs that actually appear in the gathered findings, and
    ``unsupported`` lists the URLs that do NOT — i.e. citations the model may have
    invented. A report with no citations scores 0.0: an ungrounded report should
    fail the gate.
    返回 ``(coverage, unsupported)``，其中 ``coverage`` 是报告引用的 URL 中
    实际出现在已收集发现里的比例，``unsupported`` 列出没有出现的 URL——也就是
    模型可能编造的引用。没有引用的报告得分为 0.0：缺乏依据的报告应通过不了门槛。
    """
    citations = report.get("citations", []) if isinstance(report, dict) else []
    if not citations:
        return 0.0, []
    haystack = _findings_text(findings)
    unsupported = [url for url in citations if not url or url not in haystack]
    coverage = (len(citations) - len(unsupported)) / len(citations)
    return coverage, unsupported


# ── Output guardrail: content checks ─────────────────────────────────────────
# ── 输出守卫：内容检查 ───────────────────────────────────────────────────────
# Markers that would mean our own instructions leaked into the user-facing text.
# 表明我们自己的指令泄露到面向用户文本中的标记。
_PROMPT_LEAK_MARKERS = (
    "you are a strict evaluator",
    "system prompt",
    "every claim must reference",
    "break this into research subtasks",
    "as an ai language model",
)

# Conservative PII patterns. Emails and long digit runs (phone / card numbers)
# are the realistic leaks for a web-research report; kept simple on purpose.
# 对于 Web 研究报告，电子邮件和长数字串（电话/卡号）是现实中最可能泄露的
# 个人信息；这里有意保持简单。
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LONG_DIGITS_RE = re.compile(r"\b(?:\d[ -]?){9,}\b")


def scan_content(text: str) -> list[str]:
    """Return a list of content violations found in the report text (empty = clean).
    返回报告文本中发现的内容违规列表（为空表示干净）。"""
    flags: list[str] = []
    low = text.lower()
    for marker in _PROMPT_LEAK_MARKERS:
        if marker in low:
            flags.append(f"possible leaked system prompt: '{marker}'")
    if _EMAIL_RE.search(text):
        flags.append("possible PII: email address in report")
    if _LONG_DIGITS_RE.search(text):
        flags.append("possible PII: phone/card-like number in report")
    return flags
