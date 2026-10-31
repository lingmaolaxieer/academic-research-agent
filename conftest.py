"""Shared pytest setup for the eval harness.
评估测试框架的共享 pytest 设置。

Lives at the repo root so the root is on ``sys.path`` (makes ``evals`` importable
alongside the installed ``research_agent`` package), and so the heavy "run the
agent over the whole dataset" step happens exactly once per session.
该文件位于仓库根目录，因此根目录会加入 ``sys.path``（使 ``evals`` 能与已安装
的 ``research_agent`` 包一起导入），并确保耗时的“在整个数据集上运行 agent”
步骤在每个会话中只执行一次。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import pytest
from dotenv import load_dotenv

from research_agent.state import ResearchState

load_dotenv()

_DATASET = Path(__file__).parent / "evals" / "dataset.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "eval: end-to-end agent quality evals (slow, costs LLM calls)；端到端 agent 质量评估（较慢，会产生 LLM 调用成本）")


def _load_cases():
    cases = json.loads(_DATASET.read_text(encoding="utf-8"))
    from research_agent.config import settings

    if settings.eval_max_cases > 0:
        cases = cases[: settings.eval_max_cases]
    return cases


@pytest.fixture(scope="session")
def eval_results():
    """Run the agent on every dataset case once and score each output.
    在每个数据集案例上运行一次 agent，并为每个输出评分。

    Returns a list of dicts: id, faithfulness, citation_accuracy, coverage,
    plus operational metrics (latency_s, iterations). Skips the whole eval if no
    API key is configured, so the suite degrades gracefully off-CI.
    返回字典列表：id、faithfulness、citation_accuracy、coverage，以及运行指标
    （latency_s、iterations）。如果未配置 API key，则跳过整个评估，使套件在
    CI 之外也能平稳运行。
    """
    if os.getenv("RUN_LIVE_EVALS", "").lower() != "true":
        pytest.skip("Set RUN_LIVE_EVALS=true to opt into paid model/search evaluation")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping live agent evals")

    from research_agent.graph import build_graph
    from evals.judges import faithfulness, citation_accuracy, coverage

    app = build_graph()
    results = []
    for case in _load_cases():
        start = time.perf_counter()

        state = app.invoke(cast(ResearchState, {"query": case["query"]}))
        latency_s = time.perf_counter() - start

        findings = state.get("evidence", [])
        report = state.get("report", {})

        results.append(
            {
                "id": case["id"],
                "faithfulness": faithfulness(findings, report).score,
                "citation_accuracy": citation_accuracy(findings, report),
                "coverage": coverage(case["query"], case["key_points"], report).score,
                "latency_s": round(latency_s, 1),
                "iterations": state.get("iterations", 0),
            }
        )
    return results
