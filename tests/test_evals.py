"""
CI-gated agent quality evals.
受 CI 门控的 agent 质量评估。

Run the whole agent over the eval dataset, score each output with the judges in
``evals/judges.py``, and FAIL if mean faithfulness / citation accuracy / coverage
drops below the thresholds in ``config.py``. This is the regression gate that runs
on every pull request (see ``.github/workflows/evals.yml``).
在评估数据集上运行整个 agent，对每个输出使用 ``evals/judges.py`` 中的评判器进行打分，
并在平均忠实度 / 引用准确率 / 覆盖率低于 ``config.py`` 中的阈值时失败。这是每个拉取请求
都会运行的回归门禁（参见 ``.github/workflows/evals.yml``）。

Run locally:   uv run pytest -m eval -s
本地运行：     uv run pytest -m eval -s
Cheap subset:  EVAL_MAX_CASES=3 uv run pytest -m eval -s
小规模子集：  EVAL_MAX_CASES=3 uv run pytest -m eval -s
"""

from __future__ import annotations

from statistics import mean

import pytest

from research_agent.config import settings

pytestmark = pytest.mark.eval


def _mean(results, key: str) -> float:
    return mean(r[key] for r in results)


def test_print_report(eval_results):
    """Not a gate — prints a per-case scorecard so CI logs show what happened.
    这不是门控测试——打印每个案例的评分卡，使 CI 日志显示运行结果。"""
    header = f"{'case':24} {'faith':>6} {'cite':>6} {'cover':>6} {'lat(s)':>7} {'iters':>5}"
    print("\n" + header)
    print("-" * len(header))
    for r in eval_results:
        print(
            f"{r['id']:24} {r['faithfulness']:6.2f} {r['citation_accuracy']:6.2f} "
            f"{r['coverage']:6.2f} {r['latency_s']:7.1f} {r['iterations']:5d}"
        )
    print("-" * len(header))
    print(
        f"{'MEAN':24} {_mean(eval_results, 'faithfulness'):6.2f} "
        f"{_mean(eval_results, 'citation_accuracy'):6.2f} "
        f"{_mean(eval_results, 'coverage'):6.2f}"
    )


def test_faithfulness_gate(eval_results):
    score = _mean(eval_results, "faithfulness")
    assert score >= settings.eval_min_faithfulness, (
        f"mean faithfulness {score:.2f} < threshold {settings.eval_min_faithfulness}"
    )


def test_citation_accuracy_gate(eval_results):
    score = _mean(eval_results, "citation_accuracy")
    assert score >= settings.eval_min_citation_accuracy, (
        f"mean citation accuracy {score:.2f} < threshold {settings.eval_min_citation_accuracy}"
    )


def test_coverage_gate(eval_results):
    score = _mean(eval_results, "coverage")
    assert score >= settings.eval_min_coverage, (
        f"mean coverage {score:.2f} < threshold {settings.eval_min_coverage}"
    )
