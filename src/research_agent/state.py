from typing import Annotated, TypedDict
from operator import add

from pydantic import BaseModel
from research_agent.evidence import merge_evidence


class ReportData(TypedDict):
    summary: str
    sections: list[str]
    citations: list[str]
    limitations: list[str]

class Finding(BaseModel):
    claim: str
    source_url: str
    source_title: str

class ResearchState(TypedDict, total=False):
    owner: str
    mode: str
    document_ids: list[str]
    preferences: str
    research_depth: str
    route: str
    evidence: Annotated[list[dict], merge_evidence]
    available_citations: list[str]
    source_errors: Annotated[list[str], add]
    require_approval: bool
    query: str                                # set at input 设置在输入时
    subtasks: list[str]                       # written by planner 由 planner 写入
    findings: Annotated[list[str], add]   # Researcher summaries; evidence is the source of truth.
    report: ReportData                   # Validated report serialized into checkpoint state.
    iterations: int                           # bumped each critic pass 每次 critic 轮次递增
    gaps: list[str]                           # critic's notes for the next round critic 对下一轮的备注

    # Guardrails
    rejected: bool                            # input guard verdict 输入守卫判断结果
    rejection_reason: str                     # message delivered when a query is refused 当查询被拒绝时传递的消息
    llm_calls: Annotated[int, add]            # budget accounting (summed across fan-out) 预算统计（在并发分支中累计求和）
    repairs: int                              # output-guard repair attempts so far 到目前为止的输出守卫修复尝试次数
    repair_notes: str                         # output guard → synthesizer feedback on a repair 输出守卫 → synthesizer 对修复的反馈
    guardrail_flags: list[str]                # citation/content issues surfaced on the report 报告中暴露出的引用/内容问题

    # Human-in-the-loop
    plan_feedback: str                        # human's notes when a plan is rejected → planner redrafts 当计划被拒绝时，人类的备注 → planner 重新制定计划
    _plan_rejected: bool                      # transient routing flag set by the approval gate 由 approval gate 设置的临时路由标志
    _aborted: bool                            # human aborted at the gate → route straight to END 人类在 gate 处中止 → 直接路由到 END
    _sufficient: bool                        # critic -> conditional edge; must survive state updates
    _needs_repair: bool                      # output guard -> conditional edge
