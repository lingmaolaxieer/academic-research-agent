"""Human-in-the-loop approval gate. 人在环审批门。

This node sits between the planner and the parallel fan-out — the natural place
for a human to sign off *before* the agent spends money on N concurrent
researcher branches. It uses LangGraph's ``interrupt()`` primitive:
此节点位于 planner 和并行 fan-out 之间，是人类在 agent 为 N 个并发 researcher
分支花费资源之前进行确认的自然位置。它使用 LangGraph 的 ``interrupt()`` 原语：

* ``interrupt(payload)`` **pauses** the whole graph and surfaces ``payload`` to
  the caller (the operator/UI). The current state is saved to the checkpointer,
  so the process can even die here and resume later.
* ``interrupt(payload)`` **暂停**整个图，并将 ``payload`` 提供给调用方（操作员/UI）。
  当前状态会保存到 checkpointer，因此进程即使在此处终止，也可以稍后恢复。
* The caller inspects the payload, gets a human decision, and resumes with
  ``Command(resume=decision)``. On resume LangGraph **re-runs this node from the
  top**, and this time ``interrupt()`` returns ``decision`` instead of pausing.
* 调用方检查 payload，获取人类决策，然后使用 ``Command(resume=decision)`` 恢复。
  恢复时，LangGraph 会**从头重新运行此节点**，这一次 ``interrupt()`` 返回
  ``decision``，而不是暂停。

The human can choose one of four actions:
人类可以选择以下四种操作之一：

* ``approve`` — run the plan as-is → fall through to the fan-out;
* ``approve`` —— 按原计划运行 → 进入 fan-out；
* ``edit``    — replace the subtasks with the human's edited list → fan-out;
* ``edit`` —— 用人类编辑后的列表替换子任务 → 进入 fan-out；
* ``reject``  — send it back to the planner to redraft, optionally with feedback;
* ``reject`` —— 将其发回 planner 重新起草，可选择附带反馈；
* ``abort``   — stop the run for good → route straight to END with an aborted report.
* ``abort`` —— 永久停止运行 → 直接路由到 END，并生成中止报告。

Because ``interrupt()`` requires durable state, this gate only does anything when
a checkpointer is configured AND ``settings.hitl_approval_enabled`` is True. When
disabled it is a transparent pass-through, so evals/CI/the default API path stay
fully autonomous.
由于 ``interrupt()`` 需要持久化状态，只有在配置了 checkpointer 且
``settings.hitl_approval_enabled`` 为 True 时，此审批门才会生效。禁用时它会
透明地直接通过，因此 evals/CI/默认 API 路径仍保持完全自主运行。
"""

from __future__ import annotations

from langgraph.types import interrupt

from research_agent.config import settings


def approval_node(state):
    """Pause for human approval of the FIRST plan; pass through otherwise.
    暂停并等待人类审批 FIRST plan；否则直接通过。"""
    # Two reasons to skip the gate entirely (return a clean, non-rejected state):
    #   1. HITL is turned off → stay autonomous.
    #   2. We're on a critic-loop replan (iterations > 0) → the human already
    #      approved this run's direction; don't nag them every loop. Only the
    #      first, costly fan-out needs sign-off.
    # 完全跳过审批门有两个原因（返回干净且未拒绝的状态）：
    #   1. HITL 已关闭 → 保持自主运行。
    #   2. 当前处于 critic 循环重新规划阶段（iterations > 0）→ 人类已经批准
    #      本次运行的方向；不要在每个循环中反复打扰他们。只有第一次高成本的
    #      fan-out 需要人工确认。
    if not state.get("require_approval", settings.hitl_approval_enabled) or state.get("iterations", 0) > 0:
        return {"_plan_rejected": False}

    # Snapshot EXACTLY what the human is approving into the interrupt payload, so
    # the operator sees precisely what they're signing off on (a best practice:
    # never make them approve something they can't see).
    # 将人类正在审批的内容 EXACTLY 快照到 interrupt payload 中，使操作员能够
    # 准确看到其正在确认的内容（最佳实践：绝不要让他们审批看不到的内容）。
    decision = interrupt(
        {
            "type": "plan_approval",
            "message": (
                "Review the proposed research plan, then resume with a decision: "
                "{'action': 'approve'} | "
                "{'action': 'edit', 'subtasks': [...]} | "
                "{'action': 'reject', 'feedback': '...'} | "
                "{'action': 'abort', 'reason': '...'}"
            ),
            "query": state.get("query"),
            "proposed_subtasks": state.get("subtasks", []),
        }
    )

    # ── Below here only runs AFTER the human resumes ────────────────────────────
    # ── 以下内容仅在人类恢复运行后执行 ─────────────────────────────────────────
    # `decision` is whatever was passed to Command(resume=...). Default to
    # "approve" so a bare/empty resume is treated as a green light.
    # `decision` 是传递给 Command(resume=...) 的内容。默认设为 "approve"，
    # 因此空的或不带内容的恢复操作会被视为放行。
    action = (decision or {}).get("action", "approve")

    if action == "edit":
        # Human handed us a corrected list of subtasks → use it, then fan out.
        # 人类提供了修正后的子任务列表 → 使用该列表，然后进行 fan-out。
        return {"subtasks": decision["subtasks"], "_plan_rejected": False, "plan_feedback": ""}

    if action == "reject":
        # Route back to the planner to redraft. Their free-text feedback (if any)
        # is stored so the planner can address it on the next pass.
        # 路由回 planner 重新起草。保存人类提供的自由文本反馈（如果有），
        # 以便 planner 在下一轮中处理。
        return {"_plan_rejected": True, "plan_feedback": decision.get("feedback", "")}

    if action == "abort":
        # Stop the run for good. We route straight to END (see route_after_approval)
        # and write a report-shaped result so callers get a uniform output, mirroring
        # how the input guard refuses a query. The graph reaches a terminal state, so
        # the final checkpoint records the abort rather than leaving a dangling pause.
        # 永久停止运行。我们直接路由到 END（参见 route_after_approval），
        # 并写入报告形状的结果，使调用方获得统一输出，类似于输入守卫拒绝查询的方式。
        # 图会到达终止状态，因此最终 checkpoint 会记录中止，而不是留下悬挂的暂停。
        reason = decision.get("reason") or "Run aborted by the operator at the approval gate."
        return {
            "_aborted": True,
            "report": {"summary": f"Run aborted: {reason}", "sections": [], "citations": []},
        }

    # approve (or unknown action) → proceed as-is. We still set the flag
    # explicitly to False so a stale `True` from a prior reject can't linger in
    # the persisted state and re-trigger the reject route.
    # approve（或未知操作）→ 按原样继续。我们仍将该标志显式设为 False，
    # 以防之前 reject 遗留的 `True` 保存在持久化状态中，并再次触发拒绝路由。
    return {"_plan_rejected": False, "plan_feedback": ""}
