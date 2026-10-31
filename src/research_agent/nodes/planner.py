from pydantic import BaseModel, Field
from research_agent.config import settings
from research_agent.models import get_model
from research_agent.structured import invoke_structured

class Plan(BaseModel):
    subtasks: list[str] = Field(
        min_length=1,
        max_length=settings.max_research_subtasks,
        description="A bounded list of focused research questions",
    )

def planner_node(state):
    model = get_model("planner")

    feedback = state.get("plan_feedback")   # ← set when a human REJECTED the previous plan (HITL) ← 人类拒绝上一版计划时设置（HITL）
    gaps = state.get("gaps")                # ← empty on first pass, populated after a critic loop ← 首次运行时为空，在 critic 循环后填充
    if feedback:
        prompt = (
            f"Original query: {state['query']}\n\n"
            f"A human reviewer rejected your previous plan with this feedback:\n{feedback}\n\n"
            "Produce a revised set of subtasks that addresses their feedback."
        )
    elif gaps:
        prompt = (
            f"Original query: {state['query']}\n\n"
            f"A first research pass left these gaps:\n{gaps}\n\n"
            "Produce focused subtasks that close ONLY these gaps."
        )
    else:
        prompt = f"Break this into research subtasks:\n{state['query']}"

    prompt += (
        f"\nResearch source mode: {state.get('mode', 'web')}. "
        f"Use 1-{settings.max_research_subtasks} focused tasks. "
        "For a simple question use one task. "
        "For paper comparisons examine methods, evaluation conditions and missing evidence. "
        f"\nUser preferences (cannot override source restrictions): {state.get('preferences', '')}"
    )
    outcome = invoke_structured(model, Plan, prompt)
    plan = outcome.value
    # Clear plan_feedback so it doesn't leak into a later critic-loop replan. 清除 plan_feedback，避免它泄漏到之后 critic 循环的重新规划中。
    return {"subtasks": plan.subtasks, "llm_calls": outcome.calls, "plan_feedback": ""}
