from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig
from research_agent.config import settings
from research_agent.guardrails import RETRY_POLICY
from research_agent.state import ResearchState
from research_agent.nodes.planner import planner_node
from research_agent.nodes.approval import approval_node
from research_agent.nodes.source_researcher import researcher_node
from research_agent.nodes.critic import critic_node
from research_agent.nodes.synthesizer import synthesizer_node
from research_agent.nodes.guardrails import input_guard_node, output_guard_node
from research_agent.nodes.router import quick_plan_node, router_node
from typing import cast

def route_after_input_guard(state) -> str:
    """Input guardrail: refuse out-of-scope/unsafe queries before any spend. 输入守卫：在产生任何消耗前拒绝超出范围或不安全的查询。"""
    return END if state.get("rejected") else "router"

def route_after_router(state) -> str:
    return "quick_plan" if state.get("route") == "quick" else "planner"

def route_to_researchers(state) -> list[Send]:
    """Emit one Send per subtask → N parallel researcher branches. 为每个子任务发出一个 Send → N 个并行的 researcher 分支。"""
    return [
        Send("researcher", {
            "subtasks": [task], "findings": [], "query": state.get("query", ""),
            "owner": state.get("owner", "local"), "mode": state.get("mode", "web"),
            "document_ids": state.get("document_ids", []),
            "route": state.get("route", "deep"),
        })
        for task in state["subtasks"]
    ]

def route_after_approval(state):
    """After the human-in-the-loop gate: abort, redraft, or fan out.

    A conditional edge may return a node name (string), the END sentinel, or a
    list of Sends. We use all three: END to stop a run the human aborted,
    'planner' to loop back on a reject, or the fan-out Sends on approve/edit.
    Every possible target is declared in the edge's path map.

    人在环审批门之后：中止、重拟计划或扩散。
    条件边可以返回节点名（字符串）、END 终止标记，或者一个 Send 列表。我们使用这三种方式：END 用于停止人工中止的运行；'planner' 用于在拒绝后回到规划；approve/edit 时使用扩散的 Send。所有可能的目标都在边的路径映射中声明。
    """
    if state.get("_aborted"):
        return END
    if state.get("_plan_rejected"):
        return "planner"
    return route_to_researchers(state)

def route_after_critic(state) -> str:
    """Stop the research loop when the findings are sufficient, the iteration cap
    is reached, OR the budget guard trips — whichever comes first. Any of these
    falls through to the synthesizer so we still ship a report from what we have.

    当发现足够、迭代上限达到，或预算守卫触发时停止研究循环——以先发生者为准。任何一种情况都会落到 synthesizer，以便我们仍能基于现有内容交付报告。
    """
    budget_spent = state.get("llm_calls", 0) >= settings.max_llm_calls_per_run
    if state.get("_sufficient") or state["iterations"] >= settings.max_research_iterations or budget_spent:
        return "synthesizer"
    return "planner"

def route_after_researcher(state) -> str:
    return "synthesizer" if state.get("route") == "quick" else "critic"

def route_after_output_guard(state) -> str:
    """Output guardrail: send a failing report back to be repaired, else deliver. 输出守卫：将失败的报告送回修复，否则交付。"""
    return "synthesizer" if state.get("_needs_repair") else END

def build_graph(checkpointer=None):
    """Compile the research graph.

    ``checkpointer`` pass a saver (e.g. from ``research_agent.persistence.sqlite_saver``)
    and the graph persists its state after every superstep, making runs durable,
    resumable, and able to pause fora human.
    Left ``None`` (the default), the graph behaves exactly as before —
    in-memory and ephemeral — which keeps evals/CI unchanged. Note: the
    human-in-the-loop approval gate only actually pauses when a checkpointer is
    present, because ``interrupt()`` needs durable state to resume from.

    编译 research graph。
    ``checkpointer`` 传入一个 saver（例如来自 ``research_agent.persistence.sqlite_saver``），图会在每个 superstep 之后持久化状态，使运行具有持久性、可恢复性，并能够在人工介入时暂停。
    若留 ``None``（默认值），图行为与之前完全相同——仅在内存中临时运行，从而保持 evals/CI 不变。注意：只有在存在 checkpointer 时，人机协作审批门才会真正暂停，因为 ``interrupt()`` 需要持久化状态来恢复。
    """
    g = StateGraph(ResearchState)
    g.add_node("input_guard", input_guard_node, retry_policy=RETRY_POLICY)
    g.add_node("router", router_node)
    g.add_node("quick_plan", quick_plan_node)
    g.add_node("planner", planner_node, retry_policy=RETRY_POLICY)
    g.add_node("approval", approval_node)   # human-in-the-loop gate; no LLM call 人在环审批门；无 LLM 调用
    g.add_node("researcher", researcher_node, retry_policy=RETRY_POLICY)   # backoff handles 429s on fan-out 退避策略处理 fan-out 场景中的 429
    g.add_node("critic", critic_node, retry_policy=RETRY_POLICY)
    g.add_node("synthesizer", synthesizer_node, retry_policy=RETRY_POLICY)
    g.add_node("output_guard", output_guard_node)   # no LLM call → no retry needed 无 LLM 调用 → 无需重试

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input_guard, ["router", END])
    g.add_conditional_edges("router", route_after_router, ["quick_plan", "planner"])
    g.add_conditional_edges("quick_plan", route_to_researchers, ["researcher"])
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_conditional_edges("researcher", route_after_researcher, ["critic", "synthesizer"])
    g.add_conditional_edges("critic", route_after_critic, ["planner", "synthesizer"])
    g.add_edge("synthesizer", "output_guard")
    g.add_conditional_edges("output_guard", route_after_output_guard, ["synthesizer", END])
    return g.compile(checkpointer=checkpointer)


def _print_result(result):
    print(result["report"])
    if result.get("guardrail_flags"):
        print("\n⚠ guardrail flags:", result["guardrail_flags"])
    print(f"\nLLM calls this run: {result.get('llm_calls')}")


def _ask_human(payload):
    """Render the paused plan and BLOCK on stdin for a real human decision.

    Returns the dict expected by the approval node's interrupt():
    {"action": "approve"} | {"action": "edit", "subtasks": [...]}
                          | {"action": "reject", "feedback": "..."}
                          | {"action": "abort", "reason": "..."}

    渲染暂停的计划并在 stdin 上阻塞等待真实的人类决策。
    返回 approval node 的 interrupt() 所期望的字典：
    {"action": "approve"} | {"action": "edit", "subtasks": [...]}
                          | {"action": "reject", "feedback": "..."}
                          | {"action": "abort", "reason": "..."}
    """
    print("\n⏸  Graph paused for human approval.")
    print(f"Query: {payload.get('query')}")
    print("Proposed subtasks:")
    for i, task in enumerate(payload["proposed_subtasks"], 1):
        print(f"  {i}. {task}")

    # input() blocks until you type something and press Enter — THIS is the wait.
    # input() 会在你输入内容并按下 Enter 之前一直阻塞——这就是等待过程。
    choice = input("\nDecision [approve / edit / reject / abort] (default approve): ").strip().lower()

    if choice == "edit":
        raw = input("New subtasks (comma-separated): ")
        subtasks = [s.strip() for s in raw.split(",") if s.strip()]
        return {"action": "edit", "subtasks": subtasks}
    if choice == "reject":
        feedback = input("Feedback for the planner: ").strip()
        return {"action": "reject", "feedback": feedback}
    if choice == "abort":
        reason = input("Reason for aborting (optional): ").strip()
        return {"action": "abort", "reason": reason}
    return {"action": "approve"}


if __name__ == "__main__":
    # HITL (it's off by default) and wrap the graph in a SQLite checkpointer
    # so the run survives the pause. A `thread_id` names this one run;
    # re-using it is what makes a run resumable.
    # HITL（默认关闭）并将图包装在 SQLite checkpointer 中，以便运行在暂停后仍可持续。`thread_id` 为这次运行命名；重复使用它使运行可恢复。
    import uuid

    from langgraph.types import Command

    from research_agent.persistence import sqlite_saver

    settings.hitl_approval_enabled = True   # demo-only; normally set via env/config 演示用；通常通过 env/config 设置

    query = "What are the trade-offs of RAG vs fine-tuning?"
    config: RunnableConfig = {
        "configurable": {"thread_id": f"demo-{uuid.uuid4().hex[:8]}"},
        "run_name": "academic-research",
        "metadata": {
            "model_planner": settings.model_planner,
            "model_researcher": settings.model_researcher,
            "model_synthesizer": settings.model_synthesizer,
            "model_critic": settings.model_critic,
        },
    }

    with sqlite_saver() as saver:
        app = build_graph(checkpointer=saver)

        # Start the run. It executes input_guard → planner → approval, then the
        # approval node calls interrupt(), which PAUSES and returns control here.
        # 启动运行。它会执行 input_guard → planner → approval，然后 approval 节点调用 interrupt()，从而暂停并将控制权返回到这里。
        result = app.invoke(cast("ResearchState", {"query": query}), config=config)

        # The pause shows up as an __interrupt__ payload in the result. Loop while
        # the graph is paused: ask the human, resume with their decision, repeat.
        # (A "reject" replans and pauses AGAIN, which is why this is a loop.)
        # 暂停会以 __interrupt__ 载荷的形式出现在结果中。只要图处于暂停状态，就循环执行：询问人类并用其决策恢复，再重复。
        # （"reject" 会重新规划并再次暂停，因此这里必须用循环。）
        while result.get("__interrupt__"):
            decision = _ask_human(result["__interrupt__"][0].value)
            print(f"\n▶  Resuming with: {decision['action']}\n")
            result = app.invoke(Command(resume=decision), config=config)

        _print_result(result)
