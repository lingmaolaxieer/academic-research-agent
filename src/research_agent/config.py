from dotenv import load_dotenv
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_planner: str = "openai:gpt-4o"
    model_researcher: str = "openai:gpt-4o-mini"
    model_synthesizer: str = "openai:gpt-4o"
    model_critic: str = "openai:gpt-4o-mini"
    model_judge: str = "openai:gpt-4o-mini"   # cheap model for LLM-as-judge evals 用于 LLM-as-judge 评估的低成本模型

    temperature: float = 0.0
    max_research_iterations: int = 2

    # Eval gates: the agent must stay at/above these mean scores in CI.
    # 评估门槛：agent 在 CI 中必须保持不低于这些平均分。
    eval_min_faithfulness: float = 0.7
    eval_min_citation_accuracy: float = 0.8
    eval_min_coverage: float = 0.5
    eval_max_cases: int = 0   # 0 = run the whole dataset; >0 caps it (cheap CI runs) 0 = 运行整个数据集；>0 = 限制数量（低成本 CI 运行）

    # ── Guardrails & robustness ─────────────────────────────────────
    # ── 守卫与稳健性 ─────────────────────────────────────────────────────────
    # Input guardrail. The classifier decides if a query belongs to this domain
    # and is safe to run. `guardrail_domain` is injected into its prompt so the
    # scope is configurable without code changes.
    # 输入守卫。分类器决定查询是否属于此领域且可以安全运行。
    # `guardrail_domain` 会注入其提示词，因此无需修改代码即可配置范围。
    guardrail_input_enabled: bool = True
    guardrail_domain: str = (
        "academic and technical research, uploaded document question answering, "
        "paper comparison and public-source research"
    )

    # Output guardrail. A report must cite real gathered sources for at least
    # this fraction of its citations, or it is sent back to the synthesizer to
    # be repaired — at most `max_output_repairs` times before the task fails.
    # 输出守卫。报告必须引用至少达到此比例的真实已收集来源，否则会发回
    # synthesizer 修复；最多修复 `max_output_repairs` 次后将任务标记为失败。
    guardrail_output_enabled: bool = True
    min_citation_coverage: float = 0.8
    max_output_repairs: int = 1

    # Stop threshold checked after critic, not a strict call-before-execution budget.
    # API callbacks independently enforce max_model_calls before each model invocation.
    # 研究轮后的停止阈值；API 回调另行执行调用前预算。
    max_llm_calls_per_run: int = 22

    # Robustness. Retry policy applied to every LLM node so transient failures
    # (timeouts, 429 rate limits) back off and retry instead of crashing the run.
    # 稳健性。重试策略应用于每个 LLM 节点，使暂时性故障（超时、429 速率限制）
    # 进行退避并重试，而不是导致运行崩溃。
    retry_max_attempts: int = 4
    retry_initial_interval: float = 1.0   # seconds; doubles each attempt (1, 2, 4, 8) 秒数；每次尝试翻倍（1、2、4、8）
    retry_backoff_factor: float = 2.0

    # ── Persistence & human-in-the-loop ───────────────────
    # ── 持久化与人在环 ───────────────────────────────────────────────────────
    # Where the SQLite checkpointer writes durable state. A single local file;
    # zero setup, zero cost. For production swap in PostgresSaver (see
    # research_agent/persistence.py). Relative paths resolve from the cwd, so
    # like .env it expects you to run from the project root.
    # SQLite checkpointer 写入持久化状态的位置。单个本地文件；无需配置且零成本。
    # 生产环境请替换为 PostgresSaver（参见 research_agent/persistence.py）。
    # 相对路径根据当前工作目录解析，因此和 .env 一样，要求从项目根目录运行。
    checkpoint_db_path: str = "checkpoints.db"

    # Runtime profiles. `local` preserves the zero-setup SQLite/thread-pool
    # development path; `production` uses the shared services below.
    runtime_profile: Literal["local", "production"] = "local"
    database_url: str = ""
    redis_url: str = "redis://redis:6379/0"
    task_backend: Literal["thread", "celery"] = "thread"
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    celery_queue: str = "research"
    worker_lease_seconds: int = Field(default=600, ge=30, le=7200)
    max_recovery_attempts: int = Field(default=3, ge=1, le=20)
    event_retention: int = Field(default=1000, ge=100, le=10000)
    log_format: Literal["text", "json"] = "text"
    worker_metrics_port: int = Field(default=9101, ge=1024, le=65535)
    outbox_metrics_port: int = Field(default=9102, ge=1024, le=65535)

    # Human-in-the-loop approval gate. When True, the graph pauses right after
    # the FIRST plan (before the costly fan-out) and waits for a human to
    # approve / edit / reject the subtasks. Default OFF so the agent stays fully
    # autonomous for evals, CI, and the API's default path — turn it on per-run
    # for interactive/operator use. NOTE: HITL only works with a checkpointer;
    # interrupt() needs durable state to pause and resume.
    # 人在环审批门。为 True 时，图会在 FIRST plan 之后（高成本 fan-out 之前）
    # 暂停并等待人类批准/编辑/拒绝子任务。默认关闭，使 evals、CI 和 API 默认路径
    # 保持完全自主——交互式/操作员使用时可按运行开启。注意：HITL 仅在有
    # checkpointer 时生效；interrupt() 需要持久化状态才能暂停和恢复。
    hitl_approval_enabled: bool = False


    # Single-process service limits; this is not a durable distributed queue.
    job_db_path: str = "data/jobs.db"
    # Bound graph fan-out separately from Celery's ability to run independent jobs.
    max_workers: int = Field(default=2, ge=1, le=32)
    max_pending_jobs: int = Field(default=16, ge=1, le=1000)
    max_research_subtasks: int = Field(default=3, ge=1, le=5)
    max_researcher_model_calls: int = Field(default=3, ge=2, le=10)
    max_researcher_tool_calls: int = Field(default=3, ge=1, le=10)
    graph_recursion_limit: int = Field(default=40, ge=5, le=200)
    model_timeout_seconds: float = Field(default=60.0, gt=0)
    structured_output_attempts: int = Field(default=2, ge=1, le=3)
    knowledge_db_path: str = "data/knowledge.db"
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=199)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    embedding_model: str = ""
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    # pgvector HNSW supports vector dimensions up to 2,000.
    embedding_dimensions: int = Field(default=1536, ge=2, le=2000)
    reranker_model: str = ""
    reranker_candidates: int = Field(default=20, ge=2, le=100)
    max_upload_bytes: int = Field(default=5_000_000, ge=1000, le=20_000_000)
    max_document_chars: int = Field(default=200_000, ge=1000)
    max_context_chars: int = Field(default=16000, ge=2000, le=60000)
    max_evidence: int = Field(default=50, ge=5, le=100)
    knowledge_transport: Literal["local", "mcp"] = "local"
    api_keys: dict[str, str] = Field(default_factory=dict)  # principal -> secret, JSON env var
    allow_anonymous: bool = True  # Local development only. Disable for shared deployments.
    requests_per_minute: int = Field(default=60, ge=1)
    task_timeout_seconds: int = Field(default=300, ge=10)
    max_model_calls: int = Field(default=30, ge=4, le=200)


settings = Settings()
