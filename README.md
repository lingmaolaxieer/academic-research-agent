# 学术资料调研助手 · Academic Research Agent

基于 [libattistella/deep-research-agent](https://github.com/libattistella/deep-research-agent) 的 Agent 应用工程项目，保留原 MIT 许可证。面向论文和技术文档问答、多文档比较与公开网页补充调研。

示例任务：上传可见光与红外融合资料，比较方法、输入条件、评价协议与局限，并指出哪些结论缺少依据。项目不训练视觉模型，也不声称自动发现科研创新。

## 能力与执行流程

```text
文档 → 解析/分页 → 重叠分块 → 所有者隔离的索引
                                  ↓
提交问题 → 认证/Redis 限流 → PostgreSQL 任务 + Outbox → Celery Worker
                                                        ↓
输入检查 → 成本路由 ──快速问答──→ 单次检索 ───────────────────┐
                  └─深度调研──→ 规划 → 可选人工审批         │
                                      ↓                 │
                              并行 Researcher 工具循环    │
                              文档检索 / 网页搜索 / MCP    │
                                      ↓                 │
                              证据评审 ← 补查 ← 缺口       │
                                                        ↓
                                   报告生成 → 引用校验 → 结果
                                                 ↓ 不通过
                                             有限修复 / 失败
```

- **RAG**：文本 PDF、Markdown、TXT；生产模式使用 PostgreSQL/pgvector HNSW，BM25 与向量双路召回、RRF 融合及可选 Cross-Encoder 重排。记录文档、分块、页码与原文；本地模式保留 SQLite 精确向量检索。
- **Routing / Multi-Agent**：先区分快速问答与深度调研；简单问题跳过规划与评审，复杂任务由规划器生成 1～5 个子任务并通过 LangGraph `Send` 并行执行。来源支持 local、web、hybrid。
- **证据约束**：程序保存工具返回的原文片段并生成稳定 evidence_id；写作只引用上下文中实际出现的 ID。无证据返回资料不足，引用修复耗尽则失败，不会靠附加证据索引伪装已逐条引用。引用合法性不等于语义正确性。
- **MCP**：本地 stdio Knowledge Server，客户端真实完成初始化、工具发现与调用；身份由可信宿主绑定，模型不能传入其他用户身份或任意文件路径。
- **HITL / Memory**：研究计划支持批准、编辑、拒绝、终止；LangGraph PostgreSQL Checkpoint 支持跨进程暂停恢复。长期偏好须由用户显式保存，可查询和删除。
- **异步任务**：FastAPI 负责受理、查询、审批与协作式取消；任务与 Outbox 在 PostgreSQL 同事务写入，独立投递器发送 Celery 消息。Worker 使用执行租约和状态写入令牌校验；过期任务有限次重新入队。Redis 承担 Broker 与跨实例限流。
- **API / 安全**：API Key 用户绑定；任务、文档、检索和偏好按所有者隔离；研究创建可携带 `Idempotency-Key` 避免同一用户重试时重复建任务。检索前权限过滤、上传边界、调用预算和对外错误脱敏。进度通过可续传的持久事件 SSE 推送，轮询仍可作为降级路径。
- **观测与预算**：结构化日志、Prometheus HTTP/模型/工具/任务指标、节点事件和 Token 记录；共享模型调用预算、外部调用超时、结构输出校验及有限重试；可选 LangSmith 链路追踪。
- **演示界面**：文档上传、选择来源、提交研究、审批计划、查看报告及证据。模型输出以文本呈现，不执行其 HTML。
- **评估**：固定检索样例集、离线真实工作流集成测试；付费模型评测单独显式开启。

## 本地启动

Python 3.12 与 uv：

```powershell
uv sync --locked --python 3.12 --group dev
Copy-Item .env.example .env
# 编辑 .env，填写自己的模型密钥；网页调研还需要 Tavily 密钥。
uv run uvicorn research_agent.api:api --host 127.0.0.1 --port 8000 --workers 1
```

浏览器进入 http://127.0.0.1:8000 。接口文档为 /docs。

本地配置默认使用 DeepSeek 兼容聊天接口与 BM25，无需 GPU 或 Embedding 服务。仅有聊天 API 并不意味着有向量化接口；启用 dense/hybrid 前需另行填写 EMBEDDING_MODEL / EMBEDDING_BASE_URL / EMBEDDING_API_KEY，再重新导入文档。变更向量模型或维度后必须重建索引。

注意两种 hybrid 不同：请求的 mode=hybrid 是“文档 + 网页”；RETRIEVAL_MODE=hybrid 是“关键词 + 向量”。

## MCP 与身份

设置 KNOWLEDGE_TRANSPORT=mcp，文档检索将经由 MCP 子进程执行。local 模式直接调用同一知识服务，便于对比边界与开销。stdio 子进程无需开放网络端口。

默认 ALLOW_ANONYMOUS=true 仅用于本机学习。多人使用时设置 ALLOW_ANONYMOUS=false，并通过 API_KEYS 配置“用户标识 → 独立随机密钥”的 JSON 字典。客户端发送 X-API-Key 或 Bearer token；用户身份不接受模型或请求正文覆盖。有任意 API_KEYS 配置时均要求有效密钥。

仅使用文档模式不会调用网页搜索，但文档片段仍需发送给配置的模型服务。Hybrid 网页搜索词可能由文档内容生成。开启外部追踪前须明确其数据传输范围。

## 生产模式：Docker Compose

生产拓扑包含 API、Celery Worker、事务 Outbox 投递器、PostgreSQL/pgvector、Redis 和 Prometheus。复制 `.env.example` 为 `.env` 后，至少设置：

- 独立随机的 `POSTGRES_PASSWORD`；
- `API_KEYS`，并保持 `ALLOW_ANONYMOUS=false`；
- 聊天模型配置；
- 支持 OpenAI Embedding 协议的模型、Base URL、密钥与正确维度。

```powershell
docker compose up --build
```

API 绑定 `127.0.0.1:8000`，Prometheus 绑定 `127.0.0.1:9090`；PostgreSQL 与 Redis 不暴露宿主机端口。Dockerfile 中应用以非 root 用户运行。Cross-Encoder 依赖安装在生产镜像中，首次使用配置的模型时需要下载模型权重；Compose 使用共享持久卷保存权重，并通过一次性初始化容器为非 root 进程设置目录权限。

创建研究时可发送 `Idempotency-Key` 请求头：同一用户、同一键和同一请求返回原任务；同一键对应不同请求返回 409。`POST /research/{thread_id}/cancel` 将排队、运行或审批中的任务标记为取消；运行中的外部调用不保证立即终止，后续回调和有效执行权检查阻止其继续提交受保护的任务状态。前端提供“取消任务”入口。自动恢复最多尝试 `MAX_RECOVERY_ATTEMPTS` 次；超过上限转为错误待人工排查。

## 评估入口

```powershell
uv run pytest -m "not eval" -q
uv run python -m evals.retrieval --output data/retrieval-evaluation.json
```

检索样例是人工编写的中英双语工程样例，报告文档级 Recall@K、MRR@K 和无答案查询表现；工作流轨迹另外检查节点覆盖、工具使用、错误与未闭合节点。这些结果不能当作真实论文质量成绩。

真实模型评测会产生 API 费用，仅在 RUN_LIVE_EVALS=true 且具备凭证时运行 `pytest -m eval`。Judge 以证据原文为依据，其分数仍需人工校准；旧公开网页问题集保留为独立冒烟评测。CI 默认只运行离线检查；付费 workflow 为手动触发。

## 必须理解的边界

1. 项目提供 local 与 production 两套运行配置：local 使用 SQLite 与线程池，适合学习调试；production 使用 PostgreSQL/pgvector、Redis 和 Celery，支持 API 与 Worker 分离扩展。
2. 任务和待投递消息通过 PostgreSQL Outbox 同事务提交；消息可能重复投递。Worker 通过任务状态和租约限制领取，指标、事件和终态写入校验当前令牌。LangGraph Checkpoint 与外部工具副作用尚未获得原子级执行权隔离，不能宣称完整 fencing 或端到端 exactly-once。
3. pgvector 使用 HNSW 进行向量候选召回，Cross-Encoder 只对有限候选重排；效果依赖真实语料、Embedding 与重排模型，不能由组件接入本身推导质量提升。
4. 调用预算在模型回调开始前检查，并在并行分支之间共享。任务超时是协作式截止时间；无法强杀已经阻塞的 Python 线程。暂停期间不计入本次执行段时间；模型累计次数在审批恢复后延续。
5. metrics 的 model_calls 为任务累计聊天模型次数；耗时、工具事件和 token 使用量为当前执行段，token 缺失表示未提供，不表示零费用。Embedding 不计入聊天模型调用数。
6. 文字 PDF 解析不包含 OCR、表格重建、公式理解与图像推理；网页工具使用搜索摘要，不声称阅读完整网页。
7. RRF 与 Cross-Encoder 分别承担排序融合和精排；项目不填写未经实测的准确率提升、延迟降幅或生产用户规模。
8. 删除文档会移除其检索索引，历史任务的证据快照仍保留。需彻底删除时还需设计任务、检查点与备份的数据保留策略。
9. 身份隔离是程序逻辑；提示词中的“不服从文档指令”只是额外防线，不能宣称彻底解决提示注入。
10. 真实模型、Embedding、Tavily 与容器运行的验收状态以 DEVELOPMENT.md 为准；代码完成不等于当前机器上的外部服务已经联调成功。
11. 取消是协作式的；已发出的模型/搜索请求可能继续运行并产生费用。租约过期后的自动重排依赖仍可用的 PostgreSQL、Dispatcher、Broker 与检查点，不保证所有外部副作用只发生一次。

项目保留原 MIT 许可证和上游来源说明；当前 README、接口与验收记录以本仓库实现为准。
