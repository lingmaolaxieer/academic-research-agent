# 开发验收状态 · 2026-10-30

本文保留 2026-10-17 的生产服务验收记录，并记录 2026-10-30 的后续修复。后续修复在隔离的源码副本中通过离线测试；当前 Docker Desktop 引擎不可用，未完成新镜像构建和 PostgreSQL/Redis/Celery 联调。旧镜像与旧容器不能视为包含本次修复。

## 2026-10-30 后续修复

- 为研究创建接口增加按用户隔离的 `Idempotency-Key`。同请求重试复用任务 ID，不同请求复用同一键返回 409。
- 增加取消接口及页面入口，排队、运行和审批任务可标记取消；运行中调用在回调边界协作停止，外部请求不保证立即中断。
- Worker 的任务终态、指标与事件写入校验租约令牌；Outbox 投递状态写入校验本次领取令牌。
- Dispatcher 周期扫描过期执行租约，将任务与恢复消息重新排队，恢复次数由 `MAX_RECOVERY_ATTEMPTS` 限制。检查点与外部服务的副作用仍需要更严格的执行权隔离或补偿。
- 引用修复耗尽后明确失败，不再向报告末尾附加证据索引来掩盖正文缺少逐条引用的问题；研究者模型调用异常也不再被当作可用的空分支吞掉。
- `ruff check src tests`、前端 JavaScript 语法和 Compose 配置检查通过；离线回归 68 项通过、4 项付费评估跳过。PostgreSQL、Redis、Celery 和生产镜像需要待 Docker 引擎可用后再次验收。

## 2026-10-17 基线记录

## 最终架构实现

- 保留 SQLite + 线程池本地配置，并新增 PostgreSQL/pgvector + Redis/Celery 生产配置。
- PostgreSQL 存储文档、向量、偏好、任务、持久事件和事务 Outbox；LangGraph 使用 PostgresSaver。
- pgvector HNSW 向量检索与 BM25 召回经 RRF 融合，可配置 Sentence Transformers Cross-Encoder 精排。
- API、Outbox 投递器与 Worker 分离；Outbox 与任务同事务提交，Worker 使用带续约的执行租约，重复消息不会并发执行同一任务。
- 持久事件支持带 `Last-Event-ID` 的 SSE；页面使用带 API Key 的 Fetch 流并保留轮询降级。
- Redis 用作 Celery Broker 和跨 API 实例限流；任务、文档、偏好、MCP 检索均保持 owner 隔离。
- 增加快速问答/深度调研路由；快速路径跳过规划与评审，深度路径保留规划、并发研究、评审补查与 HITL。
- 提供结构化 JSON 日志、Prometheus 指标端点和 Compose Prometheus 服务；LangSmith 继续作为可选链路追踪。
- Compose 包含 API、Worker、Outbox、PostgreSQL/pgvector、Redis 与 Prometheus，数据库和 Redis 不暴露宿主机端口。

## 当前验证

- Python 静态检查通过；离线回归 59 项通过、4 项真实模型评估跳过。
- 新增任务事件/租约、Cross-Encoder 顺序、Outbox 稳定消息 ID、成本路由、轨迹评估和 SSE 续传的离线测试。
- uv 锁文件已更新，解析 139 个包；PyTorch 明确使用 CPU wheel，避免生产镜像引入 CUDA 运行库。可选重排依赖由生产镜像安装。
- Compose 配置解析通过；页面最新 JavaScript 由 Node 24 解析通过。
- 生产镜像已实际构建；API、Worker、Outbox、PostgreSQL/pgvector、Redis、Prometheus 六个常驻服务均已启动。`/health`、`/ready` 与 Prometheus readiness 返回成功，三个 Prometheus 抓取目标均为 `up`。
- PostgreSQL 已实际创建 vector 扩展、HNSW/全文索引、任务/事件/Outbox、用户偏好与 LangGraph checkpoint 表。合成向量经生产 `PostgresKnowledgeStore` 完成 hybrid 检索并清理测试文档。
- 已验证无密钥返回 401、有效测试密钥可访问、用户偏好写入/读取/清除、Redis 分布式限流键、Celery worker ping，以及 PostgreSQL Outbox → Redis → Worker 的消息发布和消费链路。
- 已通过生产 API 完成持久 SSE 事件回放。临时数据库行、Outbox 消息和 Redis 测试键均已清理。
- 生产容器已下载并运行配置的 Cross-Encoder 权重；目标检测相关文本排在无关文本之前。共享 Hugging Face 数据卷经非 root 权限初始化，Worker 容器重建后在 `HF_HUB_OFFLINE=1` 下仍可完成推理。
- 未使用真实聊天模型、Embedding 或 Tavily 执行完整研究任务；不宣称真实回答质量、召回提升、延迟或并发指标。
