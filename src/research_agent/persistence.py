"""Durable checkpointing for the research graph. 研究图的持久化检查点。

A **checkpointer** persists the full graph state after every superstep, keyed by
a ``thread_id``. That single addition is what makes the graph durable, and it
unlocks four first-class features at once:
**checkpointer** 会在每个 superstep 之后持久化完整的图状态，并以
``thread_id`` 作为键。这一项新增功能使图具备持久性，并一次解锁四项一等功能：

* **pause/resume** — stop mid-run and continue later with the same ``thread_id``;
* **pause/resume** —— 在运行中途停止，稍后使用相同的 ``thread_id`` 继续；
* **crash recovery** — if the process dies, the last checkpoint survives on disk;
* **crash recovery** —— 如果进程终止，最后一个检查点仍会保留在磁盘上；
* **time-travel** — replay the graph from any past checkpoint to debug;
* **time-travel** —— 从任意历史检查点重放图，以便调试；
* **human-in-the-loop** — ``interrupt()`` can only pause/resume because the state
  is durably saved between the pause and the human's reply.
* **human-in-the-loop** —— ``interrupt()`` 之所以能够暂停和恢复，是因为状态
  会在暂停与人类回复之间持久化保存。

This module is the one place that decides *which* backend stores those
checkpoints. The rest of the codebase just calls ``build_graph(checkpointer=...)``
and never cares about the storage details — that is the whole point of the
checkpointer abstraction.
本模块是唯一决定使用*哪个*后端存储这些检查点的地方。代码库的其余部分只需调用
``build_graph(checkpointer=...)``，无需关心存储细节——这正是 checkpointer
抽象的全部意义。

--------------------------------------------------------------------------------
DEV (what this project uses): SqliteSaver
--------------------------------------------------------------------------------
A single local file (``checkpoints.db`` by default). Zero setup, zero cost,
perfect for development, a single-process server, and this portfolio's scope.
默认使用单个本地文件（``checkpoints.db``）。无需配置、零成本，非常适合开发、
单进程服务器以及本项目的范围。

    from research_agent.persistence import sqlite_saver
    from research_agent.graph import build_graph

    with sqlite_saver() as saver:                 # opens the SQLite connection 打开 SQLite 连接
        app = build_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "run-123"}}
        app.invoke({"query": "..."}, config=cfg)  # state is now persisted 状态现已持久化

--------------------------------------------------------------------------------
PRODUCTION: use PostgresSaver instead
--------------------------------------------------------------------------------
SQLite is a single-file, single-writer store. A real deployment with multiple
workers/processes hammering the same database needs a real database. LangGraph's
supported production backend is Postgres, and — crucially — the *only* thing that
changes is the saver; the graph code is identical.
SQLite 是单文件、单写入者存储。多个 worker/进程频繁访问同一数据库的真实部署
需要使用真正的数据库。LangGraph 支持的生产后端是 Postgres，而且关键在于：
*唯一*需要改变的是 saver，图代码保持不变。

    # pip install langgraph-checkpoint-postgres 安装 langgraph-checkpoint-postgres
    import os
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as saver:
        saver.setup()                             # first run: create the tables 首次运行：创建表
        app = build_graph(checkpointer=saver)
        app.invoke({"query": "..."}, config={"configurable": {"thread_id": "run-123"}})

Any free Postgres works (a hosted free tier, or a local ``docker run postgres``).
For an async web server there is also ``AsyncPostgresSaver`` with the same API.
任何免费的 Postgres 都可以（托管的免费层，或本地执行
``docker run postgres``）。对于异步 Web 服务器，也可以使用具有相同 API 的
``AsyncPostgresSaver``。
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.config import settings


@contextmanager
def sqlite_saver(db_path: str | None = None) -> Iterator[SqliteSaver]:
    """Yield a ready-to-use ``SqliteSaver`` bound to ``db_path``.
    提供一个绑定到 ``db_path``、可直接使用的 ``SqliteSaver``。

    Use it as a context manager so the underlying SQLite connection is opened for
    the duration of the work and closed cleanly on exit:
    将其用作上下文管理器，使底层 SQLite 连接在工作期间保持打开，并在退出时
    正常关闭：

        with sqlite_saver() as saver:
            app = build_graph(checkpointer=saver)
            ...

    ``db_path`` defaults to ``settings.checkpoint_db_path``. The connection is
    owned by the ``with`` block — which is exactly why a long-lived web server
    (Stage 10) keeps the block open for its whole lifetime instead of per-request.
    ``db_path`` 默认为 ``settings.checkpoint_db_path``。连接由 ``with`` 代码块
    所有——因此长时间运行的 Web 服务器（Stage 10）会在整个生命周期内保持代码块
    打开，而不是为每个请求单独打开。

    Note: ``SqliteSaver.from_conn_string`` creates the checkpoint tables lazily on
    first use, so there is no separate ``setup()`` call to remember here.
    注意：``SqliteSaver.from_conn_string`` 会在首次使用时延迟创建检查点表，
    因此这里无需额外记住调用 ``setup()``。
    """
    path = db_path or settings.checkpoint_db_path
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(path) as saver:
        yield saver


@contextmanager
def checkpoint_saver():
    """Open the checkpointer selected by the runtime profile."""
    if settings.runtime_profile != "production":
        with sqlite_saver() as saver:
            yield saver
        return
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for the production profile")
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=max(4, settings.max_workers + 2),
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    with pool:
        saver = PostgresSaver(pool)
        saver.setup()
        yield saver
