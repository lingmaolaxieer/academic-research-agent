# Production image for the Academic Research Agent API and workers.
# 学术资料调研助手 API 与 Worker 的生产镜像。
#
# uv variant: copy the uv binary from its official image, then install from the
# committed uv.lock for a reproducible build. `--locked` makes the build fail
# loudly if the lockfile is out of date with pyproject.toml.
# uv 版本：从官方镜像复制 uv 二进制文件，然后根据已提交的 uv.lock 安装依赖，
# 以实现可复现构建。`--frozen` 会在锁文件相对于 pyproject.toml 过期时明确使构建失败。
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /bin/uv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first so this layer is cached and only re-runs when
# the dependencies change — not on every source edit.
# 首先复制依赖清单，使该层能够被缓存；只有依赖发生变化时才重新运行，
# 而不是每次源代码编辑后都重新运行。
COPY pyproject.toml uv.lock ./

# `--no-dev` skips the dev group (pytest); installing the project also puts the
# `research_agent` package on the import path, so no PYTHONPATH is needed.
# `--no-dev` 跳过开发依赖组（pytest）；安装项目也会将 `research_agent` 包
# 加入导入路径，因此不需要设置 PYTHONPATH。
ARG INSTALL_RERANKER=false
ENV UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_RERANKER" = "true" ]; then \
      uv sync --locked --no-dev --extra rerank --no-install-project; \
    else \
      uv sync --locked --no-dev --no-install-project; \
    fi

# Keep the expensive dependency layer valid when only application code changes.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_RERANKER" = "true" ]; then \
      uv sync --locked --no-dev --extra rerank; \
    else \
      uv sync --locked --no-dev; \
    fi
RUN useradd --create-home researcher && mkdir -p /app/data && chown -R researcher:researcher /app
USER researcher
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 0.0.0.0 so the server is reachable from outside the container.
# 使用 0.0.0.0，使服务器能够从容器外部访问。
CMD ["/app/.venv/bin/uvicorn", "research_agent.api:api", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
