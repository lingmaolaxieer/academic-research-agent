# Security Policy

## Reporting a vulnerability

Report reproducible security issues through the repository issue tracker without including credentials, private documents, or provider responses.

## Deployment model

The local profile allows anonymous access for single-user development. The production Compose profile disables anonymous access and requires `API_KEYS`, mapping each principal to an independent secret. Clients authenticate with `X-API-Key` or a bearer token. The application derives document, task, preference, and retrieval ownership from that authenticated principal.

The production API binds to `127.0.0.1` by default. A public deployment still requires HTTPS termination, an explicit CORS policy, secret rotation, network controls, backups, and retention rules appropriate to its environment.

## Abuse and cost controls

Local mode uses an in-process rate limiter. Production mode uses Redis-backed principal rate limiting across API instances. Query, upload, document, evidence, model-call, recursion, retry, and task-capacity limits bound common abuse and runaway-cost paths. These controls reduce risk but do not replace provider-side budgets and monitoring.

## Secrets

Credentials are supplied through environment variables or a local `.env`. They are excluded from version control and Docker build context. Never commit `.env`, provider keys, database passwords, exported logs, or uploaded research material. Replace the example API key and database password before shared deployment.

## Data handling

Uploaded documents, chunks, embeddings, preferences, task state, events, and LangGraph checkpoints are persisted in SQLite for local mode or PostgreSQL/pgvector for production mode. Redis carries task and rate-limit state. Docker named volumes retain production data after container recreation.

Document excerpts and user queries may be sent to the configured chat-model, embedding, search, tracing, or evaluation providers. Review those providers' data policies before processing sensitive material. Deleting a document removes its current retrieval index; historical task evidence, checkpoints, logs, monitoring data, and backups require a separate retention policy.

## Agent-specific boundaries

Tool output and retrieved documents are treated as untrusted data, and citation identifiers must come from gathered evidence. These checks do not prove factual correctness, semantic entailment, or complete resistance to prompt injection. Restrict tool permissions and review high-impact actions before adding tools with external side effects.
