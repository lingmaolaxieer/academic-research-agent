"""Small single-process API-key authentication and per-principal rate limits."""
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from research_agent.config import settings


class BodyLimitMiddleware:
    """Bound multipart bytes before the upload parser writes an unbounded spool."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = settings.max_upload_bytes + 65536 if scope["path"] == "/documents" else 65536
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid content length"}, 400)(scope, receive, send)
        if declared > limit:
            return await JSONResponse({"detail": "Request body too large"}, 413)(scope, receive, send)
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > limit:
                raise HTTPException(413, "Request body too large")
            return message

        await self.app(scope, bounded_receive, send)


def install_security(app):
    app.add_middleware(BodyLimitMiddleware)
    windows = defaultdict(deque)
    lock = threading.Lock()
    redis_client = None
    rate_script = None
    if settings.rate_limit_backend == "redis":
        import redis
        redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        rate_script = redis_client.register_script("""
            local key = KEYS[1]
            local now = tonumber(ARGV[1])
            local cutoff = now - 60000
            local limit = tonumber(ARGV[2])
            redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
            if redis.call('ZCARD', key) >= limit then return 0 end
            redis.call('ZADD', key, now, ARGV[3])
            redis.call('PEXPIRE', key, 61000)
            return 1
        """)

    @app.middleware("http")
    async def authenticate(request, call_next):
        if request.url.path in {"/", "/health", "/ready", "/metrics", "/docs", "/openapi.json", "/favicon.ico"}:
            return await call_next(request)
        credential = request.headers.get("x-api-key", "")
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            credential = authorization[7:].strip()
        owner = None
        for principal, secret in settings.api_keys.items():
            if secret and secrets.compare_digest(credential.encode(), secret.encode()):
                owner = principal
        if owner is None and not settings.api_keys and settings.allow_anonymous:
            owner = "local"
        if owner is None:
            return JSONResponse({"detail": "Valid API key required"}, status_code=401)
        if rate_script is not None:
            try:
                now_ms = int(time.time() * 1000)
                member = f"{now_ms}:{secrets.token_hex(8)}"
                allowed = rate_script(
                    keys=[f"rate:api:{owner}"],
                    args=[now_ms, settings.requests_per_minute, member],
                )
            except Exception:
                return JSONResponse({"detail": "Rate limit service unavailable"}, status_code=503)
            if not allowed:
                return JSONResponse({"detail": "Request rate exceeded"}, status_code=429,
                                    headers={"Retry-After": "60"})
        else:
            now = time.monotonic()
            with lock:
                window = windows[owner]
                while window and window[0] <= now - 60:
                    window.popleft()
                if len(window) >= settings.requests_per_minute:
                    return JSONResponse({"detail": "Request rate exceeded"}, status_code=429,
                                        headers={"Retry-After": "60"})
                window.append(now)
        request.state.owner = owner
        return await call_next(request)
