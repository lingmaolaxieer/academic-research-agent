"""Durable task status, independent from LangGraph checkpoints.

SQLite adapter for one API process with a thread pool. This is not a distributed
queue: no automatic replay after a crash, no multi-process worker leases.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from research_agent.config import settings


class QueueCapacityReached(RuntimeError):
    pass


class LeaseLost(RuntimeError):
    """The current execution no longer owns this task."""


class IdempotencyConflict(ValueError):
    """An idempotency key was reused with different request content."""


class JobRegistry:
    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS research_jobs (
                thread_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                error TEXT,
                interrupt_json TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS research_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS research_idempotency (
                owner TEXT NOT NULL, request_key TEXT NOT NULL,
                request_hash TEXT NOT NULL, thread_id TEXT NOT NULL,
                PRIMARY KEY(owner, request_key)
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS research_events_thread ON research_events(thread_id,seq)"
        )
        self._db.commit()
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(research_jobs)")}
        if "owner" not in columns:
            self._db.execute("ALTER TABLE research_jobs ADD COLUMN owner TEXT NOT NULL DEFAULT 'local'")
        if "metrics_json" not in columns:
            self._db.execute("ALTER TABLE research_jobs ADD COLUMN metrics_json TEXT")
        if "lease_token" not in columns:
            self._db.execute("ALTER TABLE research_jobs ADD COLUMN lease_token TEXT")
        if "lease_until" not in columns:
            self._db.execute("ALTER TABLE research_jobs ADD COLUMN lease_until REAL")
        self._db.commit()

    def _event_locked(self, thread_id: str, kind: str, payload: dict | None = None) -> None:
        self._db.execute(
            "INSERT INTO research_events(thread_id,kind,payload_json) VALUES (?,?,?)",
            (thread_id, kind, json.dumps(payload or {})),
        )

    def start(self, thread_id: str, owner: str = "local", status: str = "running",
              outbox: dict | None = None) -> None:
        if status not in {"queued", "running"}:
            raise ValueError("Invalid initial job status")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO research_jobs(thread_id, status, owner) VALUES (?, ?, ?)",
                (thread_id, status, owner),
            )
            self._event_locked(thread_id, status)

    def start_idempotent(self, thread_id: str, owner: str, status: str,
                         request_key: str, request_hash: str, outbox=None) -> tuple[str, bool]:
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT thread_id,request_hash FROM research_idempotency "
                "WHERE owner=? AND request_key=?", (owner, request_key),
            ).fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise IdempotencyConflict("Idempotency key already used for another request")
                return row["thread_id"], False
            self._db.execute(
                "INSERT INTO research_jobs(thread_id,status,owner) VALUES (?,?,?)",
                (thread_id, status, owner),
            )
            self._db.execute(
                "INSERT INTO research_idempotency(owner,request_key,request_hash,thread_id) "
                "VALUES (?,?,?,?)", (owner, request_key, request_hash, thread_id),
            )
            self._event_locked(thread_id, status)
            return thread_id, True

    def lookup_idempotency(self, owner: str, request_key: str):
        with self._lock:
            row = self._db.execute(
                "SELECT thread_id,request_hash FROM research_idempotency "
                "WHERE owner=? AND request_key=?", (owner, request_key),
            ).fetchone()
            return (row["thread_id"], row["request_hash"]) if row else None

    def finish(self, thread_id: str, token: str | None = None) -> bool:
        return self._update(thread_id, "completed", token=token)

    def interrupt(self, thread_id: str, payload: dict, token: str | None = None) -> bool:
        return self._update(thread_id, "interrupted", payload=payload, token=token)

    def fail(self, thread_id: str, error: str, token: str | None = None) -> bool:
        return self._update(thread_id, "error", error=error, token=token)

    def _update(self, thread_id: str, status: str, error=None, payload=None,
                token: str | None = None) -> bool:
        with self._lock, self._db:
            where = "thread_id=? AND status='running'"
            args = [status, error, json.dumps(payload) if payload is not None else None, thread_id]
            if token is not None:
                where += " AND lease_token=? AND lease_until>?"
                args.extend([token, time.time()])
            changed = self._db.execute(
                "UPDATE research_jobs SET status=?, error=?, interrupt_json=?, "
                "lease_token=NULL,lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE " + where,
                args,
            )
            if changed.rowcount:
                self._event_locked(thread_id, status, {"error": error} if error else payload)
            return changed.rowcount == 1

    def cancel(self, thread_id: str) -> bool:
        with self._lock, self._db:
            changed = self._db.execute(
                "UPDATE research_jobs SET status='cancelled',lease_token=NULL,lease_until=NULL,"
                "updated_at=CURRENT_TIMESTAMP WHERE thread_id=? "
                "AND status IN ('queued','running','interrupted')", (thread_id,),
            )
            if changed.rowcount:
                self._event_locked(thread_id, "cancelled")
            return changed.rowcount == 1

    def is_cancelled(self, thread_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM research_jobs WHERE thread_id=?", (thread_id,),
            ).fetchone()
            return bool(row and row["status"] == "cancelled")

    def assert_execution(self, thread_id: str, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM research_jobs WHERE thread_id=? AND status='running' "
                "AND lease_token=? AND lease_until>?", (thread_id, token, time.time()),
            ).fetchone()
            if not row:
                raise LeaseLost("Task execution lease is no longer valid")

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM research_jobs WHERE thread_id=?", (thread_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "owner": row["owner"],
            "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            "status": row["status"],
            "error": row["error"],
            "interrupt": json.loads(row["interrupt_json"]) if row["interrupt_json"] else None,
        }

    def set_metrics(self, thread_id: str, metrics: dict,
                    token: str | None = None) -> None:
        with self._lock, self._db:
            sql = "UPDATE research_jobs SET metrics_json=? WHERE thread_id=?"
            args = [json.dumps(metrics), thread_id]
            if token is not None:
                sql += " AND status='running' AND lease_token=? AND lease_until>?"
                args.extend([token, time.time()])
            if self._db.execute(sql, args).rowcount != 1 and token is not None:
                raise LeaseLost("Task execution lease is no longer valid")

    def append_event(self, thread_id: str, kind: str, payload: dict | None = None,
                     token: str | None = None) -> None:
        with self._lock, self._db:
            if token is not None:
                row = self._db.execute(
                    "SELECT 1 FROM research_jobs WHERE thread_id=? AND status='running' "
                    "AND lease_token=? AND lease_until>?", (thread_id, token, time.time()),
                ).fetchone()
                if not row:
                    raise LeaseLost("Task execution lease is no longer valid")
            self._event_locked(thread_id, kind, payload)
            self._db.execute("""
                DELETE FROM research_events WHERE thread_id=? AND seq NOT IN (
                    SELECT seq FROM research_events WHERE thread_id=? ORDER BY seq DESC LIMIT ?
                )
            """, (thread_id, thread_id, settings.event_retention))

    def events(self, thread_id: str, after: int = 0, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("""
                SELECT seq,kind,payload_json,created_at FROM research_events
                WHERE thread_id=? AND seq>? ORDER BY seq LIMIT ?
            """, (thread_id, after, limit)).fetchall()
        return [{"seq": row["seq"], "kind": row["kind"],
                 "payload": json.loads(row["payload_json"]),
                 "created_at": row["created_at"]} for row in rows]

    def list_jobs(self, owner: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT thread_id,status,updated_at FROM research_jobs WHERE owner=? "
                "ORDER BY updated_at DESC LIMIT 100", (owner,)).fetchall()
        return [dict(row) for row in rows]

    def claim_resume(self, thread_id: str, status: str = "running",
                     outbox: dict | None = None) -> bool:
        """Only one concurrent request can change interrupted -> running."""
        with self._lock, self._db:
            changed = self._db.execute(
                "UPDATE research_jobs SET status=?, error=NULL, interrupt_json=NULL, "
                "updated_at=CURRENT_TIMESTAMP WHERE thread_id=? AND status='interrupted'",
                (status, thread_id),
            )
            if changed.rowcount == 1:
                self._event_locked(thread_id, status, {"resumed": True})
            return changed.rowcount == 1

    def claim_execution(self, thread_id: str, worker_id: str | None = None) -> str | None:
        """Acquire a fenced worker lease, allowing retry only after lease expiry."""
        token = f"{worker_id or 'worker'}:{uuid.uuid4().hex}"
        now = time.time()
        with self._lock, self._db:
            changed = self._db.execute("""
                UPDATE research_jobs SET status='running',lease_token=?,lease_until=?,
                updated_at=CURRENT_TIMESTAMP WHERE thread_id=? AND
                (status='queued' OR (status='running' AND lease_until < ?))
            """, (token, now + settings.worker_lease_seconds, thread_id, now))
            if changed.rowcount == 1:
                self._event_locked(thread_id, "running", {"lease": token})
                return token
        return None

    def renew_lease(self, thread_id: str, token: str) -> bool:
        with self._lock, self._db:
            changed = self._db.execute(
                "UPDATE research_jobs SET lease_until=? WHERE thread_id=? "
                "AND status='running' AND lease_token=? AND lease_until>?",
                (time.time() + settings.worker_lease_seconds, thread_id, token, time.time()),
            )
            return changed.rowcount == 1

    def recover_stale(self) -> int:
        """Call once at single-process startup; never pretend stopped work is HITL."""
        with self._lock, self._db:
            changed = self._db.execute(
                "UPDATE research_jobs SET status='error', error=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE status='running'",
                ("Execution interrupted by process restart; no automatic replay was performed.",),
            )
            return changed.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()


class PostgresJobRegistry:
    """Shared durable task registry, event log and fenced worker leases."""

    def __init__(self, database_url: str | None = None) -> None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.database_url = database_url or settings.database_url
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for Postgres jobs")
        self._pool = ConnectionPool(
            conninfo=self.database_url, min_size=1, max_size=10,
            kwargs={"row_factory": dict_row},
        )
        self.setup()

    def connect(self):
        return self._pool.connection()

    def setup(self):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_jobs (
                    thread_id text PRIMARY KEY, owner text NOT NULL,
                    status text NOT NULL, error text, interrupt_json jsonb,
                    metrics_json jsonb, lease_token text, lease_until timestamptz,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS research_jobs_owner_idx ON research_jobs(owner,updated_at DESC)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_idempotency (
                    owner text NOT NULL, request_key text NOT NULL,
                    request_hash text NOT NULL, thread_id text NOT NULL,
                    PRIMARY KEY(owner, request_key)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_events (
                    seq bigserial PRIMARY KEY, thread_id text NOT NULL,
                    kind text NOT NULL, payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS research_events_thread_idx ON research_events(thread_id,seq)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_outbox (
                    id bigserial PRIMARY KEY, message_id text NOT NULL UNIQUE,
                    thread_id text NOT NULL, payload_json jsonb NOT NULL,
                    status text NOT NULL DEFAULT 'pending', attempts integer NOT NULL DEFAULT 0,
                    available_at timestamptz NOT NULL DEFAULT now(),
                    locked_until timestamptz, created_at timestamptz NOT NULL DEFAULT now(),
                    sent_at timestamptz, publish_token text
                )
            """)
            cur.execute("ALTER TABLE task_outbox ADD COLUMN IF NOT EXISTS publish_token text")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS task_outbox_pending_idx
                ON task_outbox(status,available_at) WHERE status='pending'
            """)

    @staticmethod
    def _event(cur, thread_id: str, kind: str, payload=None):
        from psycopg.types.json import Jsonb
        cur.execute("""
            INSERT INTO research_events(thread_id,kind,payload_json) VALUES (%s,%s,%s)
        """, (thread_id, kind, Jsonb(payload or {})))

    @staticmethod
    def _outbox(cur, thread_id: str, payload: dict | None):
        if payload is None:
            return
        from psycopg.types.json import Jsonb
        cur.execute("""
            INSERT INTO task_outbox(message_id,thread_id,payload_json)
            VALUES (%s,%s,%s) ON CONFLICT(message_id) DO NOTHING
        """, (payload["message_id"], thread_id, Jsonb(payload)))

    def start(self, thread_id: str, owner: str = "local", status: str = "running",
              outbox: dict | None = None):
        with self.connect() as db, db.cursor() as cur:
            if status == "queued":
                cur.execute("SELECT pg_advisory_xact_lock(hashtext('research_queue_capacity'))")
                cur.execute("""
                    SELECT count(*) AS count FROM research_jobs
                    WHERE status IN ('queued','running')
                """)
                if cur.fetchone()["count"] >= settings.max_pending_jobs:
                    raise QueueCapacityReached("Research capacity reached")
            cur.execute("INSERT INTO research_jobs(thread_id,owner,status) VALUES (%s,%s,%s)",
                        (thread_id, owner, status))
            self._outbox(cur, thread_id, outbox)
            self._event(cur, thread_id, status)

    def start_idempotent(self, thread_id, owner, status, request_key,
                         request_hash, outbox=None) -> tuple[str, bool]:
        with self.connect() as db, db.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (owner + ":" + request_key,))
            cur.execute("SELECT thread_id,request_hash FROM research_idempotency "
                        "WHERE owner=%s AND request_key=%s", (owner, request_key))
            row = cur.fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise IdempotencyConflict("Idempotency key already used for another request")
                return row["thread_id"], False
            if status == "queued":
                cur.execute("SELECT pg_advisory_xact_lock(hashtext('research_queue_capacity'))")
                cur.execute("SELECT count(*) AS count FROM research_jobs "
                            "WHERE status IN ('queued','running')")
                if cur.fetchone()["count"] >= settings.max_pending_jobs:
                    raise QueueCapacityReached("Research capacity reached")
            cur.execute("INSERT INTO research_jobs(thread_id,owner,status) VALUES (%s,%s,%s)",
                        (thread_id, owner, status))
            self._outbox(cur, thread_id, outbox)
            cur.execute("INSERT INTO research_idempotency(owner,request_key,request_hash,thread_id) "
                        "VALUES (%s,%s,%s,%s)", (owner, request_key, request_hash, thread_id))
            self._event(cur, thread_id, status)
            return thread_id, True

    def lookup_idempotency(self, owner, request_key):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("SELECT thread_id,request_hash FROM research_idempotency "
                        "WHERE owner=%s AND request_key=%s", (owner, request_key))
            row = cur.fetchone()
            return (row["thread_id"], row["request_hash"]) if row else None

    def _update(self, thread_id, status, error=None, payload=None, token=None):
        from psycopg.types.json import Jsonb
        with self.connect() as db, db.cursor() as cur:
            condition = "thread_id=%s AND status='running'"
            args = [status, error, Jsonb(payload) if payload is not None else None, thread_id]
            if token is not None:
                condition += " AND lease_token=%s AND lease_until>now()"
                args.append(token)
            cur.execute("""
                UPDATE research_jobs SET status=%s,error=%s,interrupt_json=%s,
                lease_token=NULL,lease_until=NULL,updated_at=now() WHERE
            """ + condition, args)
            if cur.rowcount == 1:
                self._event(cur, thread_id, status, {"error": error} if error else payload)
                return True
            return False

    def finish(self, thread_id, token=None):
        return self._update(thread_id, "completed", token=token)

    def interrupt(self, thread_id, payload, token=None):
        return self._update(thread_id, "interrupted", payload=payload, token=token)

    def fail(self, thread_id, error, token=None):
        return self._update(thread_id, "error", error=error, token=token)

    def cancel(self, thread_id):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                UPDATE research_jobs SET status='cancelled',lease_token=NULL,
                lease_until=NULL,updated_at=now() WHERE thread_id=%s
                AND status IN ('queued','running','interrupted')
            """, (thread_id,))
            changed = cur.rowcount == 1
            if changed:
                self._event(cur, thread_id, "cancelled")
            return changed

    def is_cancelled(self, thread_id):
        job = self.get(thread_id)
        return bool(job and job["status"] == "cancelled")

    def assert_execution(self, thread_id, token):
        if token is None:
            return
        with self.connect() as db, db.cursor() as cur:
            cur.execute("SELECT 1 FROM research_jobs WHERE thread_id=%s "
                        "AND status='running' AND lease_token=%s AND lease_until>now()",
                        (thread_id, token))
            if cur.fetchone() is None:
                raise LeaseLost("Task execution lease is no longer valid")

    def get(self, thread_id):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("SELECT * FROM research_jobs WHERE thread_id=%s", (thread_id,))
            row = cur.fetchone()
        if not row:
            return None
        return {"owner": row["owner"], "metrics": row["metrics_json"] or {},
                "status": row["status"], "error": row["error"],
                "interrupt": row["interrupt_json"]}

    def set_metrics(self, thread_id, metrics, token=None):
        from psycopg.types.json import Jsonb
        with self.connect() as db, db.cursor() as cur:
            condition = "thread_id=%s"
            args = [Jsonb(metrics), thread_id]
            if token is not None:
                condition += " AND status='running' AND lease_token=%s AND lease_until>now()"
                args.append(token)
            cur.execute("UPDATE research_jobs SET metrics_json=%s WHERE " + condition, args)
            if cur.rowcount != 1 and token is not None:
                raise LeaseLost("Task execution lease is no longer valid")

    def append_event(self, thread_id, kind, payload=None, token=None):
        with self.connect() as db, db.cursor() as cur:
            if token is not None:
                cur.execute("SELECT 1 FROM research_jobs WHERE thread_id=%s "
                            "AND status='running' AND lease_token=%s "
                            "AND lease_until>now() FOR UPDATE", (thread_id, token))
                if cur.fetchone() is None:
                    raise LeaseLost("Task execution lease is no longer valid")
            self._event(cur, thread_id, kind, payload)
            cur.execute("""
                DELETE FROM research_events WHERE thread_id=%s AND seq NOT IN (
                    SELECT seq FROM research_events WHERE thread_id=%s
                    ORDER BY seq DESC LIMIT %s
                )
            """, (thread_id, thread_id, settings.event_retention))

    def events(self, thread_id, after=0, limit=100):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                SELECT seq,kind,payload_json AS payload,created_at FROM research_events
                WHERE thread_id=%s AND seq>%s ORDER BY seq LIMIT %s
            """, (thread_id, after, limit))
            return list(cur.fetchall())

    def list_jobs(self, owner):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                SELECT thread_id,status,updated_at FROM research_jobs
                WHERE owner=%s ORDER BY updated_at DESC LIMIT 100
            """, (owner,))
            return list(cur.fetchall())

    def claim_resume(self, thread_id, status="running", outbox=None):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                UPDATE research_jobs SET status=%s,error=NULL,interrupt_json=NULL,
                updated_at=now() WHERE thread_id=%s AND status='interrupted'
            """, (status, thread_id))
            if cur.rowcount == 1:
                self._outbox(cur, thread_id, outbox)
                self._event(cur, thread_id, status, {"resumed": True})
                return True
            return False

    def claim_execution(self, thread_id, worker_id=None):
        token = f"{worker_id or 'worker'}:{uuid.uuid4().hex}"
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                UPDATE research_jobs SET status='running',lease_token=%s,
                lease_until=now()+(%s * interval '1 second'),updated_at=now()
                WHERE thread_id=%s AND
                (status='queued' OR (status='running' AND lease_until < now()))
            """, (token, settings.worker_lease_seconds, thread_id))
            if cur.rowcount == 1:
                self._event(cur, thread_id, "running", {"lease": token})
                return token
            return None

    def renew_lease(self, thread_id, token):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                UPDATE research_jobs SET lease_until=now()+(%s * interval '1 second')
                WHERE thread_id=%s AND status='running' AND lease_token=%s
                AND lease_until>now()
            """, (settings.worker_lease_seconds, thread_id, token))
            return cur.rowcount == 1

    def recover_stale(self, limit: int = 20):
        """Atomically queue expired executions using the last durable message."""
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                SELECT thread_id FROM research_jobs
                WHERE status='running' AND lease_until<now()
                ORDER BY lease_until FOR UPDATE SKIP LOCKED LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            for row in rows:
                thread_id = row["thread_id"]
                cur.execute("SELECT payload_json FROM task_outbox WHERE thread_id=%s "
                            "ORDER BY id DESC LIMIT 1", (thread_id,))
                previous = cur.fetchone()
                if not previous:
                    cur.execute("UPDATE research_jobs SET status='error',error=%s,"
                                "lease_token=NULL,lease_until=NULL,updated_at=now() "
                                "WHERE thread_id=%s",
                                ("Recovery input missing; manual inspection required.", thread_id))
                    self._event(cur, thread_id, "error", {"reason": "recovery_input_missing"})
                    continue
                payload = dict(previous["payload_json"])
                recovery_count = int(payload.get("recovery_count", 0)) + 1
                if recovery_count > settings.max_recovery_attempts:
                    cur.execute("UPDATE research_jobs SET status='error',error=%s,"
                                "lease_token=NULL,lease_until=NULL,updated_at=now() "
                                "WHERE thread_id=%s",
                                ("Automatic recovery limit exceeded; inspect task and dependencies.",
                                 thread_id))
                    self._event(cur, thread_id, "error", {"reason": "recovery_limit"})
                    continue
                payload["message_id"] = uuid.uuid4().hex
                payload["recover"] = True
                payload["recovery_count"] = recovery_count
                cur.execute("UPDATE research_jobs SET status='queued',lease_token=NULL,"
                            "lease_until=NULL,updated_at=now() WHERE thread_id=%s", (thread_id,))
                self._outbox(cur, thread_id, payload)
                self._event(cur, thread_id, "queued", {"reason": "lease_expired"})
            return len(rows)

    def claim_outbox(self, limit: int = 20):
        """Claim messages with SKIP LOCKED so multiple dispatchers may run."""
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                WITH candidates AS (
                    SELECT id FROM task_outbox
                    WHERE (status='pending' AND available_at <= now())
                       OR (status='publishing' AND locked_until < now())
                    ORDER BY id FOR UPDATE SKIP LOCKED LIMIT %s
                )
                UPDATE task_outbox o SET status='publishing',attempts=attempts+1,
                    locked_until=now()+interval '60 seconds',
                    publish_token=md5(random()::text || clock_timestamp()::text || o.id::text)
                FROM candidates c WHERE o.id=c.id
                RETURNING o.id,o.message_id,o.thread_id,o.payload_json AS payload,
                          o.attempts,o.publish_token
            """, (limit,))
            return list(cur.fetchall())

    def mark_outbox_sent(self, outbox_id: int, publish_token: str):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                UPDATE task_outbox SET status='sent',sent_at=now(),locked_until=NULL,
                publish_token=NULL WHERE id=%s AND status='publishing'
                AND publish_token=%s AND locked_until>now()
            """, (outbox_id, publish_token))
            return cur.rowcount == 1

    def retry_outbox(self, outbox_id: int, attempts: int, publish_token: str):
        delay = min(300, 2 ** min(attempts, 8))
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                UPDATE task_outbox SET status='pending',locked_until=NULL,
                publish_token=NULL,available_at=now()+(%s * interval '1 second')
                WHERE id=%s AND status='publishing' AND publish_token=%s
                AND locked_until>now()
            """, (delay, outbox_id, publish_token))
            return cur.rowcount == 1

    def close(self):
        self._pool.close()


def create_job_registry(path: str | None = None):
    if settings.runtime_profile == "production":
        return PostgresJobRegistry(settings.database_url)
    return JobRegistry(path or settings.job_db_path)
