"""PostgreSQL/pgvector knowledge store for the production runtime profile."""
from __future__ import annotations

import hashlib
import math
import threading
from collections import Counter

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from research_agent.config import settings
from research_agent.evidence import Evidence, evidence_id
from research_agent.knowledge import chunks, embedding_client, parse_document, tokens
from research_agent.rerank import rerank

_SETUP_LOCK = threading.Lock()
_SETUP_KEYS: set[tuple[str, int]] = set()
_POOLS: dict[str, ConnectionPool] = {}


def _vector_literal(values: list[float]) -> str:
    if len(values) != settings.embedding_dimensions or not all(math.isfinite(x) for x in values):
        raise ValueError("Embedding dimension mismatch or non-finite vector")
    return "[" + ",".join(format(float(value), ".12g") for value in values) + "]"


class PostgresKnowledgeStore:
    """Owner-scoped documents with lexical, pgvector and RRF retrieval."""

    def __init__(self, database_url: str | None = None, embedder=None, *, setup: bool = True):
        self.database_url = database_url or settings.database_url
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for the production profile")
        self.embedder = embedder
        with _SETUP_LOCK:
            if self.database_url not in _POOLS:
                _POOLS[self.database_url] = ConnectionPool(
                    conninfo=self.database_url, min_size=1, max_size=10,
                    kwargs={"row_factory": dict_row},
                )
            self.pool = _POOLS[self.database_url]
        if setup:
            key = (self.database_url, settings.embedding_dimensions)
            with _SETUP_LOCK:
                if key not in _SETUP_KEYS:
                    self.setup()
                    _SETUP_KEYS.add(key)

    def connect(self):
        return self.pool.connection()

    def setup(self) -> None:
        dimension = int(settings.embedding_dimensions)
        with self.connect() as db, db.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id text PRIMARY KEY, owner text NOT NULL, name text NOT NULL,
                    content_hash text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE(owner, content_hash, name)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS documents_owner_idx ON documents(owner)")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id text PRIMARY KEY,
                    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page integer NOT NULL, chunk_offset integer NOT NULL, text text NOT NULL,
                    embedding vector({dimension}), embedding_model text
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS document_chunks_document_idx ON document_chunks(document_id)")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_text_idx
                ON document_chunks USING gin (to_tsvector('simple', text))
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    owner text PRIMARY KEY, instructions text NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
            """)

    def add(self, owner: str, filename: str, data: bytes) -> dict:
        name = filename.replace("\\", "/").split("/")[-1][:200]
        pages = parse_document(name, data)
        digest = hashlib.sha256(data).hexdigest()
        doc_id = "doc_" + hashlib.sha256(f"{owner}\0{name}\0{digest}".encode()).hexdigest()[:24]
        parts = [(page, offset, text) for page, body in pages
                 for offset, text in chunks(body, settings.chunk_size, settings.chunk_overlap)]
        if len(parts) > 1000:
            raise ValueError("Too many document chunks")
        with self.connect() as db, db.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE id=%s AND owner=%s", (doc_id, owner))
            if cur.fetchone():
                return {"document_id": doc_id, "name": name, "chunks": len(parts), "duplicate": True}
            cur.execute("""
                SELECT count(*) AS count FROM document_chunks c
                JOIN documents d ON d.id=c.document_id WHERE d.owner=%s
            """, (owner,))
            if cur.fetchone()["count"] + len(parts) > 10000:
                raise ValueError("Owner chunk limit reached; delete documents before adding more")
        vectors = None
        if settings.retrieval_mode in {"dense", "hybrid"}:
            vectors = (self.embedder or embedding_client()).embed_documents([part[2] for part in parts])
            if len(vectors) != len(parts):
                raise ValueError("Embedding response size mismatch")
            vectors = [_vector_literal(vector) for vector in vectors]
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                INSERT INTO documents(id,owner,name,content_hash) VALUES (%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, (doc_id, owner, name, digest))
            for index, (page, offset, text) in enumerate(parts):
                vector = vectors[index] if vectors else None
                cur.execute("""
                    INSERT INTO document_chunks
                    (id,document_id,page,chunk_offset,text,embedding,embedding_model)
                    VALUES (%s,%s,%s,%s,%s,%s::vector,%s) ON CONFLICT DO NOTHING
                """, (f"{doc_id}_{index}", doc_id, page, offset, text, vector,
                      settings.embedding_model if vector else None))
        return {"document_id": doc_id, "name": name, "chunks": len(parts), "duplicate": False}

    def list_documents(self, owner: str) -> list[dict]:
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                SELECT d.id AS document_id,d.name,d.created_at,count(c.id) AS chunks
                FROM documents d LEFT JOIN document_chunks c ON c.document_id=d.id
                WHERE d.owner=%s GROUP BY d.id ORDER BY d.created_at DESC
            """, (owner,))
            return list(cur.fetchall())

    def delete(self, owner: str, doc_id: str) -> bool:
        with self.connect() as db, db.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id=%s AND owner=%s", (doc_id, owner))
            return cur.rowcount == 1

    def _rows(self, owner: str, document_ids=None) -> list[dict]:
        sql = """
            SELECT c.id,c.document_id,c.page,c.chunk_offset AS offset,c.text,c.embedding_model,
                   c.embedding IS NOT NULL AS has_embedding,d.name
            FROM document_chunks c JOIN documents d ON d.id=c.document_id WHERE d.owner=%s
        """
        params: list = [owner]
        if document_ids:
            sql += " AND d.id = ANY(%s)"
            params.append(list(document_ids))
        sql += " ORDER BY c.id"
        with self.connect() as db, db.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def _dense_ranking(self, owner: str, query_vector: str, document_ids, limit: int):
        sql = """
            SELECT c.id, 1 - (c.embedding <=> %s::vector) AS dense_score
            FROM document_chunks c JOIN documents d ON d.id=c.document_id
            WHERE d.owner=%s AND c.embedding IS NOT NULL AND c.embedding_model=%s
        """
        params: list = [query_vector, owner, settings.embedding_model]
        if document_ids:
            sql += " AND d.id = ANY(%s)"
            params.append(list(document_ids))
        sql += " ORDER BY c.embedding <=> %s::vector, c.id LIMIT %s"
        params.extend([query_vector, limit])
        with self.connect() as db, db.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def search(self, owner: str, query: str, top_k: int = 5, document_ids=None, mode=None):
        if not query.strip() or len(query) > 2000 or not 1 <= top_k <= 20:
            raise ValueError("Invalid query or top_k")
        mode = mode or settings.retrieval_mode
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        rows = self._rows(owner, document_ids)
        if not rows:
            return []
        by_id = {row["id"]: index for index, row in enumerate(rows)}
        limit = max(settings.reranker_candidates, top_k * 4)
        rankings: list[list[int]] = []
        if mode in {"bm25", "hybrid"}:
            bags = [Counter(tokens(row["text"])) for row in rows]
            lengths = [sum(bag.values()) for bag in bags]
            average = sum(lengths) / len(rows) or 1
            scores = [0.0] * len(rows)
            for term in set(tokens(query)):
                frequency = sum(term in bag for bag in bags)
                inverse = math.log(1 + (len(rows) - frequency + .5) / (frequency + .5))
                for index, bag in enumerate(bags):
                    term_count = bag[term]
                    scores[index] += inverse * term_count * 2.5 / (
                        term_count + 1.5 * (.25 + .75 * lengths[index] / average)
                    )
            rankings.append(sorted((i for i, score in enumerate(scores) if score > 0),
                                   key=lambda i: (-scores[i], rows[i]["id"]))[:limit])
        if mode in {"dense", "hybrid"}:
            if not settings.embedding_model:
                raise ValueError("Dense retrieval requires EMBEDDING_MODEL")
            if any(not row["has_embedding"] or
                   row["embedding_model"] != settings.embedding_model for row in rows):
                raise ValueError("Embedding configuration changed or index incomplete; re-ingest documents")
            query_vector = _vector_literal((self.embedder or embedding_client()).embed_query(query))
            dense = self._dense_ranking(owner, query_vector, document_ids, limit)
            rankings.append([by_id[item["id"]] for item in dense if item["id"] in by_id])
        fused = Counter()
        for ranking in rankings:
            for rank, index in enumerate(ranking, 1):
                fused[index] += 1 / (60 + rank)
        evidence = []
        for index in sorted(fused, key=lambda i: (-fused[i], rows[i]["id"]))[:limit]:
            row = rows[index]
            source = f"document://{row['document_id']}#page={row['page']}&chunk={row['id']}"
            evidence.append(Evidence(
                evidence_id=evidence_id(source, row["text"]), source_type="document",
                title=row["name"], source=source, text=row["text"], page=row["page"],
                document_id=row["document_id"], chunk_id=row["id"], score=fused[index],
            ).model_dump())
        return rerank(query, evidence, top_k=top_k)

    def get_chunk(self, owner: str, chunk_id: str):
        with self.connect() as db, db.cursor() as cur:
            cur.execute("""
                SELECT c.id,c.document_id,c.page,c.text,d.name FROM document_chunks c
                JOIN documents d ON d.id=c.document_id WHERE c.id=%s AND d.owner=%s
            """, (chunk_id, owner))
            return cur.fetchone()

    def preferences(self, owner: str, instructions: str | None = None) -> str:
        if instructions is not None and len(instructions) > 2000:
            raise ValueError("Preferences exceed limit")
        with self.connect() as db, db.cursor() as cur:
            if instructions is not None:
                if instructions:
                    cur.execute("""
                        INSERT INTO user_preferences(owner,instructions) VALUES (%s,%s)
                        ON CONFLICT(owner) DO UPDATE SET instructions=excluded.instructions,
                        updated_at=now()
                    """, (owner, instructions))
                else:
                    cur.execute("DELETE FROM user_preferences WHERE owner=%s", (owner,))
            cur.execute("SELECT instructions FROM user_preferences WHERE owner=%s", (owner,))
            row = cur.fetchone()
            return row["instructions"] if row else ""
