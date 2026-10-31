"""Owner-scoped document store, BM25 and optional real embedding retrieval.

SQLite + exact cosine search is a small-corpus development adapter, not ANN or
a production vector database. Dense mode never silently uses fake embeddings.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from research_agent.config import settings
from research_agent.evidence import Evidence, evidence_id
from research_agent.rerank import rerank


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        words.extend(run)
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words


def chunks(text: str, size: int, overlap: int):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        part = text[start:end].strip()
        if part:
            yield start, part
        if end == len(text):
            break
        start = end - overlap


def parse_document(filename: str, data: bytes) -> list[tuple[int, str]]:
    if not data or len(data) > settings.max_upload_bytes:
        raise ValueError("Document is empty or exceeds upload limit")
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".txt"}:
        pages = [(1, data.decode("utf-8-sig"))]
    elif suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted or len(reader.pages) > 500:
                raise ValueError("Encrypted or oversized PDF is not supported")
            pages = []
            total = 0
            for index, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                total += len(text)
                if total > settings.max_document_chars:
                    raise ValueError("Extracted document exceeds character limit")
                pages.append((index, text))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Unable to parse PDF") from exc
    else:
        raise ValueError("Supported formats: PDF, Markdown and UTF-8 TXT")
    if sum(len(text) for _, text in pages) > settings.max_document_chars:
        raise ValueError("Extracted document exceeds character limit")
    if not any(text.strip() for _, text in pages):
        raise ValueError("No extractable text; scanned PDFs require OCR (not enabled)")
    return pages


def embedding_client():
    if not settings.embedding_model:
        raise ValueError("Dense retrieval requires EMBEDDING_MODEL and compatible embedding API")
    from langchain_openai import OpenAIEmbeddings
    options = {
        "model": settings.embedding_model,
        "request_timeout": 45,
        "max_retries": 1,
        # Some OpenAI-compatible providers accept only arrays of strings and
        # reject the token-id arrays used by LangChain's length-safe path.
        "check_embedding_ctx_length": False,
    }
    if settings.embedding_base_url:
        options["base_url"] = settings.embedding_base_url
    if settings.embedding_api_key:
        options["api_key"] = settings.embedding_api_key
    return OpenAIEmbeddings(**options)


class KnowledgeStore:
    def __init__(self, path: str | None = None, embedder=None):
        self.path = path or settings.knowledge_db_path
        self.embedder = embedder
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
                    content_hash TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(owner, content_hash, name)
                );
                CREATE INDEX IF NOT EXISTS documents_owner ON documents(owner);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL, offset INTEGER NOT NULL, text TEXT NOT NULL,
                    vector_json TEXT, embedding_model TEXT
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS preferences (
                    owner TEXT PRIMARY KEY, instructions TEXT NOT NULL
                );
            """)

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add(self, owner: str, filename: str, data: bytes) -> dict:
        name = filename.replace("\\", "/").split("/")[-1][:200]
        pages = parse_document(name, data)
        digest = hashlib.sha256(data).hexdigest()
        doc_id = "doc_" + hashlib.sha256(f"{owner}\0{name}\0{digest}".encode()).hexdigest()[:24]
        parts = [(page, offset, text) for page, body in pages
                 for offset, text in chunks(body, settings.chunk_size, settings.chunk_overlap)]
        if len(parts) > 1000:
            raise ValueError("Too many document chunks")
        with self.db() as db:
            exists = db.execute("SELECT id FROM documents WHERE id=? AND owner=?", (doc_id, owner)).fetchone()
        if exists:
            return {"document_id": doc_id, "name": name, "chunks": len(parts), "duplicate": True}
        vectors = None
        if settings.retrieval_mode in {"dense", "hybrid"}:
            vectors = (self.embedder or embedding_client()).embed_documents([part[2] for part in parts])
            if len(vectors) != len(parts) or not vectors:
                raise ValueError("Embedding response size mismatch")
            dimension = len(vectors[0])
            if not dimension or any(len(v) != dimension or not all(math.isfinite(x) for x in v) for v in vectors):
                raise ValueError("Invalid embedding vectors")
        with self.db() as db:
            count = db.execute("SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.owner=?", (owner,)).fetchone()[0]
            if count + len(parts) > 10000:
                raise ValueError("Owner chunk limit reached; delete documents before adding more")
            db.execute("INSERT OR IGNORE INTO documents(id,owner,name,content_hash) VALUES (?,?,?,?)", (doc_id, owner, name, digest))
            for index, (page, offset, text) in enumerate(parts):
                chunk_id = f"{doc_id}_{index}"
                db.execute("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?)", (
                    chunk_id, doc_id, page, offset, text,
                    json.dumps(vectors[index]) if vectors is not None else None,
                    settings.embedding_model if vectors is not None else None,
                ))
        return {"document_id": doc_id, "name": name, "chunks": len(parts), "duplicate": False}

    def list_documents(self, owner: str) -> list[dict]:
        with self.db() as db:
            return [dict(r) for r in db.execute(
                "SELECT d.id AS document_id,d.name,d.created_at,COUNT(c.id) AS chunks "
                "FROM documents d LEFT JOIN chunks c ON c.document_id=d.id "
                "WHERE d.owner=? GROUP BY d.id ORDER BY d.created_at DESC", (owner,))]

    def delete(self, owner: str, doc_id: str) -> bool:
        with self.db() as db:
            return db.execute("DELETE FROM documents WHERE id=? AND owner=?", (doc_id, owner)).rowcount == 1

    def _rows(self, owner, document_ids=None):
        sql = "SELECT c.*,d.name FROM chunks c JOIN documents d ON c.document_id=d.id WHERE d.owner=?"
        params = [owner]
        if document_ids:
            sql += " AND d.id IN (" + ",".join("?" for _ in document_ids) + ")"
            params.extend(document_ids)
        with self.db() as db:
            return [dict(row) for row in db.execute(sql + " ORDER BY c.id", params)]

    def search(self, owner: str, query: str, top_k: int = 5, document_ids=None, mode=None) -> list[dict]:
        if not query.strip() or len(query) > 2000 or not 1 <= top_k <= 20:
            raise ValueError("Invalid query or top_k")
        mode = mode or settings.retrieval_mode
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        rows = self._rows(owner, document_ids)
        if not rows:
            return []
        rankings = []
        if mode in {"bm25", "hybrid"}:
            bags = [Counter(tokens(row["text"])) for row in rows]
            lengths = [sum(bag.values()) for bag in bags]
            avg = sum(lengths) / len(rows) or 1
            scores = [0.0] * len(rows)
            for term in set(tokens(query)):
                df = sum(term in bag for bag in bags)
                idf = math.log(1 + (len(rows) - df + 0.5) / (df + 0.5))
                for i, bag in enumerate(bags):
                    tf = bag[term]
                    scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[i] / avg))
            rankings.append(sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], rows[i]["id"]))[:max(20, top_k * 4)])
        if mode in {"dense", "hybrid"}:
            if any(not r["vector_json"] or r["embedding_model"] != settings.embedding_model for r in rows):
                raise ValueError("Embedding configuration changed or index incomplete; re-ingest documents")
            vector = (self.embedder or embedding_client()).embed_query(query)
            norm = math.sqrt(sum(x * x for x in vector))
            if not norm or not all(math.isfinite(x) for x in vector):
                raise ValueError("Invalid query embedding")
            scores = []
            for row in rows:
                candidate = json.loads(row["vector_json"])
                if len(candidate) != len(vector):
                    raise ValueError("Embedding dimension mismatch")
                denominator = norm * math.sqrt(sum(x * x for x in candidate))
                scores.append(sum(a * b for a, b in zip(vector, candidate)) / denominator if denominator else 0)
            rankings.append(sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], rows[i]["id"]))[:max(20, top_k * 4)])
        fused = Counter()
        for ranking in rankings:
            for rank, index in enumerate(ranking, 1):
                fused[index] += 1 / (60 + rank)
        results = []
        candidate_limit = max(settings.reranker_candidates, top_k)
        for index in sorted(fused, key=lambda i: (-fused[i], rows[i]["id"]))[:candidate_limit]:
            row = rows[index]
            source = f"document://{row['document_id']}#page={row['page']}&chunk={row['id']}"
            results.append(Evidence(
                evidence_id=evidence_id(source, row["text"]), source_type="document",
                title=row["name"], source=source, text=row["text"], page=row["page"],
                document_id=row["document_id"], chunk_id=row["id"], score=fused[index],
            ).model_dump())
        return rerank(query, results, top_k=top_k)

    def get_chunk(self, owner, chunk_id):
        with self.db() as db:
            row = db.execute("SELECT c.*,d.name FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=? AND d.owner=?", (chunk_id, owner)).fetchone()
        return {key: row[key] for key in ("id", "document_id", "page", "text", "name")} if row else None

    def preferences(self, owner: str, instructions: str | None = None) -> str:
        with self.db() as db:
            if instructions is not None:
                if len(instructions) > 2000:
                    raise ValueError("Preferences exceed limit")
                if not instructions:
                    db.execute("DELETE FROM preferences WHERE owner=?", (owner,))
                else:
                    db.execute("INSERT INTO preferences VALUES (?,?) ON CONFLICT(owner) DO UPDATE SET instructions=excluded.instructions", (owner, instructions))
            row = db.execute("SELECT instructions FROM preferences WHERE owner=?", (owner,)).fetchone()
            return row[0] if row else ""


def create_knowledge_store(path: str | None = None, embedder=None):
    """Create the storage adapter selected by the runtime profile."""
    if settings.runtime_profile == "production":
        from research_agent.postgres_knowledge import PostgresKnowledgeStore
        return PostgresKnowledgeStore(settings.database_url, embedder=embedder)
    return KnowledgeStore(path, embedder=embedder)
