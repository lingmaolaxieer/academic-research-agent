"""Owner-scoped knowledge and explicit preference-memory endpoints."""
from fastapi import HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from research_agent.config import settings
from research_agent.knowledge import create_knowledge_store


class Preferences(BaseModel):
    instructions: str = Field(default="", max_length=2000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] = Field(default_factory=list, max_length=100)


def install_document_api(app):
    def store(request):
        return create_knowledge_store(request.app.state.knowledge_path)

    @app.post("/documents", status_code=201)
    async def upload(request: Request, file: UploadFile):
        try:
            data = await file.read(settings.max_upload_bytes + 1)
            return await run_in_threadpool(store(request).add, request.state.owner,
                                           file.filename or "", data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        finally:
            await file.close()

    @app.get("/documents")
    def documents(request: Request):
        return store(request).list_documents(request.state.owner)

    @app.delete("/documents/{document_id}")
    def delete(document_id: str, request: Request):
        if not store(request).delete(request.state.owner, document_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": True}

    @app.post("/knowledge/search")
    def search(body: SearchRequest, request: Request):
        try:
            return {"evidence": store(request).search(request.state.owner, body.query,
                             body.top_k, body.document_ids)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/memory/preferences")
    def preferences(request: Request):
        return {"instructions": store(request).preferences(request.state.owner)}

    @app.put("/memory/preferences")
    def remember(body: Preferences, request: Request):
        return {"instructions": store(request).preferences(request.state.owner, body.instructions)}
