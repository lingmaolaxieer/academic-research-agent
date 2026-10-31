"""Short-lived stdio MCP sessions, with bounded lifetime and process cleanup."""
import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from research_agent.config import settings


async def call_knowledge(owner: str, tool_name: str, arguments: dict) -> dict:
    if tool_name not in {"search_documents", "get_document_chunk"}:
        raise ValueError("Unsupported knowledge tool")
    environment = {key: value for key, value in os.environ.items()
                   if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE"}}
    environment.update({
        "KNOWLEDGE_OWNER": owner,
        "KNOWLEDGE_DB_PATH": str(Path(settings.knowledge_db_path).resolve()),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "PYTHONIOENCODING": "utf-8",
        "RETRIEVAL_MODE": settings.retrieval_mode,
        "EMBEDDING_MODEL": settings.embedding_model,
        "EMBEDDING_DIMENSIONS": str(settings.embedding_dimensions),
        "RUNTIME_PROFILE": settings.runtime_profile,
        "DATABASE_URL": settings.database_url,
        "RERANKER_MODEL": settings.reranker_model,
    })
    for key, value in (("EMBEDDING_API_KEY", settings.embedding_api_key),
                       ("EMBEDDING_BASE_URL", settings.embedding_base_url),
                       ("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")),
                       ("OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL"))):
        if value:
            environment[key] = value
    params = StdioServerParameters(command=sys.executable,
        args=["-B", "-m", "research_agent.knowledge_mcp"], env=environment)
    async with asyncio.timeout(60):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=50)) as session:
                await session.initialize()
                discovered = await session.list_tools()
                if tool_name not in {tool.name for tool in discovered.tools}:
                    raise ValueError("MCP server did not expose requested tool")
                result = await session.call_tool(tool_name, arguments)
                if result.isError:
                    raise ValueError("Knowledge MCP tool failed")
                if result.structuredContent is not None:
                    return result.structuredContent
                text = "".join(part.text for part in result.content if part.type == "text")
                return json.loads(text)


def search_via_mcp(owner: str, query: str, top_k: int, document_ids=None) -> list[dict]:
    # Called in the synchronous graph worker; never call asyncio.run on the API loop.
    result = asyncio.run(call_knowledge(owner, "search_documents", {
        "query": query, "top_k": top_k, "document_ids": document_ids,
    }))
    return result["evidence"]
