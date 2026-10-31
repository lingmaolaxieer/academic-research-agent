"""Read-only MCP server. Owner is bound by the trusted host, never by tool args."""
import os

from mcp.server.fastmcp import FastMCP

from research_agent.knowledge import create_knowledge_store

mcp = FastMCP("academic-knowledge")


def owner() -> str:
    value = os.environ.get("KNOWLEDGE_OWNER")
    if not value:
        raise ValueError("Host must bind KNOWLEDGE_OWNER before starting the MCP process")
    return value


@mcp.tool()
def search_documents(query: str, top_k: int = 5, document_ids: list[str] | None = None) -> dict:
    """Retrieve owner-accessible excerpts. Return evidence IDs, page numbers and text."""
    if document_ids and len(document_ids) > 100:
        raise ValueError("Too many document IDs")
    return {"evidence": create_knowledge_store().search(owner(), query, top_k, document_ids)}


@mcp.tool()
def get_document_chunk(chunk_id: str) -> dict:
    """Read an exact authorized source excerpt; never accepts a filesystem path."""
    result = create_knowledge_store().get_chunk(owner(), chunk_id)
    if result is None:
        raise ValueError("Chunk not found")
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
