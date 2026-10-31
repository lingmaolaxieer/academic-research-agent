"""Fixed-endpoint public search with transport timeouts and bounded responses."""
import json
import os
from functools import lru_cache

import httpx
from langchain.tools import tool


class SearchClient:
    def invoke(self, arguments):
        key = os.environ.get("TAVILY_API_KEY", "")
        if not key:
            raise ValueError("Configure TAVILY_API_KEY for web research")
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Invalid search query")
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=False) as client:
            with client.stream("POST", "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"query": query, "max_results": 4, "search_depth": "basic",
                          "include_answer": False, "include_raw_content": False}) as response:
                response.raise_for_status()
                body = bytearray()
                for block in response.iter_bytes():
                    body.extend(block)
                    if len(body) > 2_000_000:
                        raise ValueError("Search response exceeds size limit")
        result = json.loads(body)
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            raise ValueError("Invalid search response")
        return result


@lru_cache(maxsize=1)
def _get_search():
    return SearchClient()


@tool("web_search", description="Search public web sources and return source snippets.")
def web_search(query: str) -> dict:
    return _get_search().invoke({"query": query})
