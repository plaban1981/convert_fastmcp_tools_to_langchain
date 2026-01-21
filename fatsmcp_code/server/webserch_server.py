from fastmcp import FastMCP
import requests
import sys
from tavily import TavilyClient
from typing import Dict
tavily_client = TavilyClient(api_key="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")
mcp = FastMCP("websearch-mcp")

@mcp.tool()
def web_search(query: str) -> dict:
    """
    Perform a web search using Tavily API.
    Returns a dictionary of search results.
    """
    # Note: Don't print to stdout - it breaks stdio communication
    # Use stderr for debugging if needed: print(f"Searching for: {query}", file=sys.stderr)
    try:
        response = tavily_client.search(query)
        return response
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        return "No response Generated"

if __name__ == "__main__":
    try:
        # Run with stdio (default transport)
        # Note: Do not print to stdout - only stderr for errors
        mcp.run(transport="stdio")
    except Exception as e:
        # All errors must go to stderr, not stdout
        print(f"Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)