📦 Master MCP Project — Full Source Code (Markdown Edition)
A complete, aligned FastMCP + Groq + LangChain + CrewAI project.
import asyncio
import os
import requests
from fastmcp import FastMCP
from groq import Groq

API_KEY = os.getenv("MCP_API_KEY", "super-secret-key")

mcp = FastMCP("multi-tool-server")


def _check_auth(api_key: str):
    if api_key != API_KEY:
        raise ValueError("Unauthorized: invalid API key")


@mcp.tool()
def web_search(query: str, api_key: str) -> dict:
    _check_auth(api_key)

    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    return {
        "query": query,
        "abstract": data.get("Abstract", ""),
        "answer": data.get("Answer", ""),
    }


@mcp.tool()
def summarize_text(text: str, max_length: int, api_key: str) -> str:
    _check_auth(api_key)

    if len(text) <= max_length:
        return text
    short = text[: max_length].rsplit(" ", 1)[0]
    return short + "..."


@mcp.tool()
def math_add(a: float, b: float, api_key: str) -> float:
    _check_auth(api_key)
    return a + b


@mcp.tool()
async def web_search_stream(query: str, api_key: str):
    _check_auth(api_key)

    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}

    yield f"Starting search for: {query}"
    await asyncio.sleep(0.2)
    yield "Contacting DuckDuckGo..."

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    abstract = data.get("Abstract", "")
    answer = data.get("Answer", "")

    if abstract:
        yield f"Abstract: {abstract}"
    if answer:
        yield f"Answer: {answer}"
    if not abstract and not answer:
        yield "No direct abstract or answer found."

    yield "Search complete."


@mcp.tool()
def summarize_with_groq(text: str, api_key: str, max_words: int = 120) -> str:
    _check_auth(api_key)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set")

    client = Groq(api_key=groq_key)

    prompt = (
        f"Summarize the following text in under {max_words} words. "
        f"Keep it clear, factual, and concise.\n\nTEXT:\n{text}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message["content"]

✅ 2. server/server.py
from .tools import mcp

if __name__ == "__main__":
    mcp.run_tcp(host="0.0.0.0", port=8765)




✅ 3. client/cli_client.py
import asyncio
import os
from fastmcp import MCPClient

MCP_HOST = "127.0.0.1"
MCP_PORT = 8765
API_KEY = os.getenv("MCP_API_KEY", "super-secret-key")


async def call_basic_tools():
    client = MCPClient.tcp(MCP_HOST, MCP_PORT)
    await client.start()

    print("=== Calling web_search ===")
    res = await client.call_tool(
        "web_search",
        {"query": "India sports news", "api_key": API_KEY},
    )
    print("Result:", res)

    print("\n=== Calling math_add ===")
    res = await client.call_tool(
        "math_add",
        {"a": 2, "b": 5, "api_key": API_KEY},
    )
    print("2 + 5 =", res)

    print("\n=== Calling summarize_text ===")
    text = (
        "Sports play a vital role in our life. They keep us fit, build discipline "
        "and team spirit, and improve our overall wellbeing."
    )
    res = await client.call_tool(
        "summarize_text",
        {"text": text, "max_length": 80, "api_key": API_KEY},
    )
    print("Summary:", res)

    await client.close()


async def call_streaming():
    client = MCPClient.tcp(MCP_HOST, MCP_PORT)
    await client.start()

    print("\n=== Streaming from web_search_stream ===")
    async for chunk in client.stream_tool(
        "web_search_stream",
        {"query": "Virat Kohli latest news", "api_key": API_KEY},
    ):
        print("Chunk:", chunk)

    await client.close()


async def call_groq_summary():
    client = MCPClient.tcp(MCP_HOST, MCP_PORT)
    await client.start()

    long_text = (
        "Sports play a vital role in our life. They keep us fit, build discipline "
        "and team spirit, and improve our overall wellbeing. Many people enjoy "
        "cricket, football, badminton, and athletics."
    )

    print("\n=== Calling summarize_with_groq ===")
    res = await client.call_tool(
        "summarize_with_groq",
        {"text": long_text, "api_key": API_KEY, "max_words": 40},
    )
    print("Groq Summary:", res)

    await client.close()


if __name__ == "__main__":
    asyncio.run(call_basic_tools())
    asyncio.run(call_streaming())
    asyncio.run(call_groq_summary())



✅ 4. llm/langchain_client.py
import asyncio
import os

from fastmcp import MCPClient
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

MCP_HOST = "127.0.0.1"
MCP_PORT = 8765
API_KEY = os.getenv("MCP_API_KEY", "super-secret-key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class MCPWebSearchTool(BaseTool):
    name = "mcp_web_search"
    description = "Search the web via the MCP server."

    def _run(self, query: str):
        return asyncio.run(self._async_run(query))

    async def _async_run(self, query: str):
        client = MCPClient.tcp(MCP_HOST, MCP_PORT)
        await client.start()
        res = await client.call_tool(
            "web_search",
            {"query": query, "api_key": API_KEY},
        )
        await client.close()
        return res


class MCPSummarizeGroqTool(BaseTool):
    name = "mcp_summarize_groq"
    description = "Summarize text using Groq Llama 3.1 70B via MCP."

    def _run(self, text: str):
        return asyncio.run(self._async_run(text))

    async def _async_run(self, text: str):
        client = MCPClient.tcp(MCP_HOST, MCP_PORT)
        await client.start()
        res = await client.call_tool(
            "summarize_with_groq",
            {"text": text, "api_key": API_KEY, "max_words": 80},
        )
        await client.close()
        return res


def build_agent():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-70b-versatile",
        temperature=0.2,
    )

    tools = [MCPWebSearchTool(), MCPSummarizeGroqTool()]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


if __name__ == "__main__":
    agent = build_agent()
    question = (
        "Use the mcp_web_search tool to fetch recent sports news from India "
        "and summarize it in 3 bullet points using mcp_summarize_groq."
    )
    answer = agent.run(question)
    print("\nFinal answer:\n", answer)


✅ 5. llm/crewai_client.py
import asyncio
import os

from fastmcp import MCPClient
from crewai import Tool, Agent, Crew
from crewai import LLM

MCP_HOST = "127.0.0.1"
MCP_PORT = 8765
API_KEY = os.getenv("MCP_API_KEY", "super-secret-key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


async def mcp_search(query: str):
    client = MCPClient.tcp(MCP_HOST, MCP_PORT)
    await client.start()
    res = await client.call_tool(
        "web_search",
        {"query": query, "api_key": API_KEY},
    )
    await client.close()
    return res


async def mcp_summarize(text: str):
    client = MCPClient.tcp(MCP_HOST, MCP_PORT)
    await client.start()
    res = await client.call_tool(
        "summarize_with_groq",
        {"text": text, "api_key": API_KEY, "max_words": 80},
    )
    await client.close()
    return res


web_search_tool = Tool(
    name="mcp_web_search",
    description="Search the web using the MCP server.",
    func=lambda q: asyncio.run(mcp_search(q)),
)

summarize_tool = Tool(
    name="mcp_summarize_groq",
    description="Summarize text using Groq Llama 3.1 70B via MCP.",
    func=lambda t: asyncio.run(mcp_summarize(t)),
)


def build_crew():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    llm = LLM(
        model="groq/llama-3.1-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )

    researcher = Agent(
        role="Sports Researcher",
        goal="Find and summarize sports information.",
        backstory="Expert in Indian sports and news.",
        tools=[web_search_tool, summarize_tool],
        llm=llm,
        verbose=True,
    )

    return Crew(agents=[researcher], verbose=True)


if __name__ == "__main__":
    crew = build_crew()
    task = (
        "Get the latest major sports headlines from India and summarize them "
        "as 4–5 bullet points."
    )
    result = crew.kickoff(task)
    print("\nCrew result:\n", result)



