from __future__ import annotations

import os

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP


# 这是一个本地 MCP server。
# 它通过 stdio 被主项目拉起，然后暴露网页搜索 / 网页抓取两个工具。
mcp = FastMCP("Web Search MCP")
DEFAULT_MAX_RESULTS = int(os.getenv("MCP_WEB_SEARCH_MAX_RESULTS", "5"))
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@mcp.tool()
async def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """搜索公开网页并返回简要结果列表。

    这里为了部署简单，直接用 DuckDuckGo 的 HTML 搜索页做轻量搜索。
    """
    async with httpx.AsyncClient(
        timeout=20.0,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        response = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
        )
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    items = []
    for result in soup.select(".result"):
        title_link = result.select_one(".result__title a") or result.select_one("a")
        snippet_node = result.select_one(".result__snippet")
        if not title_link:
            continue

        title = title_link.get_text(" ", strip=True)
        href = title_link.get("href", "").strip()
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        if not title or not href:
            continue
        items.append((title, href, snippet))
        if len(items) >= max_results:
            break

    if not items:
        return "未找到网页搜索结果。"

    lines = []
    for index, (title, href, snippet) in enumerate(items, start=1):
        lines.append(f"[{index}] {title}\nURL: {href}\n摘要: {snippet or '无摘要'}")
    return "\n\n".join(lines)


@mcp.tool()
async def fetch_webpage(url: str, max_chars: int = 8000) -> str:
    """抓取网页正文并提取可读文本。

    适合在 search_web 返回候选链接后，再让 agent 深读某个页面。
    """
    async with httpx.AsyncClient(
        timeout=20.0,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = soup.get_text("\n", strip=True)
    text = text[:max_chars]
    return f"标题: {title}\nURL: {url}\n\n{text}"


if __name__ == "__main__":
    # stdio 是 MCP 最常见的本地进程通信方式。
    mcp.run(transport="stdio")
