from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP


# 这个 MCP server 提供当前工作区内的文件访问能力。
# 通过限制 ROOT，可以避免 agent 任意访问机器上的其他路径。
mcp = FastMCP("Filesystem MCP")
ROOT = Path(os.getenv("MCP_FILESYSTEM_ROOT", ".")).resolve()


def _resolve_path(path_str: str) -> Path:
    """把用户传入路径解析到允许访问的根目录之下。"""
    raw_path = Path(path_str)
    target = (ROOT / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    if ROOT not in target.parents and target != ROOT:
        raise ValueError(f"Path is outside allowed root: {target}")
    return target


@mcp.tool()
def list_local_files(subpath: str = ".", max_results: int = 50) -> str:
    """列出允许访问根目录下的文件。"""
    target = _resolve_path(subpath)
    if not target.exists():
        return f"路径不存在: {target}"

    files = []
    for item in target.rglob("*"):
        if item.is_file():
            files.append(str(item.relative_to(ROOT)))
        if len(files) >= max_results:
            break

    if not files:
        return "未找到文件。"
    return "\n".join(files)


@mcp.tool()
def read_local_file(path: str, max_chars: int = 8000) -> str:
    """读取本地文本文件内容。"""
    target = _resolve_path(path)
    if not target.exists() or not target.is_file():
        return f"文件不存在: {target}"

    text = target.read_text(encoding="utf-8", errors="ignore")
    return f"文件: {target.relative_to(ROOT)}\n\n{text[:max_chars]}"


@mcp.tool()
def search_local_files(query: str, subpath: str = ".", max_results: int = 20) -> str:
    """在本地文件内容中搜索关键字。

    优先使用 `rg`，因为它更快。
    如果环境里没有 `rg`，就退化到 Python 逐文件扫描。
    """
    target = _resolve_path(subpath)
    if not target.exists():
        return f"路径不存在: {target}"

    try:
        result = subprocess.run(
            [
                "rg",
                "-n",
                "--no-heading",
                "--color",
                "never",
                "--glob",
                "!*.pyc",
                query,
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        matches = [
            line for line in result.stdout.splitlines() if line.strip()
        ][:max_results]
        if matches:
            return "\n".join(matches)
    except FileNotFoundError:
        pass

    matches = []
    for file_path in target.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if query.lower() in line.lower():
                rel_path = file_path.relative_to(ROOT)
                matches.append(f"{rel_path}:{lineno}:{line.strip()}")
                if len(matches) >= max_results:
                    return "\n".join(matches)

    return "\n".join(matches) if matches else "未找到匹配内容。"


if __name__ == "__main__":
    # 通过 stdio 模式运行，供主 agent 进程以 MCP 客户端方式调用。
    mcp.run(transport="stdio")
