from __future__ import annotations

from pathlib import Path

from langgraph_lightrag_demo.lightrag_client import (
    _build_ingest_documents,
    _clean_pdf_page_lines,
    _remove_repeated_headers_and_footers,
)
from langgraph_lightrag_demo.semantic_chunker import chunk_text_by_semantics


def test_clean_pdf_page_lines_removes_page_number_and_keeps_caption() -> None:
    cleaned = _clean_pdf_page_lines(
        [
            "公司内部资料",
            "第 3 页",
            "这是正文第一段。",
            "图 1: 系统架构",
            "***",
        ]
    )

    assert "第 3 页" not in cleaned
    assert "图 1: 系统架构" in cleaned
    assert "这是正文第一段。" in cleaned


def test_remove_repeated_headers_and_footers_strips_repeated_edge_lines() -> None:
    pages = [
        ["机密资料", "第一页正文", "页脚声明"],
        ["机密资料", "第二页正文", "页脚声明"],
        ["机密资料", "第三页正文", "页脚声明"],
    ]

    cleaned_pages = _remove_repeated_headers_and_footers(pages)

    assert cleaned_pages == [
        ["第一页正文"],
        ["第二页正文"],
        ["第三页正文"],
    ]


def test_semantic_chunker_splits_by_section_and_page_marker() -> None:
    text = """
    # 第一章 总览

    [PDF Page 1]
    这是第一段。这是第二段。

    ## 1.1 细节

    [PDF Page 2]
    这是一个很长的说明。它会被继续拆分。这里继续补充更多内容。
    """.strip()

    chunks = chunk_text_by_semantics(text, max_tokens=18, min_chunk_tokens=4)

    assert len(chunks) >= 2
    assert chunks[0].section_path == ("第一章 总览",)
    assert chunks[0].page_marker == "[PDF Page 1]"
    assert any(chunk.page_marker == "[PDF Page 2]" for chunk in chunks)
    assert any("细节" in " > ".join(chunk.section_path) for chunk in chunks)


def test_build_ingest_documents_adds_source_and_section_metadata() -> None:
    content = """
    # 项目介绍

    [PDF Page 3]
    这是一个用于测试的段落。这一段会作为知识库内容被导入。
    """.strip()

    documents = _build_ingest_documents(Path("demo.md"), content)

    assert documents
    assert "[Source File]\ndemo.md" in documents[0]
    assert "[Section Path]\n项目介绍" in documents[0]
    assert "[PDF Page 3]" in documents[0]
