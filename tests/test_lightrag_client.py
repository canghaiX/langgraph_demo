from __future__ import annotations

from langgraph_lightrag_demo.lightrag_client import (
    _clean_pdf_page_lines,
    _remove_repeated_headers_and_footers,
)


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
