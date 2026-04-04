from __future__ import annotations

from pathlib import Path

from langgraph_lightrag_demo.cli import SUPPORTED_SUFFIXES, _collect_files


def test_supported_suffixes_include_pdf() -> None:
    assert ".pdf" in SUPPORTED_SUFFIXES


def test_collect_files_filters_supported_suffixes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()

    keep_md = docs / "a.md"
    keep_txt = docs / "b.txt"
    keep_pdf = docs / "c.pdf"
    skip_png = docs / "d.png"

    keep_md.write_text("hello", encoding="utf-8")
    keep_txt.write_text("world", encoding="utf-8")
    keep_pdf.write_text("fake pdf for collection test", encoding="utf-8")
    skip_png.write_text("nope", encoding="utf-8")

    collected = _collect_files(tmp_path)

    assert collected == sorted([keep_md, keep_txt, keep_pdf])
