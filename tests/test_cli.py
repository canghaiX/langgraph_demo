from __future__ import annotations

from pathlib import Path

from langgraph_lightrag_demo.cli import SUPPORTED_SUFFIXES, _collect_files, build_parser


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


def test_build_parser_supports_metrics_flags() -> None:
    parser = build_parser()

    ask_args = parser.parse_args(["ask", "hello", "--metrics", "--user-id", "alice"])
    chat_args = parser.parse_args(["chat", "--metrics", "--user-id", "alice"])
    set_pref_args = parser.parse_args(
        ["set-pref", "--user-id", "alice", "--language", "zh", "--response-style", "detailed"]
    )
    show_summary_args = parser.parse_args(["show-summary", "--user-id", "alice"])

    assert ask_args.metrics is True
    assert chat_args.metrics is True
    assert ask_args.user_id == "alice"
    assert chat_args.user_id == "alice"
    assert set_pref_args.language == "zh"
    assert set_pref_args.response_style == "detailed"
    assert show_summary_args.user_id == "alice"
