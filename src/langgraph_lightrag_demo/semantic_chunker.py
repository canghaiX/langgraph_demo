from __future__ import annotations

from dataclasses import dataclass
import re


_CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_PDF_PAGE_MARKER_PATTERN = re.compile(r"^\[PDF Page \d+\]$")
_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+){0,5})\s+(.+?)\s*$")
_CHINESE_HEADING_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百千万\d]+[章节部分篇卷])\s*(.+)?$"
)
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
_SENTENCE_PATTERN = re.compile(r"[^。！？!?]+[。！？!?]?")
_CLAUSE_PATTERN = re.compile(r"[^，；;：:]+[，；;：:]?")


@dataclass(frozen=True)
class SemanticChunk:
    """表示一个带元数据的语义切分结果。"""

    text: str
    section_path: tuple[str, ...] = ()
    page_marker: str | None = None


@dataclass(frozen=True)
class _SemanticBlock:
    """表示结构化预切分后的中间块。"""

    text: str
    section_path: tuple[str, ...]
    page_marker: str | None


def estimate_token_count(text: str) -> int:
    """对文本 token 数做轻量近似估计。

    这里不依赖具体 tokenizer，只做保守估算：
    - 中文按字计数
    - 英文/数字按词计数
    - 额外符号按非空白字符计数
    """
    if not text.strip():
        return 0

    cjk_count = len(_CJK_CHAR_PATTERN.findall(text))
    word_count = len(_WORD_PATTERN.findall(text))
    symbol_count = sum(
        1
        for char in text
        if not char.isspace()
        and not _CJK_CHAR_PATTERN.fullmatch(char)
        and not (char.isascii() and (char.isalnum() or char == "_"))
    )
    return cjk_count + word_count + symbol_count


def chunk_text_by_semantics(
    text: str,
    max_tokens: int,
    min_chunk_tokens: int | None = None,
    overlap_sentences: int = 0,
) -> list[SemanticChunk]:
    """按语义单元递归切分文本。

    切分顺序采用“结构优先、长度兜底”的思路：
    1. 先识别页码标记和标题边界
    2. 超长块再按段落切
    3. 段落仍然过长，再按句子切
    4. 句子仍然过长，再按分句切
    5. 最后退化到硬切分
    """
    normalized = text.strip()
    if not normalized:
        return []

    lower_bound = min_chunk_tokens or max(80, max_tokens // 5)
    blocks = _split_into_structural_blocks(normalized)

    chunks: list[SemanticChunk] = []
    for block in blocks:
        chunks.extend(
            _chunk_block_recursively(
                block,
                max_tokens=max_tokens,
                remaining_levels=("paragraph", "sentence", "clause"),
            )
        )

    merged = _merge_small_chunks(chunks, max_tokens=max_tokens, min_tokens=lower_bound)
    return _apply_chunk_overlap(
        merged,
        max_tokens=max_tokens,
        overlap_sentences=overlap_sentences,
    )


def _split_into_structural_blocks(text: str) -> list[_SemanticBlock]:
    """先按标题和 PDF 页码做一轮结构化切分。"""
    blocks: list[_SemanticBlock] = []
    current_lines: list[str] = []
    section_path: list[str] = []
    current_page_marker: str | None = None

    def flush_current_lines() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            current_lines.clear()
            return
        blocks.append(
            _SemanticBlock(
                text=body,
                section_path=tuple(section_path),
                page_marker=current_page_marker,
            )
        )
        current_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if _PDF_PAGE_MARKER_PATTERN.fullmatch(line):
            flush_current_lines()
            current_page_marker = line
            continue

        heading_info = _parse_heading_line(line)
        if heading_info is not None:
            flush_current_lines()
            level, heading_text = heading_info
            section_path = _update_section_path(section_path, level, heading_text)
            continue

        current_lines.append(line)

    flush_current_lines()

    if blocks:
        return blocks

    return [
        _SemanticBlock(
            text=normalized.strip(),
            section_path=(),
            page_marker=None,
        )
        for normalized in [text]
        if normalized.strip()
    ]


def _parse_heading_line(line: str) -> tuple[int, str] | None:
    """识别 Markdown / 编号 / 中文章节标题。"""
    markdown_match = _MARKDOWN_HEADING_PATTERN.match(line)
    if markdown_match:
        return len(markdown_match.group(1)), markdown_match.group(2).strip()

    numbered_match = _NUMBERED_HEADING_PATTERN.match(line)
    if numbered_match:
        return numbered_match.group(1).count(".") + 1, line.strip()

    chinese_match = _CHINESE_HEADING_PATTERN.match(line)
    if chinese_match:
        return 1, line.strip()

    return None


def _update_section_path(
    current_path: list[str],
    level: int,
    heading_text: str,
) -> list[str]:
    """按标题层级更新 section path。"""
    if level <= 1:
        return [heading_text]

    next_path = list(current_path[: level - 1])
    while len(next_path) < level - 1:
        next_path.append("")
    next_path.append(heading_text)
    return next_path


def _chunk_block_recursively(
    block: _SemanticBlock,
    max_tokens: int,
    remaining_levels: tuple[str, ...],
) -> list[SemanticChunk]:
    """对单个结构块继续做递归语义切分。"""
    chunk = SemanticChunk(
        text=block.text.strip(),
        section_path=block.section_path,
        page_marker=block.page_marker,
    )
    if estimate_token_count(_render_chunk_text(chunk)) <= max_tokens:
        return [chunk]

    if not remaining_levels:
        return _hard_split_chunk(chunk, max_tokens=max_tokens)

    splitter = remaining_levels[0]
    units = _split_text_by_level(block.text, splitter)
    if len(units) <= 1:
        return _chunk_block_recursively(
            block,
            max_tokens=max_tokens,
            remaining_levels=remaining_levels[1:],
        )

    chunks: list[SemanticChunk] = []
    for unit in units:
        child_block = _SemanticBlock(
            text=unit,
            section_path=block.section_path,
            page_marker=block.page_marker,
        )
        chunks.extend(
            _chunk_block_recursively(
                child_block,
                max_tokens=max_tokens,
                remaining_levels=remaining_levels[1:],
            )
        )
    return chunks


def _split_text_by_level(text: str, level: str) -> list[str]:
    """按指定语义层级切分正文。"""
    if level == "paragraph":
        return _split_with_pattern(text, _PARAGRAPH_SPLIT_PATTERN)
    if level == "sentence":
        return _extract_units_with_pattern(text, _SENTENCE_PATTERN)
    if level == "clause":
        return _extract_units_with_pattern(text, _CLAUSE_PATTERN)
    return [text.strip()]


def _split_with_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    """按 pattern 切分并清理空结果。"""
    return [part.strip() for part in pattern.split(text) if part.strip()]


def _extract_units_with_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    """提取按标点保留边界的句子/分句。"""
    units = [match.group(0).strip() for match in pattern.finditer(text) if match.group(0).strip()]
    return units or [text.strip()]


def _hard_split_chunk(chunk: SemanticChunk, max_tokens: int) -> list[SemanticChunk]:
    """对没有明显语义边界的超长文本做兜底切分。"""
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", chunk.text)
    if not tokens:
        return []

    parts: list[SemanticChunk] = []
    current_tokens: list[str] = []
    for token in tokens:
        current_tokens.append(token)
        candidate = "".join(current_tokens)
        if estimate_token_count(_render_chunk_text(chunk, body=candidate)) > max_tokens:
            overflow = current_tokens.pop()
            if current_tokens:
                parts.append(
                    SemanticChunk(
                        text="".join(current_tokens),
                        section_path=chunk.section_path,
                        page_marker=chunk.page_marker,
                    )
                )
            current_tokens = [overflow]

    if current_tokens:
        parts.append(
            SemanticChunk(
                text="".join(current_tokens),
                section_path=chunk.section_path,
                page_marker=chunk.page_marker,
            )
        )
    return parts


def _merge_small_chunks(
    chunks: list[SemanticChunk],
    max_tokens: int,
    min_tokens: int,
) -> list[SemanticChunk]:
    """把同一 section/page 上下文中过短的相邻块尽量回并。"""
    if not chunks:
        return []

    merged: list[SemanticChunk] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue

        previous = merged[-1]
        same_context = _can_section_merge(previous, chunk)
        if not same_context:
            merged.append(chunk)
            continue

        previous_tokens = estimate_token_count(_render_chunk_text(previous))
        candidate = SemanticChunk(
            text=f"{previous.text}\n\n{chunk.text}".strip(),
            section_path=previous.section_path,
            page_marker=previous.page_marker,
        )
        candidate_tokens = estimate_token_count(_render_chunk_text(candidate))
        current_tokens = estimate_token_count(_render_chunk_text(chunk))
        if (
            candidate_tokens <= max_tokens
            and (previous_tokens < min_tokens or current_tokens < min_tokens)
        ):
            merged[-1] = candidate
            continue

        merged.append(chunk)

    return merged


def _can_section_merge(previous: SemanticChunk, current: SemanticChunk) -> bool:
    """只允许同一 section/page 内的短块回并，避免跨主题拼接。"""
    return (
        previous.section_path == current.section_path
        and previous.page_marker == current.page_marker
    )


def _apply_chunk_overlap(
    chunks: list[SemanticChunk],
    max_tokens: int,
    overlap_sentences: int,
) -> list[SemanticChunk]:
    """为同一 section 内的相邻 chunk 注入少量句子级 overlap。"""
    if overlap_sentences <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[SemanticChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        previous = overlapped[-1]
        if not _can_section_merge(previous, chunk):
            overlapped.append(chunk)
            continue

        overlap_text = _build_overlap_text(previous.text, overlap_sentences)
        if not overlap_text or chunk.text.startswith(overlap_text):
            overlapped.append(chunk)
            continue

        candidate = SemanticChunk(
            text=f"{overlap_text}\n\n{chunk.text}".strip(),
            section_path=chunk.section_path,
            page_marker=chunk.page_marker,
        )
        if estimate_token_count(_render_chunk_text(candidate)) <= max_tokens:
            overlapped.append(candidate)
            continue

        fitted_overlap = _fit_overlap_within_budget(
            previous.text,
            chunk,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
        )
        overlapped.append(fitted_overlap or chunk)

    return overlapped


def _build_overlap_text(text: str, overlap_sentences: int) -> str:
    """从上一个 chunk 末尾提取若干句，作为下一个 chunk 的重叠上下文。"""
    sentences = _extract_units_with_pattern(text, _SENTENCE_PATTERN)
    if not sentences:
        return ""
    tail = sentences[-overlap_sentences:]
    return "".join(tail).strip()


def _fit_overlap_within_budget(
    previous_text: str,
    chunk: SemanticChunk,
    max_tokens: int,
    overlap_sentences: int,
) -> SemanticChunk | None:
    """在 token 预算内尽量保留 overlap。"""
    sentences = _extract_units_with_pattern(previous_text, _SENTENCE_PATTERN)
    if not sentences:
        return None

    for size in range(min(overlap_sentences, len(sentences)), 0, -1):
        overlap_text = "".join(sentences[-size:]).strip()
        if not overlap_text or chunk.text.startswith(overlap_text):
            continue
        candidate = SemanticChunk(
            text=f"{overlap_text}\n\n{chunk.text}".strip(),
            section_path=chunk.section_path,
            page_marker=chunk.page_marker,
        )
        if estimate_token_count(_render_chunk_text(candidate)) <= max_tokens:
            return candidate
    return None


def _render_chunk_text(chunk: SemanticChunk, body: str | None = None) -> str:
    """把 chunk 渲染成最终会被送入向量库的近似文本。"""
    parts: list[str] = []
    if chunk.section_path:
        parts.append(" > ".join(part for part in chunk.section_path if part))
    if chunk.page_marker:
        parts.append(chunk.page_marker)
    parts.append((body if body is not None else chunk.text).strip())
    return "\n".join(part for part in parts if part).strip()
