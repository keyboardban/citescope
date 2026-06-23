"""Heading-aware text chunking for the chunk-relevance visualizer.

Chunk similarity is treated as a *semantic overlap proxy* — it indicates which
passages of a page are most related to the AI answer, not which passages the
model actually read.
"""

from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def extract_headings(markdown: str) -> list[str]:
    """Pull markdown headings (lines beginning with #...) in document order."""
    out: list[str] = []
    for line in (markdown or "").splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            text = m.group(2).strip()
            if text:
                out.append(text)
    return out


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def chunk_text(
    text: str,
    target_words: int = 120,
    overlap_words: int = 25,
    max_chunks: int = 60,
) -> list[dict]:
    """Split text into overlapping, heading-tagged chunks.

    Returns a list of dicts: {index, heading, text, n_words}. Prefers markdown
    headings as natural boundaries; falls back to a sliding word window.
    """
    if not text or not text.strip():
        return []

    # Segment by markdown headings so chunks stay topically coherent.
    segments: list[tuple[str, str]] = []  # (heading, body)
    current_head = ""
    buf: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            if buf:
                segments.append((current_head, "\n".join(buf).strip()))
                buf = []
            current_head = m.group(2).strip()
        else:
            buf.append(line)
    if buf:
        segments.append((current_head, "\n".join(buf).strip()))
    if not segments:
        segments = [("", text.strip())]

    chunks: list[dict] = []
    step = max(1, target_words - overlap_words)
    for heading, body in segments:
        words = re.findall(r"\S+", body)
        if not words:
            continue
        if len(words) <= target_words:
            windows = [words]
        else:
            windows = [words[i : i + target_words] for i in range(0, len(words), step)]
        for w in windows:
            piece = " ".join(w).strip()
            if _word_count(piece) < 8:  # skip trivially short fragments
                continue
            chunks.append(
                {
                    "index": len(chunks),
                    "heading": heading,
                    "text": piece,
                    "n_words": _word_count(piece),
                }
            )
            if len(chunks) >= max_chunks:
                return chunks
    return chunks
