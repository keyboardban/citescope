from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class NormalizedScrapeResult:
    provider: str
    provider_mode: str
    requested_url: str
    final_url: str
    normalized_url: str
    status_code: int | None
    success: bool
    error: str
    fetched_at: str
    html: str
    markdown: str
    text: str
    title: str
    meta_description: str
    raw_response_path: str
    text_char_count: int
    word_count: int
    heading_count: int
    table_count: int
    link_count: int
    image_count: int
    content_quality_flag: str

    def to_dict(self) -> dict:
        return asdict(self)
