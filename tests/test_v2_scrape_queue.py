from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.scrape_queue import build_scrape_queue


def test_scrape_queue_deduplicates_strips_tracking_and_counts(tmp_path):
    src = pd.DataFrame(
        [
            {"normalized_url": "https://example.com/a?utm_source=x", "source_url": "https://example.com/a?utm_source=x", "answer_id": "a1", "cited": 1},
            {"normalized_url": "https://example.com/a", "source_url": "https://example.com/a", "answer_id": "a2", "cited": 0},
        ]
    )
    q, summary = build_scrape_queue(src, raw_dir=tmp_path)
    assert len(q) == 1
    assert q.iloc[0]["n_source_rows"] == 2
    assert q.iloc[0]["n_cited_rows"] == 1
    assert q.iloc[0]["n_more_only_rows"] == 1
    assert summary["should_scrape"] == 1
