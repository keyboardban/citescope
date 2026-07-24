from __future__ import annotations

from src.url_utils import normalize_url
from src.econometrics_eda_v2.normalize_sources import normalize_text, stable_hash


def test_v2_url_normalization_strips_tracking_params():
    assert normalize_url("http://www.Example.com/a/?utm_source=x&b=2#frag") == "https://example.com/a?b=2"


def test_v2_text_and_hash_are_stable():
    assert normalize_text(" Hello   World? ") == "hello world"
    assert stable_hash("a", "b") == stable_hash("a", "b")
