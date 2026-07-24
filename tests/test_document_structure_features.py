from __future__ import annotations

import json

import pandas as pd

from src.econometrics_eda_v2.document_structure_features import (
    DOCUMENT_STRUCTURE_VERSION,
    _snapshot_key,
    extract_document_structure,
    run_document_structure_layer,
)


HTML = """
<!doctype html>
<html>
  <head>
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": ["FAQPage", "Article"],
        "mainEntity": [
          {"@type": "Question", "name": "What is included?"},
          {"@type": "Question", "name": "How much?"}
        ]
      }
    </script>
  </head>
  <body>
    <header><a href="https://ads.example.net">Advertisement</a></header>
    <main>
      <h1>Project facts</h1>
      <h3>Comparison</h3>
      <p>This is a concise introductory paragraph.</p>
      <p>This paragraph contains more detail about the available units and prices.</p>
      <ul><li>Swimming pool</li><li>Gym</li></ul>
      <ol><li>Review details</li></ol>
      <table>
        <tr><th>Type</th><th>Detail</th></tr>
        <tr><td>Price</td><td>THB 10,000,000</td></tr>
        <tr><td>Unit size</td><td>85 sqm, 2 bedrooms</td></tr>
        <tr><td>Comparison</td><td>Plan A vs. Plan B</td></tr>
      </table>
      <a href="/internal">Internal</a>
      <a href="https://official.example.org/report" rel="nofollow">External report</a>
    </main>
  </body>
</html>
"""


def test_html_first_extractor_preserves_requested_structure():
    features, texts = extract_document_structure(HTML, "https://example.com/page")

    assert features["document_structure_version"] == DOCUMENT_STRUCTURE_VERSION
    assert features["html_table_count"] == 1
    assert features["table_row_count"] == 4
    assert features["table_column_max"] == 2
    assert features["price_row_count"] >= 1
    assert features["unit_size_row_count"] >= 1
    assert features["comparison_row_count"] >= 1
    assert features["h1_count"] == 1
    assert features["h3_count"] == 1
    assert features["heading_level_skip_count"] == 1
    assert features["paragraph_count"] == 2
    assert features["unordered_list_count"] == 1
    assert features["ordered_list_count"] == 1
    assert features["list_item_count"] == 3
    assert features["internal_link_count"] == 1
    assert features["outbound_link_count"] == 1
    assert features["external_link_domains"] == "example.org"
    assert features["nofollow_link_count"] == 1
    assert features["has_faqpage_schema"] == 1
    assert features["has_article_schema"] == 1
    assert features["faq_schema_question_count"] == 2
    assert "# Project facts" in texts["generated_markdown"]
    assert "| Type | Detail |" in texts["generated_markdown"]
    assert "Advertisement" in texts["full_body_text"]
    assert "Advertisement" not in texts["main_content_text"]


def test_missing_html_keeps_structure_missing_instead_of_zero():
    features, texts = extract_document_structure("", "https://example.com", "Fallback text")

    assert features["html_available"] == 0
    assert features["structure_features_available"] == 0
    assert pd.isna(features["html_table_count"])
    assert features["text_available"] == 1
    assert texts["main_content_text"] == "Fallback text"
    assert texts["generated_markdown"] == ""


def test_document_structure_runner_writes_url_and_row_outputs(tmp_path):
    package = tmp_path / "package"
    data = package / "data"
    data.mkdir(parents=True)
    url = "https://example.com/page?utm_source=chatgpt.com"
    normalized = "https://example.com/page"
    pd.DataFrame(
        [{
            "normalized_url": normalized,
            "source_url": url,
            "source_root_domain": "example.com",
            "scrape_success": True,
            "content_strength": "strong",
        }]
    ).to_csv(data / "url_content_evidence_compact.csv", index=False)
    pd.DataFrame(
        [{"cited": 1, "prompt_id": "p1", "normalized_url": normalized}]
    ).to_csv(data / "content_lpm_measurable_rows.csv", index=False)
    snapshots = tmp_path / "snapshots" / "crawler_api"
    snapshots.mkdir(parents=True)
    (snapshots / f"{_snapshot_key(url)}.json").write_text(
        json.dumps({"html": HTML, "text": "Fallback"}), encoding="utf-8"
    )

    result = run_document_structure_layer(package, snapshots.parent)

    assert result["status"] == "document_structure_features_ready_for_descriptive_qa"
    assert result["urls_measurable"] == 1
    assert (package / "data/content_lpm_measurable_rows_with_document_structure_features.csv").exists()
    output = package / "tables/12_document_structure_features"
    assert (output / "url_document_structure_features.csv").exists()
    assert (output / "url_full_body_text_and_generated_markdown.csv.gz").exists()
    assert len(list((output / "markdown").glob("*.md"))) == 1
