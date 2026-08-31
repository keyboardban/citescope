import json

import pandas as pd

from src.econometrics_eda_v2.gemini_position_features import (
    ContentBlock,
    SemanticDetection,
    SemanticDetectionResponse,
    aggregate_page_detections,
    build_page_feature_scorecard,
    build_chunk_prompt,
    classify_page,
    extract_semantic_blocks,
    select_smoke_urls,
)


def test_html_blocks_exclude_navigation_and_keep_deterministic_positions():
    html = """
    <html><body>
      <nav><h2>What is cited?</h2><ul><li>Home</li><li>Contact</li></ul></nav>
      <main>
        <h1>Buyer guide</h1>
        <h2>Which option is better?</h2>
        <p>Option A costs 20% less than option B.</p>
        <ol><li>Check the price.</li><li>Compare the terms.</li></ol>
      </main>
      <footer><p>Rank and citation information</p></footer>
    </body></html>
    """
    blocks, meta = extract_semantic_blocks(html)

    assert meta["block_extraction_status"] == "measured"
    assert [block.block_id for block in blocks] == [f"B{i:04d}" for i in range(1, len(blocks) + 1)]
    assert all("Home" not in block.text for block in blocks)
    assert all("Rank and citation" not in block.text for block in blocks)
    assert any(block.tag == "h2" and "better" in block.text for block in blocks)
    assert all(0 <= block.position_ratio <= 1 for block in blocks)


def test_prompt_contains_only_page_context_and_block_fields():
    block = ContentBlock("B0001", "p", 0, 0.0, "This citation term is page content.", ())
    prompt = build_chunk_prompt([block], page_title="A cited-source title")

    payload = json.loads(prompt.split("BLOCKS_JSON:\n", 1)[1])
    assert set(payload[0]) == {"block_id", "tag", "text"}
    assert "rule_candidates" not in prompt
    assert "observed_rank" not in prompt
    assert "source_position" not in prompt


def test_fake_classifier_is_cached_and_aggregated(tmp_path):
    blocks = [
        ContentBlock("B0001", "h2", 10, 0.1, "Which plan is better?", ("question_heading",)),
        ContentBlock("B0002", "p", 20, 0.2, "Plan A costs 25% less than Plan B.", ()),
    ]
    calls = []

    def fake_request(client, chunk, *, page_title, model):
        calls.append((page_title, model))
        parsed = SemanticDetectionResponse(detections=[
            SemanticDetection(
                block_id="B0001",
                feature="question_heading",
                confidence="high",
                rationale="The heading asks a user-facing question.",
            ),
            SemanticDetection(
                block_id="B0002",
                feature="comparison",
                confidence="high",
                rationale="The paragraph explicitly compares price.",
            ),
        ])
        return parsed, {"fake": True}, {"input_tokens": 100, "output_tokens": 20, "thinking_tokens": 0, "total_tokens": 120}

    detections, meta = classify_page(
        client=object(),
        normalized_url="https://example.com/guide",
        page_title="Guide",
        blocks=blocks,
        cache_dir=tmp_path,
        model="fake-model",
        execute_live=True,
        force=False,
        request_fn=fake_request,
    )
    result, evidence = aggregate_page_detections(blocks, detections)

    assert len(calls) == 1
    assert meta["gemini_status"] == "success"
    assert result["has_question_heading_gemini_v1"] == 1
    assert result["has_comparison_gemini_v1"] == 1
    assert {row["block_id"] for row in evidence} == {"B0001", "B0002"}

    _, cached_meta = classify_page(
        client=None,
        normalized_url="https://example.com/guide",
        page_title="Guide",
        blocks=blocks,
        cache_dir=tmp_path,
        model="fake-model",
        execute_live=False,
        force=False,
        request_fn=fake_request,
    )
    assert cached_meta["gemini_status"] == "success"
    assert cached_meta["chunks_cached"] == 1
    assert len(calls) == 1


def test_smoke_selection_does_not_require_outcome_columns():
    rows = []
    for index in range(12):
        row = {
            "normalized_url": f"https://example.com/{index}",
            "position_features_available": 1,
        }
        columns = [
            "has_direct_answer", "has_definition_block", "has_comparison",
            "has_steps", "has_numeric_evidence", "has_question_heading",
        ]
        row.update({column: 0 for column in columns})
        row[columns[index % len(columns)]] = 1
        rows.append(row)

    selected = select_smoke_urls(pd.DataFrame(rows), max_urls=6)
    assert len(selected) == 6
    assert "cited" not in selected.columns

    full = select_smoke_urls(pd.DataFrame(rows), max_urls=None)
    assert len(full) == 12
    assert full["normalized_url"].is_monotonic_increasing


def test_manual_scorecard_distinguishes_absent_from_unmeasured():
    page = {
        "gemini_status": "success",
        "has_direct_answer": 1,
        "has_direct_answer_gemini_v1": 1,
        "direct_answer_count_gemini_v1": 2,
        "first_direct_answer_position_ratio_gemini_v1": 0.25,
    }
    evidence = pd.DataFrame(
        [{"feature": "direct_answer", "confidence": "high"}]
    )
    measured = build_page_feature_scorecard(page, evidence)
    direct = measured[measured["feature"].eq("Direct answer")].iloc[0]
    definition = measured[measured["feature"].eq("Definition")].iloc[0]
    assert direct["gemini_score"] == 1
    assert direct["confidence"] == "high"
    assert direct["first_position"] == "25.0%"
    assert definition["gemini_score"] == 0

    unmeasured = build_page_feature_scorecard(
        {"gemini_status": "unmeasured_no_blocks", "has_direct_answer": 0},
        pd.DataFrame(),
    )
    assert set(unmeasured["gemini_score"]) == {"NA"}
