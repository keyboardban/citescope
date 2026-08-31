import pandas as pd

from src.econometrics_eda_v2.verified_table_diagnostics import (
    domain_diagnostics,
    identifying_support,
    overall_summary,
)


def _sample() -> pd.DataFrame:
    rows = []
    for prompt, domain, values in (
        ("p1", "a.com", [0, 1, 1, 0, 1, 0]),
        ("p2", "a.com", [0, 0, 1, 1, 1, 0]),
        ("p3", "b.com", [1, 1, 1, 1]),
        ("p4", "c.com", [0, 0, 0, 0]),
    ):
        for index, table in enumerate(values):
            rows.append(
                {
                    "prompt_id": prompt,
                    "source_root_domain": domain,
                    "normalized_url": f"https://{domain}/{prompt}/{index}",
                    "has_verified_html_table": table,
                    "cited": int(index % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_overall_summary_preserves_raw_state_counts_and_difference():
    data = _sample()
    summary = overall_summary(data).set_index("table_status")

    assert summary.loc[0, "n_rows"] == 10
    assert summary.loc[1, "n_rows"] == 10
    assert summary.loc[0, "n_rows"] + summary.loc[1, "n_rows"] == len(data)
    assert summary["raw_cited_rate_difference_pp"].nunique() == 1


def test_domain_support_requires_rows_and_both_table_states():
    diagnostics = domain_diagnostics(_sample()).set_index("domain")

    assert diagnostics.loc["a.com", "adequate_difference_support"]
    assert not diagnostics.loc["b.com", "adequate_difference_support"]
    assert diagnostics.loc["b.com", "prevalence_flag"] == "100_percent"
    assert diagnostics.loc["c.com", "prevalence_flag"] == "0_percent"


def test_identifying_support_partitions_every_row_once():
    data = _sample()
    summary, rows = identifying_support(data)
    metrics = summary.set_index("metric")["value"]

    assert metrics["prompts_with_both_states"] == 2
    assert metrics["domains_with_both_states"] == 1
    assert len(rows) == len(data)
    assert rows["identifying_support_category"].value_counts().sum() == len(data)
