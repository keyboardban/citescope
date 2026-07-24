from __future__ import annotations

import math
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.source_type import classify
from src.url_utils import domain as url_domain
from src.url_utils import normalize_url
from src.url_utils import root_domain
from src.econometrics_eda_v2.page_type_classifier import classify_page_type_details


def _path_depth(url: str) -> int:
    try:
        return len([p for p in urlparse(url).path.split("/") if p])
    except Exception:
        return 0


def _query_flag(url: str) -> int:
    try:
        return int(bool(urlparse(url).query))
    except Exception:
        return 0


def source_domain_raw_looks_like_label(raw: str) -> bool:
    raw = str(raw or "").strip()
    if not raw:
        return False
    if re.search(r"\s|\(|\)|[ก-๙]", raw):
        return True
    return "." not in raw


def _host(url: str) -> str:
    if not str(url or "").strip():
        return ""
    try:
        p = urlparse(url if re.match(r"^[a-z][a-z0-9+.-]*://", url, flags=re.I) else "https://" + str(url))
        return (p.hostname or "").lower()
    except Exception:
        return ""


def build_source_url_features(source_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    work = source_rows.copy()
    work["normalized_url"] = work["normalized_url"].fillna("").map(lambda u: normalize_url(str(u)) if str(u).strip() else "")
    work["_source_root_domain"] = work["source_url"].fillna(work["normalized_url"]).map(lambda u: root_domain(str(u)) if str(u).strip() else "")
    domain_counts = work.groupby("_source_root_domain", dropna=False)["_source_root_domain"].transform("size")
    for idx, row in work.iterrows():
        source_url = str(row.get("source_url") or row.get("normalized_url") or "")
        nurl = str(row.get("normalized_url") or normalize_url(source_url))
        stype, institutional = classify(source_url or nurl)
        pt = classify_page_type_details({"source_url": source_url, "normalized_url": nurl, "source_type_url": stype}, stype)
        raw_domain = str(row.get("source_domain") or "")
        host = _host(source_url or nurl)
        root = root_domain(source_url or nurl) if (source_url or nurl) else ""
        domain = raw_domain or url_domain(nurl)
        domain_plot_label = root if root else "(missing_url)"
        seen = float(domain_counts.loc[idx]) if len(domain_counts) else math.nan
        rows.append(
            {
                "source_row_id": row.get("source_row_id"),
                "normalized_url": nurl,
                "source_url": source_url,
                "source_domain": domain,
                "source_domain_ai_label": raw_domain,
                "source_domain_host": host,
                "source_root_domain": root,
                "domain_plot_label": domain_plot_label,
                "domain_raw_looks_like_label": source_domain_raw_looks_like_label(raw_domain),
                "source_type_url": stype,
                "institutional_official": bool(institutional),
                "official_source": bool(institutional),
                "page_type_url_seed": pt.page_type,
                "page_type_url_seed_source": "url_path_source_type_evidence",
                "page_type_url_seed_confidence": pt.confidence,
                "page_type_url_seed_evidence": pt.evidence,
                "url_length": len(nurl or source_url),
                "url_path_depth": _path_depth(nurl or source_url),
                "https_flag": int(str(nurl or source_url).startswith("https://")),
                "url_has_query_params": _query_flag(nurl or source_url),
                "domain_seen_count": seen,
                "domain_seen_count_loo": max(seen - 1, 0) if pd.notna(seen) else np.nan,
                "log1p_domain_seen_count": float(np.log1p(max(seen, 0))) if pd.notna(seen) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "rows": int(len(df)),
        "valid_url_rows": int((df["normalized_url"].fillna("") != "").sum()) if len(df) else 0,
        "raw_domains_that_look_like_labels": int(df["domain_raw_looks_like_label"].sum()) if len(df) else 0,
        "url_derived_domains": int((df["source_root_domain"].fillna("") != "").sum()) if len(df) else 0,
        "page_type_url_seed_coverage": float(df["page_type_url_seed"].notna().mean()) if len(df) else 0.0,
        "source_type_distribution": df["source_type_url"].value_counts(dropna=False).to_dict() if len(df) else {},
        "page_type_url_seed_distribution": df["page_type_url_seed"].value_counts(dropna=False).to_dict() if len(df) else {},
        "top_raw_domain_labels": df.loc[df["domain_raw_looks_like_label"], "source_domain_ai_label"].value_counts().head(20).to_dict() if len(df) else {},
        "top_root_domains": df["source_root_domain"].value_counts().head(20).to_dict() if len(df) else {},
    }
    return df, summary
