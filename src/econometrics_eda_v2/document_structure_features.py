"""HTML-first, cross-domain document-structure feature extraction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

from src.url_utils import root_domain


DOCUMENT_STRUCTURE_VERSION = "document_structure_v1"
SNAPSHOT_MODES = ("crawler_api", "browser_api", "unlocker_api")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*|[\u0E00-\u0E7F]+", re.UNICODE)
PRICE_ROW_RE = re.compile(
    r"(?:฿|\bthb\b|\bbaht\b|\busd\b|\$|€|£|\bprice\b|ราคา|บาท|เริ่มต้น)"
    r"|(?:\d[\d,.]*\s*(?:million|ล้าน))",
    re.I | re.U,
)
UNIT_ROW_RE = re.compile(
    r"(?:\d[\d,.]*\s*(?:sqm|sq\.?\s*m\.?|m²|square\s+met(?:er|re)s?|ตร\.?\s*ม\.?|ตารางเมตร))"
    r"|(?:\d+\s*(?:bedrooms?|beds?|ห้องนอน))|\b(?:studio|penthouse|duplex)\b",
    re.I | re.U,
)
COMPARISON_ROW_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|difference|pros?|cons?)\b|เปรียบเทียบ|เทียบ|แตกต่าง|ข้อดี|ข้อเสีย",
    re.I | re.U,
)
NOISE_RE = re.compile(
    r"(?:^|[-_\s])(nav|menu|footer|header|sidebar|cookie|consent|popup|modal|advert|ads|"
    r"social|share|related|breadcrumb)(?:$|[-_\s])",
    re.I,
)

STRUCTURE_NUMERIC_FEATURES = (
    "full_body_chars", "full_body_word_count", "main_content_chars", "main_content_word_count",
    "main_content_ratio", "html_table_count", "table_row_count", "table_column_max",
    "table_cell_count", "table_header_cell_count", "tables_with_header_count", "price_row_count",
    "unit_size_row_count", "comparison_row_count", "h1_count", "h2_count", "h3_count",
    "h4_count", "h5_count", "h6_count", "heading_count", "heading_max_depth",
    "heading_level_skip_count", "heading_text_chars", "paragraph_count", "median_paragraph_words",
    "p90_paragraph_words", "short_paragraph_share", "long_paragraph_share", "unordered_list_count",
    "ordered_list_count", "list_item_count", "max_list_depth", "link_count_total",
    "internal_link_count", "outbound_link_count", "external_link_domain_count", "nofollow_link_count",
    "jsonld_block_count", "jsonld_parsed_block_count", "jsonld_parse_error_count", "schema_type_count",
    "faq_schema_question_count",
)
STRUCTURE_BINARY_FEATURES = (
    "html_available", "text_available", "main_content_available", "structure_features_available",
    "document_features_measurable", "markdown_generated", "has_html_table", "has_heading_hierarchy",
    "has_paragraph_structure", "has_list_structure", "has_outbound_links", "has_jsonld",
    "has_faqpage_schema", "has_article_schema", "has_product_schema", "has_breadcrumb_schema",
    "has_organization_schema", "has_localbusiness_schema",
)
STRUCTURE_TEXT_FEATURES = (
    "document_structure_version", "main_content_extraction_method", "external_link_domains",
    "schema_types", "full_body_text_path", "main_content_text_path", "generated_markdown_path",
)
MODEL_FEATURES = (*STRUCTURE_BINARY_FEATURES, *STRUCTURE_NUMERIC_FEATURES)


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def _clean_lines(text: str) -> str:
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in (text or "").splitlines()]
    output: list[str] = []
    blank = False
    for line in lines:
        if line:
            output.append(line)
            blank = False
        elif output and not blank:
            output.append("")
            blank = True
    return "\n".join(output).strip()


def _inline_markdown(node: Tag | NavigableString, base_url: str, traversal_depth: int = 0) -> str:
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, Tag):
        return ""
    if traversal_depth >= 80:
        return node.get_text(" ", strip=True)
    name = node.name.casefold()
    if name == "br":
        return "\n"
    content = "".join(
        _inline_markdown(child, base_url, traversal_depth + 1) for child in node.children
    ).strip()
    if name == "a":
        href = str(node.get("href") or "").strip()
        if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return f"[{content or href}]({urljoin(base_url, href)})"
    if name in {"strong", "b"} and content:
        return f"**{content}**"
    if name in {"em", "i"} and content:
        return f"*{content}*"
    if name == "code" and content:
        return f"`{content}`"
    return content


def _table_markdown(table: Tag, base_url: str) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        if cells:
            rows.append([_inline_markdown(cell, base_url).replace("|", "\\|").strip() for cell in cells])
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    return "\n".join(
        ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        + ["| " + " | ".join(row) + " |" for row in padded[1:]]
    )


def _block_markdown(
    node: Tag | NavigableString,
    base_url: str,
    list_depth: int = 0,
    traversal_depth: int = 0,
) -> str:
    if isinstance(node, NavigableString):
        text = re.sub(r"\s+", " ", str(node)).strip()
        return text + "\n\n" if text else ""
    if not isinstance(node, Tag):
        return ""
    if traversal_depth >= 80:
        text = node.get_text(" ", strip=True)
        return text + "\n\n" if text else ""
    name = node.name.casefold()
    if re.fullmatch(r"h[1-6]", name):
        return f"{'#' * int(name[1])} {_inline_markdown(node, base_url).strip()}\n\n"
    if name == "p":
        text = _inline_markdown(node, base_url).strip()
        return text + "\n\n" if text else ""
    if name in {"ul", "ol"}:
        lines = []
        for index, item in enumerate(node.find_all("li", recursive=False), start=1):
            prefix = f"{index}." if name == "ol" else "-"
            clone_text = " ".join(
                _inline_markdown(child, base_url).strip()
                for child in item.children
                if not (isinstance(child, Tag) and child.name in {"ul", "ol"})
            ).strip()
            if clone_text:
                lines.append("  " * list_depth + f"{prefix} {clone_text}")
            for nested in item.find_all(["ul", "ol"], recursive=False):
                nested_text = _block_markdown(
                    nested, base_url, list_depth + 1, traversal_depth + 1
                ).strip()
                if nested_text:
                    lines.append(nested_text)
        return "\n".join(lines) + "\n\n" if lines else ""
    if name == "table":
        rendered = _table_markdown(node, base_url)
        return rendered + "\n\n" if rendered else ""
    if name in {"pre", "blockquote"}:
        text = node.get_text("\n", strip=True)
        return f"```\n{text}\n```\n\n" if name == "pre" else f"> {text}\n\n"
    return "".join(
        _block_markdown(child, base_url, list_depth, traversal_depth + 1)
        for child in node.children
    )


def html_to_markdown(node: Tag, base_url: str) -> str:
    return _clean_lines(_block_markdown(node, base_url))


def _remove_noise(node: Tag) -> None:
    for tag in list(node.find_all(["script", "style", "noscript", "template", "svg", "canvas", "iframe"])):
        tag.decompose()
    for tag in list(node.find_all(["nav", "footer", "header", "aside", "dialog"])):
        tag.decompose()
    for tag in list(node.find_all(True)):
        if tag.attrs is None:
            continue
        role = " ".join(tag.get("role") or []).casefold() if isinstance(tag.get("role"), list) else str(tag.get("role") or "").casefold()
        marker = " ".join(
            [str(tag.get("id") or ""), " ".join(tag.get("class") or [])]
        )
        if role in {"navigation", "banner", "contentinfo"} or NOISE_RE.search(marker):
            tag.decompose()


def _select_main_node(soup: BeautifulSoup) -> tuple[Tag, str]:
    candidates: list[tuple[Tag, str]] = []
    selectors = (("article", "article"), ("main", "main"), ('[role="main"]', "role_main"))
    for selector, method in selectors:
        candidates.extend((node, method) for node in soup.select(selector))
    content_marker = re.compile(r"(?:^|[-_])(main|content|article|post|entry)(?:$|[-_])", re.I)
    for node in soup.find_all(["div", "section"]):
        marker = " ".join([str(node.get("id") or ""), " ".join(node.get("class") or [])])
        if content_marker.search(marker):
            candidates.append((node, "content_container"))
    if not candidates:
        return (soup.body or soup), "body_fallback"

    def score(item: tuple[Tag, str]) -> float:
        node, method = item
        text = node.get_text(" ", strip=True)
        link_text = " ".join(link.get_text(" ", strip=True) for link in node.find_all("a"))
        density = len(link_text) / max(len(text), 1)
        semantic_bonus = 1.15 if method in {"article", "main", "role_main"} else 1.0
        return len(text) * max(0.2, 1 - min(density, 0.8)) * semantic_bonus

    return max(candidates, key=score)


def _schema_features(soup: BeautifulSoup) -> dict[str, Any]:
    blocks = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)})
    parsed = 0
    errors = 0
    types: set[str] = set()
    faq_questions = 0

    def walk(value: Any, inside_faq: bool = False) -> None:
        nonlocal faq_questions
        if isinstance(value, list):
            for item in value:
                walk(item, inside_faq)
            return
        if not isinstance(value, dict):
            return
        raw_types = value.get("@type", [])
        raw_types = raw_types if isinstance(raw_types, list) else [raw_types]
        node_types = {str(item).strip() for item in raw_types if str(item).strip()}
        types.update(node_types)
        faq = inside_faq or any(item.casefold() == "faqpage" for item in node_types)
        if faq and any(item.casefold() == "question" for item in node_types):
            faq_questions += 1
        for child in value.values():
            walk(child, faq)

    for block in blocks:
        raw = block.string or block.get_text() or ""
        try:
            value = json.loads(raw.strip())
        except (json.JSONDecodeError, TypeError, ValueError):
            errors += 1
            continue
        parsed += 1
        walk(value)
    folded = {item.casefold() for item in types}
    return {
        "jsonld_block_count": len(blocks),
        "jsonld_parsed_block_count": parsed,
        "jsonld_parse_error_count": errors,
        "schema_type_count": len(types),
        "schema_types": ";".join(sorted(types, key=str.casefold)),
        "has_jsonld": int(bool(blocks)),
        "has_faqpage_schema": int("faqpage" in folded),
        "faq_schema_question_count": faq_questions,
        "has_article_schema": int(bool(folded & {"article", "newsarticle", "blogposting"})),
        "has_product_schema": int(bool(folded & {"product", "offer"})),
        "has_breadcrumb_schema": int("breadcrumblist" in folded),
        "has_organization_schema": int(bool(folded & {"organization", "corporation"})),
        "has_localbusiness_schema": int("localbusiness" in folded),
    }


def _table_features(node: Tag) -> dict[str, Any]:
    tables = node.find_all("table")
    row_count = cell_count = header_count = tables_with_header = 0
    column_max = price_rows = unit_rows = comparison_rows = 0
    for table in tables:
        rows = table.find_all("tr")
        if table.find("th"):
            tables_with_header += 1
        table_text = table.get_text(" ", strip=True)
        comparison_table = bool(COMPARISON_ROW_RE.search(table_text[:1000]))
        for tr in rows:
            cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
            if not cells:
                continue
            row_count += 1
            cell_count += len(cells)
            header_count += sum(cell.name == "th" for cell in cells)
            logical_columns = 0
            for cell in cells:
                try:
                    logical_columns += max(1, int(cell.get("colspan") or 1))
                except (TypeError, ValueError):
                    logical_columns += 1
            column_max = max(column_max, logical_columns)
            row_text = " | ".join(cell.get_text(" ", strip=True) for cell in cells)
            price_rows += int(bool(PRICE_ROW_RE.search(row_text)))
            unit_rows += int(bool(UNIT_ROW_RE.search(row_text)))
            comparison_rows += int(comparison_table or bool(COMPARISON_ROW_RE.search(row_text)))
    return {
        "html_table_count": len(tables), "table_row_count": row_count,
        "table_column_max": column_max, "table_cell_count": cell_count,
        "table_header_cell_count": header_count, "tables_with_header_count": tables_with_header,
        "price_row_count": price_rows, "unit_size_row_count": unit_rows,
        "comparison_row_count": comparison_rows, "has_html_table": int(bool(tables)),
    }


def _heading_features(node: Tag) -> dict[str, Any]:
    headings = node.find_all(re.compile(r"^h[1-6]$", re.I))
    levels = [int(heading.name[1]) for heading in headings]
    counts = {f"h{level}_count": levels.count(level) for level in range(1, 7)}
    skips = sum(current - previous > 1 for previous, current in zip(levels, levels[1:]))
    return {
        **counts, "heading_count": len(headings), "heading_max_depth": max(levels, default=0),
        "heading_level_skip_count": skips,
        "heading_text_chars": sum(len(heading.get_text(" ", strip=True)) for heading in headings),
        "has_heading_hierarchy": int(bool(headings)),
    }


def _paragraph_list_features(node: Tag) -> dict[str, Any]:
    paragraphs = [p.get_text(" ", strip=True) for p in node.find_all("p")]
    lengths = [len(_words(text)) for text in paragraphs if _words(text)]
    unordered = node.find_all("ul")
    ordered = node.find_all("ol")
    items = node.find_all("li")
    depths = [sum(parent.name in {"ul", "ol"} for parent in item.parents) for item in items]
    return {
        "paragraph_count": len(lengths),
        "median_paragraph_words": float(np.median(lengths)) if lengths else 0.0,
        "p90_paragraph_words": float(np.quantile(lengths, 0.9)) if lengths else 0.0,
        "short_paragraph_share": float(np.mean(np.asarray(lengths) <= 20)) if lengths else 0.0,
        "long_paragraph_share": float(np.mean(np.asarray(lengths) >= 100)) if lengths else 0.0,
        "unordered_list_count": len(unordered), "ordered_list_count": len(ordered),
        "list_item_count": len(items), "max_list_depth": max(depths, default=0),
        "has_paragraph_structure": int(bool(lengths)), "has_list_structure": int(bool(items)),
    }


def _link_features(node: Tag, source_url: str) -> dict[str, Any]:
    source_root = root_domain(source_url)
    internal = outbound = nofollow = 0
    external_domains: set[str] = set()
    links = []
    for anchor in node.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        resolved = urljoin(source_url, href)
        if urlparse(resolved).scheme not in {"http", "https"}:
            continue
        links.append(resolved)
        target_root = root_domain(resolved)
        if target_root and target_root != source_root:
            outbound += 1
            external_domains.add(target_root)
        else:
            internal += 1
        rel = anchor.get("rel") or []
        nofollow += int("nofollow" in {str(item).casefold() for item in rel})
    return {
        "link_count_total": len(links), "internal_link_count": internal,
        "outbound_link_count": outbound, "external_link_domain_count": len(external_domains),
        "external_link_domains": ";".join(sorted(external_domains)),
        "nofollow_link_count": nofollow, "has_outbound_links": int(outbound > 0),
    }


def extract_document_structure(html: str, source_url: str, fallback_text: str = "") -> tuple[dict[str, Any], dict[str, str]]:
    """Extract structure features plus full-text audit artifacts from one page."""
    html = str(html or "")
    fallback_text = _clean_lines(str(fallback_text or ""))
    if not html.strip():
        features: dict[str, Any] = {name: np.nan for name in STRUCTURE_NUMERIC_FEATURES}
        features.update({name: 0 for name in STRUCTURE_BINARY_FEATURES})
        features.update({name: "" for name in STRUCTURE_TEXT_FEATURES})
        features.update({
            "document_structure_version": DOCUMENT_STRUCTURE_VERSION,
            "text_available": int(bool(fallback_text)),
            "main_content_available": int(bool(fallback_text)),
            "main_content_extraction_method": "text_fallback" if fallback_text else "unavailable",
        })
        return features, {"full_body_text": fallback_text, "main_content_text": fallback_text, "generated_markdown": ""}

    soup = BeautifulSoup(html, "html.parser")
    schema = _schema_features(soup)
    for tag in list(soup.find_all(["script", "style", "noscript", "template", "svg", "canvas", "iframe"])):
        tag.decompose()
    body = soup.body or soup
    full_body_text = _clean_lines(body.get_text("\n", strip=True))
    main_node, method = _select_main_node(soup)
    _remove_noise(main_node)
    main_text = _clean_lines(main_node.get_text("\n", strip=True)) or full_body_text
    markdown = html_to_markdown(main_node, source_url)
    full_words = len(_words(full_body_text))
    main_words = len(_words(main_text))
    features = {
        "document_structure_version": DOCUMENT_STRUCTURE_VERSION,
        "html_available": 1, "text_available": int(bool(full_body_text or fallback_text)),
        "main_content_available": int(bool(main_text)), "structure_features_available": 1,
        "markdown_generated": int(bool(markdown)), "main_content_extraction_method": method,
        "full_body_chars": len(full_body_text), "full_body_word_count": full_words,
        "main_content_chars": len(main_text), "main_content_word_count": main_words,
        "main_content_ratio": len(main_text) / max(len(full_body_text), 1),
        **_table_features(main_node), **_heading_features(main_node),
        **_paragraph_list_features(main_node), **_link_features(main_node, source_url), **schema,
        "full_body_text_path": "", "main_content_text_path": "", "generated_markdown_path": "",
        "document_features_measurable": 0,
    }
    return features, {
        "full_body_text": full_body_text or fallback_text,
        "main_content_text": main_text or fallback_text,
        "generated_markdown": markdown,
    }


def _snapshot_key(source_url: str) -> str:
    return hashlib.sha256(str(source_url).encode("utf-8")).hexdigest()[:20]


def _snapshot_path(source_url: str, snapshot_root: Path) -> Path | None:
    filename = f"{_snapshot_key(source_url)}.json"
    for mode in SNAPSHOT_MODES:
        candidate = snapshot_root / mode / filename
        if candidate.exists():
            return candidate
    return None


def _write_feature_dictionary(path: Path) -> None:
    descriptions = {
        "price_row_count": "HTML table rows containing currency or explicit price signals.",
        "unit_size_row_count": "HTML table rows containing area, bedroom, or unit-size signals.",
        "comparison_row_count": "Rows in tables with explicit comparison signals.",
        "heading_level_skip_count": "Adjacent heading transitions that skip one or more levels.",
        "main_content_ratio": "Main-content characters divided by full-body characters.",
        "external_link_domains": "Semicolon-separated unique outbound root domains; diagnostic text field.",
        "schema_types": "Semicolon-separated JSON-LD @type values; diagnostic text field.",
    }
    rows = []
    for feature in STRUCTURE_BINARY_FEATURES:
        rows.append({"feature": feature, "row_level_feature": feature, "type": "binary", "source": "HTML/DOM or availability", "model_role": "candidate_or_availability_control", "description": descriptions.get(feature, feature.replace("_", " "))})
    for feature in STRUCTURE_NUMERIC_FEATURES:
        role = "sensitivity" if feature in {"price_row_count", "unit_size_row_count", "comparison_row_count", "faq_schema_question_count"} else "candidate_after_eda"
        rows.append({"feature": feature, "row_level_feature": "dom_heading_count" if feature == "heading_count" else feature, "type": "numeric", "source": "HTML/DOM", "model_role": role, "description": descriptions.get(feature, feature.replace("_", " "))})
    for feature in STRUCTURE_TEXT_FEATURES:
        rows.append({"feature": feature, "row_level_feature": feature, "type": "text_or_provenance", "source": "HTML/DOM", "model_role": "diagnostic_only", "description": descriptions.get(feature, feature.replace("_", " "))})
    pd.DataFrame(rows).to_csv(path, index=False)


def run_document_structure_layer(package_dir: Path, snapshot_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run structure extraction over all URL snapshots and merge measurable rows."""
    package_dir = Path(package_dir).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    output_dir = Path(output_dir or package_dir / "tables/12_document_structure_features").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir = output_dir / "markdown"
    text_dir = output_dir / "text"
    markdown_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)

    evidence = pd.read_csv(package_dir / "data/url_content_evidence_compact.csv", low_memory=False)
    measurable = pd.read_csv(package_dir / "data/content_lpm_measurable_rows.csv", low_memory=False)
    features_path = output_dir / "url_document_structure_features.csv"
    text_archive_path = output_dir / "url_full_body_text_and_generated_markdown.csv.gz"
    reuse = features_path.exists() and text_archive_path.exists()
    if reuse:
        features = pd.read_csv(features_path, low_memory=False)
        reuse = (
            len(features) == len(evidence)
            and features["normalized_url"].astype(str).equals(evidence["normalized_url"].astype(str))
            and features["document_structure_version"].eq(DOCUMENT_STRUCTURE_VERSION).all()
        )
    if reuse:
        text_rows = pd.read_csv(text_archive_path, low_memory=False).to_dict("records")
    else:
        feature_rows: list[dict[str, Any]] = []
        text_rows: list[dict[str, str]] = []
        for row in evidence.itertuples(index=False):
            source_url = str(getattr(row, "source_url", "") or getattr(row, "normalized_url"))
            normalized_url = str(getattr(row, "normalized_url"))
            snapshot_path = _snapshot_path(source_url, snapshot_root)
            snapshot = json.loads(snapshot_path.read_text("utf-8")) if snapshot_path else {}
            features, texts = extract_document_structure(
                str(snapshot.get("html") or ""), source_url, str(snapshot.get("text") or "")
            )
            key = _snapshot_key(source_url)
            full_path = text_dir / f"{key}_full.txt"
            main_path = text_dir / f"{key}_main.txt"
            markdown_path = markdown_dir / f"{key}.md"
            if texts["full_body_text"]:
                full_path.write_text(texts["full_body_text"], encoding="utf-8")
                features["full_body_text_path"] = str(full_path)
            if texts["main_content_text"]:
                main_path.write_text(texts["main_content_text"], encoding="utf-8")
                features["main_content_text_path"] = str(main_path)
            if texts["generated_markdown"]:
                markdown_path.write_text(texts["generated_markdown"], encoding="utf-8")
                features["generated_markdown_path"] = str(markdown_path)
            scrape_success = bool(getattr(row, "scrape_success", False))
            content_strength = str(getattr(row, "content_strength", "") or "").casefold()
            features["document_features_measurable"] = int(
                bool(features["structure_features_available"])
                and scrape_success
                and content_strength in {"strong", "medium"}
            )
            feature_rows.append({
                "normalized_url": normalized_url, "source_url": source_url,
                "source_root_domain": str(getattr(row, "source_root_domain", "") or ""),
                "snapshot_path": str(snapshot_path or ""), **features,
            })
            text_rows.append({
                "normalized_url": normalized_url, "source_url": source_url,
                "full_body_text": texts["full_body_text"], "main_content_text": texts["main_content_text"],
                "generated_markdown": texts["generated_markdown"],
            })

        features = pd.DataFrame(feature_rows)
        features.to_csv(features_path, index=False)
        pd.DataFrame(text_rows).to_csv(text_archive_path, index=False, compression="gzip")
    merge_columns = ["normalized_url", *MODEL_FEATURES, *STRUCTURE_TEXT_FEATURES]
    row_name_map = {"heading_count": "dom_heading_count"}
    merge_frame = features[merge_columns].rename(columns=row_name_map)
    merged_model_features = [row_name_map.get(feature, feature) for feature in MODEL_FEATURES]
    merged = measurable.merge(merge_frame, on="normalized_url", how="left", validate="many_to_one")
    merged.to_csv(
        package_dir / "data/content_lpm_measurable_rows_with_document_structure_features.csv",
        index=False,
    )

    summary_rows = [
        ("unique_urls", len(features), "count"),
        ("urls_with_html", int(features.html_available.sum()), "count"),
        ("urls_with_generated_markdown", int(features.markdown_generated.sum()), "count"),
        ("urls_document_features_measurable", int(features.document_features_measurable.sum()), "count"),
        ("urls_with_html_tables", int(features.has_html_table.sum()), "count"),
        ("urls_with_jsonld", int(features.has_jsonld.sum()), "count"),
        ("urls_with_faqpage_schema", int(features.has_faqpage_schema.sum()), "count"),
        ("urls_with_outbound_links", int(features.has_outbound_links.sum()), "count"),
        ("html_availability_rate", float(features.html_available.mean()), "rate"),
        ("document_measurable_rate", float(features.document_features_measurable.mean()), "rate"),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value", "value_type"])
    summary.to_csv(output_dir / "document_structure_coverage_summary.csv", index=False)

    numeric_summary = features[list(STRUCTURE_NUMERIC_FEATURES)].describe(percentiles=[.5, .9, .95, .99]).T.reset_index(names="feature")
    numeric_summary.to_csv(output_dir / "document_structure_numeric_summary.csv", index=False)
    by_cited = merged.groupby("cited", dropna=False)[merged_model_features].mean(numeric_only=True).T.reset_index(names="feature")
    by_cited.to_csv(output_dir / "document_structure_mean_by_cited_status.csv", index=False)

    review = features.copy()
    review["review_priority"] = (
        pd.to_numeric(review["table_row_count"], errors="coerce").fillna(0)
        + pd.to_numeric(review["outbound_link_count"], errors="coerce").fillna(0) / 10
        + review["has_faqpage_schema"].fillna(0) * 5
        + (1 - review["document_features_measurable"].fillna(0)) * 3
    )
    review = review.sort_values("review_priority", ascending=False).head(100)
    preview_map = {
        str(row["normalized_url"]): (
            "" if pd.isna(row.get("main_content_text")) else str(row.get("main_content_text", ""))[:1000]
        )
        for row in text_rows
    }
    review["main_content_preview"] = review["normalized_url"].map(preview_map)
    review.to_csv(output_dir / "document_structure_manual_review_sample_100.csv", index=False)
    _write_feature_dictionary(output_dir / "document_structure_feature_dictionary.csv")

    report = f"""# Document Structure Feature Report

## Status

`{DOCUMENT_STRUCTURE_VERSION}` completed for {len(features):,} unique URLs. This layer uses HTML/DOM as the structural source and generates Markdown for inspection. It does not call an LLM and does not fit an LPM.

## Coverage

- HTML available: {int(features.html_available.sum()):,} ({features.html_available.mean():.1%})
- Generated Markdown: {int(features.markdown_generated.sum()):,} ({features.markdown_generated.mean():.1%})
- Measurable document structure: {int(features.document_features_measurable.sum()):,} ({features.document_features_measurable.mean():.1%})
- HTML tables: {int(features.has_html_table.sum()):,}
- JSON-LD: {int(features.has_jsonld.sum()):,}
- FAQPage schema: {int(features.has_faqpage_schema.sum()):,}
- Outbound links: {int(features.has_outbound_links.sum()):,}

## Interpretation boundary

The full body and generated Markdown are audit evidence, not direct LPM predictors. Structural values are missing when HTML is unavailable; absence is not imputed as zero. Price, unit-size, comparison-row, and FAQ-schema measures are sensitivity candidates. Existing notebook 10 and 11 outputs remain frozen.

## Next step

Manually review the 100-page sample, then use descriptive plots and missingness checks before specifying any new econometric model.
"""
    (output_dir / "document_structure_feature_report.md").write_text(report, encoding="utf-8")
    result = {
        "status": "document_structure_features_ready_for_descriptive_qa",
        "version": DOCUMENT_STRUCTURE_VERSION,
        "unique_urls": len(features),
        "urls_with_html": int(features.html_available.sum()),
        "urls_measurable": int(features.document_features_measurable.sum()),
        "output_dir": str(output_dir),
        "row_dataset": str(package_dir / "data/content_lpm_measurable_rows_with_document_structure_features.csv"),
    }
    (output_dir / "document_structure_run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
