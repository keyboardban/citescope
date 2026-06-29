"""Non-branded Brand Visibility Audit — an analysis layer over the ChatGPT
Bright Data parser + Prompt Manifest matching.

Two business questions, answered with careful **observational** wording:

1. **Prompt / intent visibility** — for non-branded prompts (that do not mention
   the client brand directly), which prompts / intents cause ChatGPT to surface,
   mention, or cite the client's website/brand or competitor websites/brands?
2. **Citation content** — among the surfaced client/competitor pages, what content
   features are associated with a page being **cited** rather than only **shown but
   not cited** (more-only)?

Framing rule (never broken): we only describe **observable brand visibility**.
*more-only* = shown-but-not-cited, **NOT** rejected or ignored. Content features
are associations, not proof of why the model cited a page. This layer never claims
to reveal ChatGPT's internal retrieval set.

The brand terms come from the **Prompt Manifest** (per-prompt
`client_brand_terms_to_detect_in_output` / `competitor_terms_to_detect_in_output`).
Hardcoded defaults are used only as a fallback when a record carries no terms.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

from .chunking import extract_headings
from .config import (
    DEFAULT_CLIENT_BRAND_TERMS,
    DEFAULT_COMPETITOR_BRAND_TERMS,
    POSITION_BANDS,
)
from .similarity import SimilarityEngine
from .source_type import classify
from .url_utils import domain

# --------------------------------------------------------------------------- #
# term normalisation + matching
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{6,}\d)")


def _norm(s) -> str:
    """NFKC normalise + casefold. Casefold only affects Latin; Thai is unchanged,
    so brand terms still match as substrings (Thai has no case)."""
    return unicodedata.normalize("NFKC", str(s or "")).casefold()


def compile_terms(terms) -> list[tuple[str, str, object]]:
    """Pre-compile detection patterns. Each entry is (original, kind, value):

    - domain/url-ish terms (containing '.' or '/') and non-ASCII (e.g. Thai) terms
      match as plain **substrings** of the normalised text.
    - ASCII word/phrase terms match on **word boundaries** to avoid false positives
      (e.g. brand "TCC" must not match inside an unrelated longer token).
    """
    out: list[tuple[str, str, object]] = []
    for term in terms or []:
        t = _norm(term)
        if not t:
            continue
        if "." in t or "/" in t or not t.isascii():
            out.append((term, "sub", t))
        else:
            out.append((term, "re", re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")))
    return out


def _match_compiled(text: str, compiled) -> list[str]:
    x = _norm(text)
    if not x or not compiled:
        return []
    out = []
    for original, kind, val in compiled:
        if (val in x) if kind == "sub" else bool(val.search(x)):
            out.append(original)
    return out


def detect_terms(text: str, terms) -> list[str]:
    """Return the brand terms (originals) detected in `text` (case-insensitive for
    English, substring for Thai/domains). Convenience wrapper around compile_terms."""
    return _match_compiled(text, compile_terms(terms))


def _source_blobs(s: dict, page: dict | None) -> str:
    """Text surfaces a brand term may appear on for one source: panel metadata
    (url/domain/title/description) plus scraped page title/headings/body if present."""
    blobs = [s.get("url", ""), s.get("normalized_url", ""), s.get("domain", ""),
             s.get("title", ""), s.get("description", "")]
    if page and page.get("status") == "success":
        blobs.append(page.get("title", ""))
        blobs.append(" ".join(extract_headings(page.get("markdown") or "")))
        blobs.append((page.get("text") or page.get("markdown") or "")[:4000])
    return "\n".join(b for b in blobs if b)


# --------------------------------------------------------------------------- #
# small numeric helpers (bool -> 1/0, skip None/NaN)
# --------------------------------------------------------------------------- #
def _num(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _clean(vals):
    return [x for x in (_num(v) for v in vals) if x is not None]


def _mean(vals):
    xs = _clean(vals)
    return round(sum(xs) / len(xs), 4) if xs else None


def _median(vals):
    xs = sorted(_clean(vals))
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return round(xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2, 4)


def _n(vals):
    return len(_clean(vals))


# --------------------------------------------------------------------------- #
# content feature extraction (heuristic, bilingual EN/TH)
# --------------------------------------------------------------------------- #
_KW = {
    "has_faq": ["faq", "q&a", "q & a", "frequently asked", "คำถามที่พบบ่อย", "ถาม-ตอบ", "ถาม ตอบ"],
    "has_step_by_step": ["step by step", "step-by-step", "how to", "procedure", "steps", "ขั้นตอน", "วิธีการ", "วิธี"],
    "has_contact_info": ["contact us", "contact", "ติดต่อ", "ติดต่อเรา"],
    "has_location_info": ["address", "location", "directions", "how to get", "ที่ตั้ง", "แผนที่", "การเดินทาง"],
    "has_price_or_package": ["price", "pricing", "package", "cost", "fee", "promotion", "ราคา", "แพ็กเกจ", "แพคเกจ", "ค่าบริการ", "โปรโมชั่น"],
    "has_opening_hours": ["opening hours", "open hours", "hours of operation", "business hours", "เวลาทำการ", "เปิดบริการ", "เวลาเปิด"],
    "has_booking_or_appointment": ["booking", "book now", "appointment", "reserve", "reservation", "test drive", "นัดหมาย", "จองคิว", "จอง", "ทดลองขับ"],
    "has_author": ["written by", "author", "เขียนโดย", "ผู้เขียน", "แพทย์ผู้เขียน"],
    "has_reviewer": ["reviewed by", "medically reviewed", "fact checked", "ตรวจทานโดย", "ผู้ตรวจทาน"],
    "has_published_date": ["published", "posted on", "date published", "เผยแพร่", "วันที่เผยแพร่"],
    "has_updated_date": ["updated", "last updated", "อัปเดต", "ปรับปรุงล่าสุด", "แก้ไขล่าสุด"],
}

CONTENT_BOOL_FEATURES = [
    "has_faq", "has_step_by_step", "has_contact_info", "has_location_info",
    "has_price_or_package", "has_opening_hours", "has_booking_or_appointment",
    "has_phone_number", "has_email", "has_author", "has_reviewer",
    "has_published_date", "has_updated_date", "has_schema", "has_table",
    "has_bullets", "has_many_headings", "heading_prompt_match", "title_contains_intent_terms",
]
CONTENT_NUM_FEATURES = ["answer_like_text_in_first_500_chars"]

PAGE_TYPES = [
    "article", "service_page", "department_page", "appointment_page", "contact_page",
    "location_page", "faq_page", "product_page", "price_package_page", "news_page",
    "forum_thread", "review_page", "directory_page", "marketplace_page", "unknown",
]

_GENERIC_INTENT_TOKENS = {"unspecified", "other", "general", "info", "intent", "and", "the", "for"}


def _empty_content() -> dict:
    out = {k: None for k in CONTENT_BOOL_FEATURES + CONTENT_NUM_FEATURES}
    out["page_type"] = "unknown"
    return out


def _page_type(url: str, source_type: str, flags: dict) -> str:
    path = (url or "").lower()
    st = source_type or "unknown"
    if st == "forum":
        return "forum_thread"
    if st == "review":
        return "review_page"
    if st == "news":
        return "news_page"
    if st == "ecommerce":
        return ("marketplace_page"
                if any(k in path for k in ("/search", "/category", "/c/", "listing", "/browse"))
                else "product_page")
    if any(k in path for k in ("/contact", "ติดต่อ")):
        return "contact_page"
    if any(k in path for k in ("/appointment", "/booking", "/book", "นัดหมาย")):
        return "appointment_page"
    if any(k in path for k in ("/location", "/map", "/directions", "แผนที่")):
        return "location_page"
    if any(k in path for k in ("/faq", "faq", "คำถาม")):
        return "faq_page"
    if any(k in path for k in ("/price", "/pricing", "/package", "ราคา", "แพ็กเกจ")) or flags.get("has_price_or_package"):
        return "price_package_page"
    if any(k in path for k in ("/service", "/treatment", "บริการ")):
        return "service_page"
    if any(k in path for k in ("/department", "/dept", "/center", "/centre", "แผนก", "ศูนย์")):
        return "department_page"
    if any(k in path for k in ("/directory", "/find-a", "/listing")):
        return "directory_page"
    if flags.get("has_faq"):
        return "faq_page"
    if flags.get("has_booking_or_appointment"):
        return "appointment_page"
    if flags.get("has_author") or flags.get("has_published_date"):
        return "article"
    return "unknown"


def extract_content_features(page: dict, prompt: str, intent: str, source_type: str,
                             sim_engine: SimilarityEngine) -> dict:
    """Heuristic content features for one scraped page (used to explain citation).

    All flags are transparent keyword / structure heuristics, not ground truth.
    """
    markdown = page.get("markdown") or ""
    text = page.get("text") or markdown or ""
    title = page.get("title") or ""
    html = page.get("html") or ""          # usually absent (crawler returns markdown/text)
    headings = extract_headings(markdown)
    norm_all = _norm("\n".join([title, " ".join(headings), text]))

    flags: dict = {}
    for feat, kws in _KW.items():
        flags[feat] = any(_norm(kw) in norm_all for kw in kws)

    flags["has_phone_number"] = bool(_PHONE_RE.search(text))
    flags["has_email"] = bool(_EMAIL_RE.search(_norm(text)))
    flags["has_schema"] = any(m in _norm(text + " " + html) for m in ("schema.org", "ld+json", "application/ld+json"))

    md_lines = markdown.splitlines()
    pipe_lines = sum(1 for ln in md_lines if ln.strip().count("|") >= 2)
    flags["has_table"] = pipe_lines >= 2 or "|--" in markdown.replace(" ", "") or "<table" in _norm(html)
    flags["has_bullets"] = any(re.match(r"^\s*([-*•]|\d+[.)])\s+", ln) for ln in md_lines)
    flags["has_many_headings"] = len(headings) >= 5

    # prompt-aligned signals
    if prompt and headings:
        flags["heading_prompt_match"] = max((sim_engine.score(prompt, h) for h in headings), default=0.0) >= 0.12
    else:
        flags["heading_prompt_match"] = False

    intent_tokens = [t for t in re.findall(r"[a-z]{3,}", _norm(intent)) if t not in _GENERIC_INTENT_TOKENS]
    ntitle = _norm(title)
    flags["title_contains_intent_terms"] = any(
        re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", ntitle) for t in intent_tokens
    ) if intent_tokens and ntitle else False

    answer_like = sim_engine.score(prompt, text[:500]) if (prompt and text) else None

    out = {k: bool(flags.get(k)) for k in CONTENT_BOOL_FEATURES}
    out["answer_like_text_in_first_500_chars"] = round(answer_like, 4) if answer_like is not None else None
    out["page_type"] = _page_type(page.get("url") or "", source_type, flags)
    return out


# --------------------------------------------------------------------------- #
# record-level visibility (one row per prompt/record; ALL prompts kept)
# --------------------------------------------------------------------------- #
def compute_records(run: dict, pages: dict | None,
                    default_client_terms: list[str], default_competitor_terms: list[str]) -> list[dict]:
    pages = pages or {}
    rows: list[dict] = []
    for rec in run.get("records", []):
        client_terms = rec.get("client_terms") or default_client_terms
        comp_terms = rec.get("competitor_terms") or default_competitor_terms
        cpat, kpat = compile_terms(client_terms), compile_terms(comp_terms)

        prompt = rec.get("prompt", "") or ""
        answer = rec.get("answer_text", "") or rec.get("answer_markdown", "") or ""

        prompt_has_client = bool(_match_compiled(prompt, cpat))
        prompt_has_comp = bool(_match_compiled(prompt, kpat))
        explicit = rec.get("prompt_is_nonbranded")
        is_nonbranded = bool(explicit) if explicit is not None else (not prompt_has_client)

        client_in_answer = bool(_match_compiled(answer, cpat))
        comp_in_answer = bool(_match_compiled(answer, kpat))

        n_c = n_c_cited = n_c_more = 0
        n_k = n_k_cited = n_k_more = 0
        for s in rec.get("sources", []):
            blob = _source_blobs(s, pages.get(s.get("normalized_url")))
            mc = bool(_match_compiled(blob, cpat)) if cpat else False
            mk = bool(_match_compiled(blob, kpat)) if kpat else False
            cited = s.get("cited_label") == 1
            if mc:
                n_c += 1
                n_c_cited += 1 if cited else 0
                n_c_more += 0 if cited else 1
            if mk:
                n_k += 1
                n_k_cited += 1 if cited else 0
                n_k_more += 0 if cited else 1

        client_in_sources, comp_in_sources = n_c > 0, n_k > 0
        client_cited, comp_cited = n_c_cited > 0, n_k_cited > 0
        # record-level "more-only" = appeared but NEVER cited (cited wins, mirrors the parser)
        client_more_only = client_in_sources and not client_cited
        comp_more_only = comp_in_sources and not comp_cited
        client_appeared = client_in_answer or client_in_sources
        comp_appeared = comp_in_answer or comp_in_sources

        rows.append({
            "run_id": run.get("run_id"),
            "record_id": rec.get("record_id"),
            "prompt_id": rec.get("prompt_id"),
            "prompt_hash": rec.get("prompt_hash"),
            "topic": rec.get("topic") or "",
            "intent": rec.get("intent") or "",
            "prompt": prompt,
            "answer_text": answer[:2000],
            "visibility_goal": rec.get("visibility_goal") or "",
            "prompt_contains_client_brand": prompt_has_client,
            "prompt_contains_competitor_brand": prompt_has_comp,
            "is_nonbranded_prompt": is_nonbranded,
            "client_appeared_in_answer": client_in_answer,
            "client_appeared_in_sources": client_in_sources,
            "client_cited": client_cited,
            "client_more_only": client_more_only,
            "n_client_sources": n_c,
            "n_client_cited_sources": n_c_cited,
            "n_client_more_only_sources": n_c_more,
            "competitor_appeared_in_answer": comp_in_answer,
            "competitor_appeared_in_sources": comp_in_sources,
            "competitor_cited": comp_cited,
            "competitor_more_only": comp_more_only,
            "n_competitor_sources": n_k,
            "n_competitor_cited_sources": n_k_cited,
            "n_competitor_more_only_sources": n_k_more,
            # convenience aggregates
            "client_appeared": client_appeared,
            "competitor_appeared": comp_appeared,
            "any_target_brand_appeared": client_appeared or comp_appeared,
            "any_target_brand_cited": client_cited or comp_cited,
            "has_brand_terms": bool(cpat or kpat),
        })
    return rows


# --------------------------------------------------------------------------- #
# intent rollup (denominator = number of NON-BRANDED prompts in the intent)
# --------------------------------------------------------------------------- #
def summarize_by_intent(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["topic"], r["intent"])].append(r)

    rows = []
    for (topic, intent), rs in sorted(groups.items()):
        nb = [r for r in rs if r["is_nonbranded_prompt"]]
        denom = len(nb)

        def share(pred):
            return round(sum(1 for r in nb if pred(r)) / denom, 3) if denom else 0.0

        def count(pred):
            return sum(1 for r in nb if pred(r))

        client_cited_rate = share(lambda r: r["client_cited"])
        comp_cited_rate = share(lambda r: r["competitor_cited"])
        ex_client = [r["prompt"][:90] for r in nb if r["client_cited"]][:3]
        ex_comp = [r["prompt"][:90] for r in nb if r["competitor_cited"]][:3]
        rows.append({
            "topic": topic, "intent": intent,
            "total_prompts": len(rs), "nonbranded_prompts": denom,
            "client_appeared_prompts": count(lambda r: r["client_appeared"]),
            "client_appeared_rate": share(lambda r: r["client_appeared"]),
            "client_cited_prompts": count(lambda r: r["client_cited"]),
            "client_cited_rate": client_cited_rate,
            "client_more_only_prompts": count(lambda r: r["client_more_only"]),
            "client_more_only_rate": share(lambda r: r["client_more_only"]),
            "competitor_appeared_prompts": count(lambda r: r["competitor_appeared"]),
            "competitor_appeared_rate": share(lambda r: r["competitor_appeared"]),
            "competitor_cited_prompts": count(lambda r: r["competitor_cited"]),
            "competitor_cited_rate": comp_cited_rate,
            "competitor_more_only_prompts": count(lambda r: r["competitor_more_only"]),
            "competitor_more_only_rate": share(lambda r: r["competitor_more_only"]),
            "client_vs_competitor_cited_delta": round(client_cited_rate - comp_cited_rate, 3),
            "top_example_client_cited_prompts": " | ".join(ex_client),
            "top_example_competitor_cited_prompts": " | ".join(ex_comp),
        })
    return rows


# --------------------------------------------------------------------------- #
# source/page-level brand analysis (ONLY client/competitor-matched sources)
# --------------------------------------------------------------------------- #
def build_source_pages(run: dict, features: list[dict], pages: dict | None,
                       sim_engine: SimilarityEngine | None,
                       default_client_terms: list[str], default_competitor_terms: list[str]) -> list[dict]:
    pages = pages or {}
    sim = sim_engine or SimilarityEngine("lexical")
    feat_by_id = {f.get("source_id"): f for f in (features or [])}
    rows: list[dict] = []

    for rec in run.get("records", []):
        client_terms = rec.get("client_terms") or default_client_terms
        comp_terms = rec.get("competitor_terms") or default_competitor_terms
        cpat, kpat = compile_terms(client_terms), compile_terms(comp_terms)
        if not cpat and not kpat:
            continue

        for s in rec.get("sources", []):
            page = pages.get(s.get("normalized_url"))
            blob = _source_blobs(s, page)
            mc = _match_compiled(blob, cpat) if cpat else []
            mk = _match_compiled(blob, kpat) if kpat else []
            if not mc and not mk:
                continue
            group = "both" if (mc and mk) else ("client" if mc else "competitor")

            f = feat_by_id.get(s.get("source_id"), {})
            stype = f.get("source_type") or classify(s.get("url", ""))[0]
            scraped = bool(page and page.get("status") == "success")
            content = (extract_content_features(page, rec.get("prompt", ""), rec.get("intent", ""), stype, sim)
                       if scraped else _empty_content())
            canon = s.get("canonical_url") or (page or {}).get("canonical_url")
            final = s.get("final_url") or (page or {}).get("final_url")

            row = {
                "run_id": run.get("run_id"), "record_id": rec.get("record_id"),
                "prompt_id": rec.get("prompt_id"), "topic": rec.get("topic") or "",
                "intent": rec.get("intent") or "", "prompt": rec.get("prompt", ""),
                "source_id": s.get("source_id"), "url": s.get("url"),
                "normalized_url": s.get("normalized_url"), "final_url": final, "canonical_url": canon,
                "domain": s.get("domain"),
                "canonical_host": (domain(canon) if canon else (domain(final) if final else s.get("domain"))),
                "title": s.get("title"), "description": s.get("description"),
                "source_group": s.get("source_group"), "cited": s.get("cited_label"),
                "source_position": s.get("source_position"), "observed_rank": s.get("observed_rank"),
                "brand_match_group": group,
                "matched_terms": "; ".join(dict.fromkeys(mc + mk)),
                "is_client_source": bool(mc), "is_competitor_source": bool(mk),
                "source_type": stype,
                "institutional_official": f.get("institutional_official"),
                "brand_official_candidate": f.get("brand_official_candidate"),
                "scrape_success": scraped,
                "word_count": f.get("word_count"), "char_count": f.get("char_count"),
                "heading_count": f.get("heading_count"), "freshness_days": f.get("freshness_days"),
                "title_prompt_similarity": f.get("title_prompt_similarity"),
                "description_prompt_similarity": f.get("description_prompt_similarity"),
                "page_prompt_similarity": f.get("page_prompt_similarity"),
                "max_chunk_prompt_similarity": f.get("max_chunk_prompt_similarity"),
                "page_answer_similarity": f.get("page_answer_similarity"),
                "max_chunk_answer_similarity": f.get("max_chunk_answer_similarity"),
                "page_type": content["page_type"],
            }
            row.update({k: content[k] for k in CONTENT_BOOL_FEATURES + CONTENT_NUM_FEATURES})
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# cited vs more-only content comparison
# --------------------------------------------------------------------------- #
COMPARE_FEATURES = CONTENT_BOOL_FEATURES + CONTENT_NUM_FEATURES + [
    "word_count", "heading_count", "char_count", "freshness_days",
    "title_prompt_similarity", "description_prompt_similarity",
    "page_prompt_similarity", "max_chunk_prompt_similarity",
    "page_answer_similarity", "max_chunk_answer_similarity", "source_position",
]


def _compare_rows(rows: list[dict], group: str, topic: str, intent: str) -> list[dict]:
    cited = [r for r in rows if r.get("cited") == 1]
    more = [r for r in rows if r.get("cited") == 0]
    out = []
    for feat in COMPARE_FEATURES:
        cv = [r.get(feat) for r in cited]
        mv = [r.get(feat) for r in more]
        cm, mm = _mean(cv), _mean(mv)
        out.append({
            "group": group, "topic": topic, "intent": intent, "feature": feat,
            "cited_mean": cm, "more_only_mean": mm,
            "cited_median": _median(cv), "more_only_median": _median(mv),
            "delta": round(cm - mm, 4) if (cm is not None and mm is not None) else None,
            "n_cited": _n(cv), "n_more_only": _n(mv),
        })
    return out


def compare_cited_more_only(source_pages: list[dict]) -> list[dict]:
    out: list[dict] = []
    out += _compare_rows(source_pages, "all", "(all)", "(all)")
    out += _compare_rows([r for r in source_pages if r["is_client_source"]], "client", "(all)", "(all)")
    out += _compare_rows([r for r in source_pages if r["is_competitor_source"]], "competitor", "(all)", "(all)")
    by_intent: dict[tuple, list] = defaultdict(list)
    for r in source_pages:
        by_intent[(r["topic"], r["intent"])].append(r)
    for (topic, intent), rs in sorted(by_intent.items()):
        out += _compare_rows(rs, "by_intent", topic, intent)
    return out


# --------------------------------------------------------------------------- #
# position-controlled comparison (cited vs more-only within similar position bands)
# --------------------------------------------------------------------------- #
def _band(pos) -> str:
    try:
        p = int(pos)
    except (TypeError, ValueError):
        return "unknown"
    if p <= 0:
        return "unknown"
    if p <= 3:
        return "1-3"
    if p <= 6:
        return "4-6"
    if p <= 10:
        return "7-10"
    return "11+"


def position_controlled(source_pages: list[dict]) -> list[dict]:
    out: list[dict] = []
    slices = (
        ("all", source_pages),
        ("client", [r for r in source_pages if r["is_client_source"]]),
        ("competitor", [r for r in source_pages if r["is_competitor_source"]]),
    )
    for group, subset in slices:
        bands: dict[str, list] = defaultdict(list)
        for r in subset:
            pos = r.get("source_position")
            pos = pos if pos is not None else r.get("observed_rank")
            bands[_band(pos)].append(r)
        for band in POSITION_BANDS:
            rs = bands.get(band, [])
            if not rs:
                continue
            cited = [r for r in rs if r.get("cited") == 1]
            more = [r for r in rs if r.get("cited") == 0]
            for feat in COMPARE_FEATURES:
                cv = [r.get(feat) for r in cited]
                mv = [r.get(feat) for r in more]
                cm, mm = _mean(cv), _mean(mv)
                out.append({
                    "topic": "(all)", "intent": "(all)", "brand_match_group": group,
                    "position_band": band, "feature": feat,
                    "cited_mean": cm, "more_only_mean": mm,
                    "delta": round(cm - mm, 4) if (cm is not None and mm is not None) else None,
                    "n_cited": _n(cv), "n_more_only": _n(mv),
                })
    return out


# --------------------------------------------------------------------------- #
# overall summary + example prompts + client-vs-competitor table
# --------------------------------------------------------------------------- #
def overall_summary(records: list[dict]) -> dict:
    nb = [r for r in records if r["is_nonbranded_prompt"]]
    denom = len(nb)

    def share(pred):
        return round(sum(1 for r in nb if pred(r)) / denom, 3) if denom else 0.0

    return {
        "total_prompts": len(records),
        "nonbranded_prompts": denom,
        "client_appeared_rate": share(lambda r: r["client_appeared"]),
        "client_cited_rate": share(lambda r: r["client_cited"]),
        "client_more_only_rate": share(lambda r: r["client_more_only"]),
        "competitor_appeared_rate": share(lambda r: r["competitor_appeared"]),
        "competitor_cited_rate": share(lambda r: r["competitor_cited"]),
        "competitor_more_only_rate": share(lambda r: r["competitor_more_only"]),
        "any_target_brand_appeared_rate": share(lambda r: r["any_target_brand_appeared"]),
        "any_target_brand_cited_rate": share(lambda r: r["any_target_brand_cited"]),
        "client_vs_competitor_cited_delta": round(
            share(lambda r: r["client_cited"]) - share(lambda r: r["competitor_cited"]), 3),
        "n_client_source_pages": sum(r["n_client_sources"] for r in records),
        "n_competitor_source_pages": sum(r["n_competitor_sources"] for r in records),
    }


def build_examples(records: list[dict], n: int = 8) -> dict:
    nb = [r for r in records if r["is_nonbranded_prompt"]]

    def ex(pred):
        return [{"prompt_id": r.get("prompt_id"), "topic": r["topic"], "intent": r["intent"],
                 "prompt": r["prompt"][:140]} for r in nb if pred(r)][:n]

    return {
        "client_cited": ex(lambda r: r["client_cited"]),
        "competitor_cited_client_absent": ex(lambda r: r["competitor_cited"] and not r["client_appeared"]),
        "client_more_only": ex(lambda r: r["client_more_only"]),
        "neither_appeared": ex(lambda r: not r["any_target_brand_appeared"]),
    }


def client_vs_competitor(by_intent: list[dict], summary: dict) -> list[dict]:
    rows = [{
        "scope": "(overall)", "topic": "(all)", "intent": "(all)",
        "nonbranded_prompts": summary["nonbranded_prompts"],
        "client_appeared_rate": summary["client_appeared_rate"],
        "client_cited_rate": summary["client_cited_rate"],
        "competitor_appeared_rate": summary["competitor_appeared_rate"],
        "competitor_cited_rate": summary["competitor_cited_rate"],
        "client_vs_competitor_cited_delta": summary["client_vs_competitor_cited_delta"],
    }]
    for r in by_intent:
        rows.append({
            "scope": "intent", "topic": r["topic"], "intent": r["intent"],
            "nonbranded_prompts": r["nonbranded_prompts"],
            "client_appeared_rate": r["client_appeared_rate"],
            "client_cited_rate": r["client_cited_rate"],
            "competitor_appeared_rate": r["competitor_appeared_rate"],
            "competitor_cited_rate": r["competitor_cited_rate"],
            "client_vs_competitor_cited_delta": r["client_vs_competitor_cited_delta"],
        })
    return rows


# --------------------------------------------------------------------------- #
# top-level assembler
# --------------------------------------------------------------------------- #
def build_brand_visibility(run: dict, features: list[dict] | None = None, pages: dict | None = None,
                           sim_engine: SimilarityEngine | None = None,
                           default_client_terms: list[str] | None = None,
                           default_competitor_terms: list[str] | None = None) -> dict:
    """Assemble the full Non-branded Brand Visibility Audit for a ChatGPT run.

    Safe to call even with no manifest / no brand terms / no scraped pages: it
    returns every table (possibly empty) so exports always generate.
    """
    dft_c = list(default_client_terms if default_client_terms is not None else DEFAULT_CLIENT_BRAND_TERMS)
    dft_k = list(default_competitor_terms if default_competitor_terms is not None else DEFAULT_COMPETITOR_BRAND_TERMS)

    records = compute_records(run, pages, dft_c, dft_k)
    by_intent = summarize_by_intent(records)
    source_pages = build_source_pages(run, features or [], pages, sim_engine, dft_c, dft_k)
    summary = overall_summary(records)

    client_terms = sorted({t for r in run.get("records", []) for t in (r.get("client_terms") or [])} | set(dft_c))
    comp_terms = sorted({t for r in run.get("records", []) for t in (r.get("competitor_terms") or [])} | set(dft_k))

    return {
        "records": records,
        "by_intent": by_intent,
        "source_pages": source_pages,
        "cited_vs_moreonly": compare_cited_more_only(source_pages),
        "by_position_band": position_controlled(source_pages),
        "summary": summary,
        "examples": build_examples(records),
        "client_vs_competitor": client_vs_competitor(by_intent, summary),
        "client_terms": client_terms,
        "competitor_terms": comp_terms,
        "has_terms": bool(client_terms or comp_terms),
        "n_records": len(records),
        "n_source_pages": len(source_pages),
        "n_scraped_source_pages": sum(1 for r in source_pages if r.get("scrape_success")),
        "compare_features": COMPARE_FEATURES,
        "content_bool_features": CONTENT_BOOL_FEATURES,
    }
