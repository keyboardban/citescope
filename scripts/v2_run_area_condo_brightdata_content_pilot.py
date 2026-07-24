#!/usr/bin/env python3
"""Run an isolated Bright Data content-quality pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.brightdata_config import require_live_brightdata_config
from src.econometrics_eda_v2.scrape_providers.brightdata_provider import (
    download_brightdata_snapshot,
    normalize_brightdata_response,
    prepare_brightdata_url,
    scrape_url_brightdata,
    trigger_brightdata_crawler_async,
    wait_for_brightdata_snapshot,
)
from src.url_utils import normalize_url


WEAK_FLAGS = {"parse_failed", "empty_text", "very_short_text", "boilerplate_only", "nav_footer_only", "dynamic_js_likely", "blocked_or_error_page"}


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]


def _crawler_response_url(page: dict) -> str:
    request_input = page.get("input") if isinstance(page.get("input"), dict) else {}
    return str(page.get("url") or page.get("final_url") or request_input.get("url") or "")


def _crawler_progress_reporter(batch_id: str, batch_number: int, total_batches: int, snapshot_id: str):
    started = time.monotonic()

    def report(progress: dict) -> None:
        status = str(progress.get("status") or "waiting")
        details = []
        for key in ("progress", "records", "errors", "completed", "processed", "pending", "total"):
            value = progress.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        suffix = f" {' '.join(details)}" if details else ""
        elapsed = int(time.monotonic() - started)
        print(
            f"Waiting {batch_id} ({batch_number}/{total_batches}) snapshot {snapshot_id}: "
            f"status={status} elapsed={elapsed}s{suffix}",
            flush=True,
        )

    return report


def _content_strength(result: dict) -> str:
    if not bool(result.get("success")) or not str(result.get("text") or "").strip():
        return "failed"
    words = int(result.get("word_count") or 0)
    if words >= 300 and result.get("content_quality_flag") == "ok":
        return "strong"
    if words >= 100:
        return "medium"
    return "weak"


def _is_weak(result: dict) -> bool:
    return not bool(result.get("success")) or int(result.get("word_count") or 0) < 100 or str(result.get("content_quality_flag") or "") in WEAK_FLAGS


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _select_pilot(sources: pd.DataFrame, max_urls: int, provider_mode: str) -> pd.DataFrame:
    work = sources.copy()
    work["cited"] = pd.to_numeric(work["cited"], errors="coerce").fillna(0).astype(int)
    grouped = work.groupby("normalized_url", dropna=False).agg(
        source_url=("source_url", "first"), source_root_domain=("source_root_domain", "first"),
        source_type=("source_type", "first"), source_rows=("cited", "size"), cited_rows=("cited", "sum"),
        intent_n=("intent", "nunique"),
    ).reset_index()
    grouped = grouped[grouped["normalized_url"].fillna("").ne("")].copy()
    grouped["cited"] = grouped["cited_rows"].gt(0).astype(int)
    grouped["initial_mode"] = provider_mode
    grouped["selection_reason"] = "high_impact_cited"
    difficult = grouped.iloc[0:0].copy()
    if provider_mode == "browser_api":
        difficult = grouped[grouped["source_type"].isin(["forum", "social", "video"]) | grouped["source_root_domain"].eq("reddit.com")].copy()
        difficult = difficult.sort_values(["cited_rows", "source_rows"], ascending=False).head(min(5, max_urls))
        difficult["initial_mode"] = "unlocker_api"
        difficult["selection_reason"] = "social_forum_unlocker_first"

    remaining = grouped[~grouped["normalized_url"].isin(difficult["normalized_url"])].copy()
    # Cap one domain at three selections so the pilot covers several website types.
    picked: list[pd.Series] = []
    per_domain: dict[str, int] = {}
    def pick_from(frame: pd.DataFrame, target: int, reason: str) -> None:
        for _, row in frame.sort_values(["cited_rows", "source_rows"], ascending=False).iterrows():
            domain = str(row["source_root_domain"])
            if per_domain.get(domain, 0) >= 3 or len(picked) >= target:
                continue
            row = row.copy()
            row["selection_reason"] = reason
            picked.append(row)
            per_domain[domain] = per_domain.get(domain, 0) + 1
    cited_target = max_urls - len(difficult) - min(10, max_urls - len(difficult))
    pick_from(remaining[remaining["cited"].eq(1)], cited_target, "high_impact_cited")
    pick_from(remaining[remaining["cited"].eq(0)], max_urls - len(difficult), "more_only_comparator")
    pick_from(remaining, max_urls - len(difficult), "exposure_fill")
    selected = pd.concat([difficult, pd.DataFrame(picked)], ignore_index=True)
    return selected.head(max_urls).reset_index(drop=True)


def _select_all_urls(sources: pd.DataFrame, provider_mode: str) -> pd.DataFrame:
    work = sources.copy()
    work["cited"] = pd.to_numeric(work["cited"], errors="coerce").fillna(0).astype(int)
    grouped = work.groupby("normalized_url", dropna=False).agg(
        source_url=("source_url", "first"), source_root_domain=("source_root_domain", "first"),
        source_type=("source_type", "first"), source_rows=("cited", "size"), cited_rows=("cited", "sum"),
        intent_n=("intent", "nunique"),
    ).reset_index()
    grouped = grouped[grouped["normalized_url"].fillna("").ne("")].copy()
    grouped["cited"] = grouped["cited_rows"].gt(0).astype(int)
    grouped["initial_mode"] = provider_mode
    grouped["selection_reason"] = "all_unique_urls"
    return grouped.reset_index(drop=True)


def _call(url: str, mode: str, settings, raw_dir: Path, normalized_dir: Path, force: bool, retry_failed: bool) -> dict:
    key = _key(url)
    raw_path = raw_dir / mode / f"{key}.json"
    normalized_path = normalized_dir / mode / f"{key}.json"
    if normalized_path.exists() and raw_path.exists() and not force:
        cached = json.loads(normalized_path.read_text(encoding="utf-8"))
        if bool(cached.get("success")) or not retry_failed:
            return cached
    result = scrape_url_brightdata(url, mode, settings, live=True, raw_response_path="")
    _save_json(raw_path, {
        "requested_url": url, "provider_mode": mode, "request_payload": result.get("request_payload"),
        "request_params": result.get("request_params"),
        "response_headers": result.get("response_headers"), "raw_response": result.get("raw_response"),
    })
    normalized = result.get("normalized_result") or {}
    _save_json(normalized_path, normalized)
    return normalized


def _provider_error(url: str, mode: str, raw_dir: Path) -> dict[str, str]:
    path = raw_dir / mode / f"{_key(url)}.json"
    if not path.exists():
        return {"provider_error_code": "", "provider_error_message": ""}
    payload = json.loads(path.read_text(encoding="utf-8"))
    headers = {str(k).casefold(): str(v) for k, v in (payload.get("response_headers") or {}).items()}
    return {
        "provider_error_code": headers.get("x-brd-err-code", ""),
        "provider_error_message": headers.get("x-brd-err-msg", headers.get("x-brd-error", "")),
    }


def _checkpoint_path(raw_dir: Path) -> Path:
    return raw_dir / "crawler_api" / "async_snapshot_checkpoint.json"


def _run_crawler_async(queue: pd.DataFrame, settings, raw_dir: Path, normalized_dir: Path, resume_snapshot_id: str = "") -> dict[int, tuple[pd.Series, dict]]:
    urls = [str(url) for url in queue["source_url"]]
    checkpoint = _checkpoint_path(raw_dir)
    if resume_snapshot_id:
        snapshot_id = resume_snapshot_id
        triggered = {"snapshot_id": snapshot_id, "request_payload": [], "request_params": {}, "response_headers": {}, "raw_response": {}}
        _save_json(checkpoint, {"snapshot_id": snapshot_id, "status": "resuming", "resumed_at": datetime.now(UTC).isoformat(), "source_urls": urls})
        print(f"Resuming Crawler snapshot {snapshot_id} for {len(urls)} URLs; waiting for results.", flush=True)
    else:
        triggered = trigger_brightdata_crawler_async(urls, settings)
        snapshot_id = triggered["snapshot_id"]
        _save_json(checkpoint, {"snapshot_id": snapshot_id, "status": "submitted", "submitted_at": datetime.now(UTC).isoformat(), "source_urls": urls, "request_params": triggered["request_params"]})
        print(f"Crawler snapshot {snapshot_id} submitted for {len(urls)} URLs; waiting for results.", flush=True)
    try:
        progress = wait_for_brightdata_snapshot(
            snapshot_id,
            settings,
            progress_callback=_crawler_progress_reporter("batch_0001", 1, 1, snapshot_id),
        )
        _save_json(checkpoint, {"snapshot_id": snapshot_id, "status": "ready", "ready_at": datetime.now(UTC).isoformat(), "source_urls": urls, "progress": progress})
        pages = download_brightdata_snapshot(snapshot_id, settings)
    except Exception as exc:
        _save_json(checkpoint, {"snapshot_id": snapshot_id, "status": "interrupted_or_failed", "updated_at": datetime.now(UTC).isoformat(), "source_urls": urls, "error": str(exc)})
        raise
    by_url: dict[str, list[dict]] = {}
    for page in pages:
        key = normalize_url(_crawler_response_url(page))
        if key:
            by_url.setdefault(key, []).append(page)
    results: dict[int, tuple[pd.Series, dict]] = {}
    for position, (_, item) in enumerate(queue.iterrows(), start=1):
        source_url = str(item["source_url"])
        key = normalize_url(source_url)
        raw_page = (by_url.get(key) or [{}]).pop(0)
        if not raw_page:
            raw_page = {"error": f"Crawler snapshot {snapshot_id} contained no record for this URL."}
        normalized = normalize_brightdata_response(raw_page, prepare_brightdata_url(source_url), mode="crawler_api").to_dict()
        raw_path = raw_dir / "crawler_api" / f"{_key(source_url)}.json"
        normalized_path = normalized_dir / "crawler_api" / f"{_key(source_url)}.json"
        _save_json(raw_path, {"requested_url": source_url, "provider_mode": "crawler_api", "snapshot_id": snapshot_id, "snapshot_progress": progress, "request_payload": triggered["request_payload"], "request_params": triggered["request_params"], "response_headers": triggered["response_headers"], "raw_response": raw_page})
        _save_json(normalized_path, normalized)
        results[position] = (item, normalized)
    _save_json(checkpoint, {"snapshot_id": snapshot_id, "status": "completed", "completed_at": datetime.now(UTC).isoformat(), "source_urls": urls, "progress": progress, "result_record_count": len(pages)})
    return results


def _batch_checkpoint_path(raw_dir: Path) -> Path:
    return raw_dir / "crawler_api" / "async_batch_checkpoint.json"


def _cached_batch_results(batch: pd.DataFrame, normalized_dir: Path) -> dict[int, tuple[pd.Series, dict]]:
    results = {}
    for position, (_, item) in enumerate(batch.iterrows(), start=1):
        path = normalized_dir / "crawler_api" / f"{_key(str(item['source_url']))}.json"
        if not path.exists():
            return {}
        results[position] = (item, json.loads(path.read_text(encoding="utf-8")))
    return results


def _write_batch_state(state: dict, checkpoint: Path, out: Path) -> None:
    _save_json(checkpoint, state)
    rows = []
    for batch in state["batches"]:
        for source_url in batch["source_urls"]:
            rows.append({"batch_id": batch["batch_id"], "source_url": source_url, "snapshot_id": batch.get("snapshot_id", ""), "batch_status": batch["status"], "attempt_type": state["attempt_type"], "submitted_at": batch.get("submitted_at", ""), "completed_at": batch.get("completed_at", "")})
    pd.DataFrame(rows).to_csv(out / "crawler_async_batch_manifest.csv", index=False)


def _run_crawler_async_batches(queue: pd.DataFrame, settings, raw_dir: Path, normalized_dir: Path, out: Path, batch_size: int, resume: bool, attempt_type: str) -> tuple[dict[int, tuple[pd.Series, dict]], pd.DataFrame]:
    checkpoint = _batch_checkpoint_path(raw_dir)
    if resume:
        if not checkpoint.exists():
            raise RuntimeError(f"No batch checkpoint found: {checkpoint}")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("attempt_type") != attempt_type:
            raise RuntimeError("Checkpoint attempt type does not match this run.")
    else:
        batches = []
        for offset in range(0, len(queue), batch_size):
            batch = queue.iloc[offset:offset + batch_size]
            batches.append({"batch_id": f"batch_{len(batches) + 1:04d}", "source_urls": batch["source_url"].astype(str).tolist(), "status": "pending", "snapshot_id": ""})
        state = {"attempt_type": attempt_type, "created_at": datetime.now(UTC).isoformat(), "batch_size": batch_size, "batches": batches}
        _write_batch_state(state, checkpoint, out)

    source_index = {str(row.source_url): row for _, row in queue.iterrows()}
    all_results: dict[int, tuple[pd.Series, dict]] = {}
    audit_rows = []
    def write_coverage() -> None:
        pd.DataFrame(audit_rows).to_csv(out / "crawler_async_coverage_audit.csv", index=False)
    position = 1
    for batch_number, entry in enumerate(state["batches"], start=1):
        batch = pd.DataFrame([source_index[url] for url in entry["source_urls"]]).reset_index(drop=True)
        cached = _cached_batch_results(batch, normalized_dir) if entry["status"] == "completed" else {}
        if cached:
            for _, value in cached.items():
                all_results[position] = value; position += 1
            audit_rows.append({"batch_id": entry["batch_id"], "snapshot_id": entry.get("snapshot_id", ""), "input_urls": len(batch), "returned_records": entry.get("result_record_count", len(batch)), "matched_urls": len(batch), "missing_urls": 0, "duplicate_result_urls": 0, "parse_success_urls": sum(bool(x[1].get("success")) for x in cached.values()), "batch_status": "reused_completed_cache"})
            write_coverage()
            continue
        try:
            if entry.get("snapshot_id"):
                entry["status"] = "resuming"
                _write_batch_state(state, checkpoint, out)
                print(f"Resuming {entry['batch_id']} snapshot {entry['snapshot_id']} ({len(batch)} URLs).", flush=True)
            else:
                triggered = trigger_brightdata_crawler_async(batch["source_url"].astype(str).tolist(), settings)
                entry["snapshot_id"] = triggered["snapshot_id"]
                entry["status"] = "submitted"
                entry["submitted_at"] = datetime.now(UTC).isoformat()
                _write_batch_state(state, checkpoint, out)
                print(f"Submitted {entry['batch_id']} snapshot {entry['snapshot_id']} ({len(batch)} URLs).", flush=True)
            progress = wait_for_brightdata_snapshot(
                entry["snapshot_id"],
                settings,
                progress_callback=_crawler_progress_reporter(
                    entry["batch_id"], batch_number, len(state["batches"]), entry["snapshot_id"]
                ),
            )
            entry["status"] = "ready"; _write_batch_state(state, checkpoint, out)
            pages = download_brightdata_snapshot(entry["snapshot_id"], settings)
            by_url: dict[str, list[dict]] = {}
            for page in pages:
                key = normalize_url(_crawler_response_url(page))
                if key: by_url.setdefault(key, []).append(page)
            batch_results = {}
            matched = 0
            for local_position, (_, item) in enumerate(batch.iterrows(), start=1):
                source_url = str(item["source_url"])
                raw_page = (by_url.get(normalize_url(source_url)) or [{}]).pop(0)
                if raw_page: matched += 1
                else: raw_page = {"error": f"Crawler snapshot {entry['snapshot_id']} contained no record for this URL."}
                normalized = normalize_brightdata_response(raw_page, prepare_brightdata_url(source_url), mode="crawler_api").to_dict()
                _save_json(raw_dir / "crawler_api" / f"{_key(source_url)}.json", {"requested_url": source_url, "provider_mode": "crawler_api", "batch_id": entry["batch_id"], "snapshot_id": entry["snapshot_id"], "snapshot_progress": progress, "raw_response": raw_page})
                _save_json(normalized_dir / "crawler_api" / f"{_key(source_url)}.json", normalized)
                batch_results[local_position] = (item, normalized)
            entry["status"] = "completed"; entry["completed_at"] = datetime.now(UTC).isoformat(); entry["result_record_count"] = len(pages)
            entry.pop("error", None)
            _write_batch_state(state, checkpoint, out)
            print(
                f"Completed {entry['batch_id']}: {len(pages)} records returned; "
                f"matched {matched}/{len(batch)} submitted URLs.",
                flush=True,
            )
            for _, value in batch_results.items(): all_results[position] = value; position += 1
            page_keys = [normalize_url(_crawler_response_url(page)) for page in pages]
            audit_rows.append({"batch_id": entry["batch_id"], "snapshot_id": entry["snapshot_id"], "input_urls": len(batch), "returned_records": len(pages), "matched_urls": matched, "missing_urls": len(batch) - matched, "duplicate_result_urls": len(page_keys) - len(set(page_keys)), "parse_success_urls": sum(bool(x[1].get("success")) for x in batch_results.values()), "batch_status": "completed"})
            write_coverage()
        except (Exception, KeyboardInterrupt) as exc:
            entry["status"] = "interrupted_or_failed"; entry["error"] = str(exc); entry["updated_at"] = datetime.now(UTC).isoformat(); _write_batch_state(state, checkpoint, out)
            raise
    audit = pd.DataFrame(audit_rows)
    return all_results, audit


def _plot(df: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    overall = df["cited"].mean() * 100
    strength = df.groupby("content_strength").agg(source_urls=("normalized_url", "size"), cited_rate=("cited", "mean")).reindex(["strong", "medium", "weak", "failed"]).fillna(0).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(strength["content_strength"], strength["cited_rate"] * 100, color="#187b8d"); ax.axhline(overall, color="#a64646", linestyle="--", label="overall cited rate")
    for i, row in strength.iterrows(): ax.text(i, row["cited_rate"] * 100 + .7, f"n={int(row['source_urls'])}", ha="center")
    ax.set_ylabel("Cited rate (%)"); ax.set_title("Cited rate by final content strength"); ax.legend(); fig.tight_layout(); fig.savefig(output / "cited_rate_by_content_strength.png", dpi=170); plt.close(fig)
    providers = df["final_provider_mode"].value_counts().rename_axis("provider").reset_index(name="urls")
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(providers["provider"], providers["urls"], color="#b87820"); ax.set_ylabel("URLs"); ax.set_title("Final provider selected after quality check")
    for i, row in providers.iterrows(): ax.text(i, row["urls"] + .25, str(int(row["urls"])), ha="center")
    fig.tight_layout(); fig.savefig(output / "final_provider_selection.png", dpi=170); plt.close(fig)
    box = [df.loc[df["cited"].eq(value), "word_count"].dropna() for value in [0, 1]]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.boxplot(box, tick_labels=["More-only", "Cited"], showfliers=False); ax.set_ylabel("Final extracted word count"); ax.set_title("Content length by cited status (pilot only)"); fig.tight_layout(); fig.savefig(output / "word_count_by_cited_status.png", dpi=170); plt.close(fig)


def run(sources_path: Path, out: Path, figures: Path, max_urls: int, force: bool, retry_failed: bool, workers: int = 4, all_urls: bool = False, resume_snapshot_id: str = "", batch_size: int = 250, resume_batches: bool = False) -> dict:
    settings = require_live_brightdata_config()
    print(
        "Bright Data pilot config: "
        f"provider_mode={settings.provider_mode} "
        f"crawler_async={settings.crawler_async} "
        f"endpoint={settings.endpoint} "
        f"dataset_id_present={bool(settings.dataset_id)}",
        flush=True,
    )
    sources = pd.read_csv(sources_path, low_memory=False)
    previous_detail = None
    if retry_failed and settings.provider_mode == "crawler_api":
        detail_path = out / "brightdata_content_pilot_url_results.csv"
        if not detail_path.exists():
            raise RuntimeError(f"Failed-only retry requires an existing pilot table: {detail_path}")
        previous_detail = pd.read_csv(detail_path, low_memory=False)
        failed = ~previous_detail["scrape_success"].fillna(False) | pd.to_numeric(previous_detail["content_chars"], errors="coerce").fillna(0).eq(0)
        queue = previous_detail.loc[failed].copy()
        queue["initial_mode"] = "crawler_api"; queue["selection_reason"] = "failed_only_retry"
        queue[["normalized_url", "source_url", "source_root_domain", "scrape_success", "content_chars", "content_strength", "scrape_error"]].to_csv(out / "crawler_failed_retry_queue.csv", index=False)
        if queue.empty:
            return {"pilot_urls": int(len(previous_detail)), "retry_queue_urls": 0, "message": "No failed or empty-content URLs to retry."}
    else:
        queue = _select_all_urls(sources, settings.provider_mode) if all_urls else _select_pilot(sources, max_urls, settings.provider_mode)
    raw_dir, normalized_dir = out / "raw", out / "normalized"
    primary_results: dict[int, tuple[pd.Series, dict]] = {}
    if settings.provider_mode == "crawler_api" and settings.crawler_async:
        if all_urls or retry_failed:
            primary_results, _ = _run_crawler_async_batches(queue, settings, raw_dir, normalized_dir, out, batch_size, resume_batches, "failed_only_retry" if retry_failed else "all_unique_urls")
        else:
            primary_results = _run_crawler_async(queue, settings, raw_dir, normalized_dir, resume_snapshot_id)
    else:
        def fetch_primary(position: int, item: pd.Series) -> tuple[int, pd.Series, dict]:
            url, first = str(item["source_url"]), str(item["initial_mode"])
            print(f"[{position}/{len(queue)}] {first} {url}", flush=True)
            return position, item, _call(url, first, settings, raw_dir, normalized_dir, force, retry_failed)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(fetch_primary, position, item) for position, (_, item) in enumerate(queue.iterrows(), start=1)]
            for future in as_completed(futures):
                position, item, primary = future.result()
                primary_results[position] = (item, primary)

    fallbacks: dict[int, dict] = {}
    def fetch_fallback(position: int, item: pd.Series) -> tuple[int, dict]:
        print(f"[{position}/{len(queue)}] Browser response weak; retrying with unlocker_api", flush=True)
        return position, _call(str(item["source_url"]), "unlocker_api", settings, raw_dir, normalized_dir, force, retry_failed)
    if settings.provider_mode == "browser_api":
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(fetch_fallback, position, item) for position, (item, primary) in primary_results.items() if str(item["initial_mode"]) == "browser_api" and _is_weak(primary)]
            for future in as_completed(futures):
                position, fallback = future.result()
                fallbacks[position] = fallback

    rows = []
    for position in sorted(primary_results):
        item, primary = primary_results[position]
        first = str(item["initial_mode"])
        browser = primary if first == "browser_api" else None
        unlocker = primary if first == "unlocker_api" else fallbacks.get(position)
        crawler = primary if first == "crawler_api" else None
        fallback_used = position in fallbacks
        candidates = [x for x in [browser, unlocker, crawler] if x is not None]
        final = max(candidates, key=lambda x: (bool(x.get("success")), int(x.get("word_count") or 0), len(str(x.get("text") or ""))))
        rows.append({
            **item.to_dict(), "browser_attempted": browser is not None, "unlocker_attempted": unlocker is not None,
            "crawler_attempted": crawler is not None,
            "fallback_used": fallback_used, "browser_success": browser.get("success") if browser else pd.NA,
            "browser_word_count": browser.get("word_count") if browser else pd.NA,
            "browser_error": browser.get("error") if browser else "", "unlocker_success": unlocker.get("success") if unlocker else pd.NA,
            "unlocker_word_count": unlocker.get("word_count") if unlocker else pd.NA,
            "unlocker_error": unlocker.get("error") if unlocker else "", "final_provider_mode": final.get("provider_mode"),
            "crawler_success": crawler.get("success") if crawler else pd.NA,
            "crawler_word_count": crawler.get("word_count") if crawler else pd.NA,
            "crawler_error": crawler.get("error") if crawler else "",
            "scrape_success": final.get("success"), "final_url": final.get("final_url"), "status_code": final.get("status_code"),
            "final_request_url": final.get("requested_url"),
            "tracking_parameters_removed": str(item["source_url"]).strip() != str(final.get("requested_url") or "").strip(),
            "page_title": final.get("title"), "meta_description": final.get("meta_description"), "content_quality_flag": final.get("content_quality_flag"),
            "content_chars": final.get("text_char_count"), "word_count": final.get("word_count"), "heading_count": final.get("heading_count"),
            "table_count": final.get("table_count"), "link_count": final.get("link_count"), "content_strength": _content_strength(final),
            "scrape_error": final.get("error"), "page_text_excerpt": str(final.get("text") or "")[:1200],
            **_provider_error(str(item["source_url"]), first, raw_dir),
        })
    detail = pd.DataFrame(rows)
    if previous_detail is not None:
        detail = pd.concat([previous_detail.loc[~previous_detail["normalized_url"].isin(detail["normalized_url"])], detail], ignore_index=True, sort=False)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "brightdata_content_pilot_url_results.csv", index=False)
    tracking = detail[["source_url", "final_request_url", "tracking_parameters_removed", "final_provider_mode", "scrape_success", "content_strength", "scrape_error"]].copy()
    tracking.to_csv(out / "tracking_parameter_request_audit.csv", index=False)
    quality = detail.groupby(["final_provider_mode", "content_strength"], dropna=False).agg(urls=("normalized_url", "size"), scrape_success=("scrape_success", "sum"), median_word_count=("word_count", "median")).reset_index()
    quality.to_csv(out / "brightdata_content_pilot_quality_summary.csv", index=False)
    cited = detail.groupby("cited").agg(urls=("normalized_url", "size"), scrape_success_rate=("scrape_success", "mean"), strong_content_rate=("content_strength", lambda x: x.eq("strong").mean()), median_word_count=("word_count", "median"), median_heading_count=("heading_count", "median"), median_link_count=("link_count", "median")).reset_index()
    cited.to_csv(out / "brightdata_content_pilot_cited_comparison.csv", index=False)
    provider_errors = detail.assign(provider_error_code=detail["provider_error_code"].replace("", "none")).groupby(["final_provider_mode", "provider_error_code", "provider_error_message"], dropna=False).agg(urls=("normalized_url", "size"), scrape_success=("scrape_success", "sum")).reset_index().sort_values("urls", ascending=False)
    provider_errors.to_csv(out / "brightdata_content_pilot_provider_error_audit.csv", index=False)
    _plot(detail, figures)
    summary = {"pilot_urls": int(len(detail)), "browser_attempted": int(detail["browser_attempted"].sum()), "unlocker_attempted": int(detail["unlocker_attempted"].sum()), "crawler_attempted": int(detail["crawler_attempted"].sum()), "fallbacks": int(detail["fallback_used"].sum()), "final_scrape_success_rate": float(detail["scrape_success"].mean()), "strong_content_rate": float(detail["content_strength"].eq("strong").mean()), "batch_size": batch_size if all_urls or retry_failed else None, "output": str(out), "figures": str(figures)}
    _save_json(out / "brightdata_content_pilot_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--max-urls", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute-live-brightdata", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Re-request only cached failures; keep successful cache entries.")
    parser.add_argument("--all-urls", action="store_true", help="Submit every unique source URL in one async Crawler snapshot; ignores --max-urls.")
    parser.add_argument("--resume-snapshot", default="", help="Resume polling and downloading an existing Bright Data Crawler snapshot without triggering another job.")
    parser.add_argument("--resume-last-crawler-snapshot", action="store_true", help="Resume the snapshot ID stored in output-dir/raw/crawler_api/async_snapshot_checkpoint.json.")
    parser.add_argument("--batch-size", type=int, default=250, help="URLs per asynchronous Crawler snapshot for --all-urls or --retry-failed.")
    args = parser.parse_args()
    if not args.execute_live_brightdata:
        raise SystemExit("Live Bright Data calls require --execute-live-brightdata.")
    resume_snapshot_id = str(args.resume_snapshot or "")
    resume_batches = False
    if args.resume_last_crawler_snapshot:
        batch_checkpoint = _batch_checkpoint_path(args.output_dir / "raw")
        if batch_checkpoint.exists():
            resume_batches = True
        else:
            checkpoint = _checkpoint_path(args.output_dir / "raw")
            if not checkpoint.exists():
                raise SystemExit(f"No Crawler checkpoint found: {checkpoint}")
            resume_snapshot_id = str(json.loads(checkpoint.read_text(encoding="utf-8")).get("snapshot_id") or "")
            if not resume_snapshot_id:
                raise SystemExit(f"Crawler checkpoint has no snapshot_id: {checkpoint}")
    print(json.dumps(run(args.sources, args.output_dir, args.figure_dir, args.max_urls, args.force, args.retry_failed, args.workers, args.all_urls, resume_snapshot_id, args.batch_size, resume_batches), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
