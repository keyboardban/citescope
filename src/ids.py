"""Small, dependency-free ID and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Sortable, human-readable run id: YYYYmmdd-HHMMSS-<6 hex>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rand = hashlib.sha1(now_iso().encode() + ts.encode()).hexdigest()[:6]
    return f"{ts}-{rand}"


def stable_hash(payload: Any) -> str:
    """Deterministic hash of any JSON-serialisable payload (for caching keys)."""
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def short_id(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]
