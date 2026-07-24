from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import CODE_ROOT as PROJECT_ROOT
from .paths import ECONOMETRICS_DATA_DIR as DATA_DIR
from .paths import ECONOMETRICS_OUTPUT_DIR as OUTPUT_DIR

AUDIT_DIR = DATA_DIR / "audit"
EXPORTS_DIR = DATA_DIR / "exports"
QUEUE_DIR = DATA_DIR / "scrape_queue"
RAW_CACHE_DIR = DATA_DIR / "scrape_cache" / "raw"
PARSED_CACHE_DIR = DATA_DIR / "scrape_cache" / "parsed"
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_v2_dirs() -> None:
    for p in [
        DATA_DIR / "raw_inputs" / "ai_search_outputs",
        DATA_DIR / "raw_inputs" / "manifests",
        QUEUE_DIR,
        RAW_CACHE_DIR,
        PARSED_CACHE_DIR,
        EXPORTS_DIR,
        AUDIT_DIR,
        TABLES_DIR,
        PLOTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), "utf-8")


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def env_token_present(name: str = "APIFY_TOKEN") -> bool:
    import os

    return bool(os.environ.get(name))
