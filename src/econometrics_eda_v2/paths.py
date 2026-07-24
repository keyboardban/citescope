"""Filesystem contract for the external econometrics research archive."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - base application installs python-dotenv
    load_dotenv = None


CODE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = CODE_ROOT.parent
LEGACY_WORKSPACE_ROOT = Path.home() / "Code" / "Metier" / "Research"
if load_dotenv is not None:
    load_dotenv(CODE_ROOT / ".env", override=False)


def relocate_workspace_path(path: Path) -> Path:
    """Translate paths saved under the former workspace root after a disk move."""
    expanded = path.expanduser()
    try:
        relative = expanded.relative_to(LEGACY_WORKSPACE_ROOT)
    except ValueError:
        return expanded.resolve()
    if WORKSPACE_ROOT == LEGACY_WORKSPACE_ROOT:
        return expanded.resolve()
    return (WORKSPACE_ROOT / relative).resolve()


def _configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = CODE_ROOT / path
    return relocate_workspace_path(path)


RESEARCH_ROOT = _configured_path("CITESCOPE_RESEARCH_DATA_DIR", CODE_ROOT)
ECONOMETRICS_DATA_DIR = _configured_path(
    "CITESCOPE_ECONOMETRICS_DATA_DIR",
    RESEARCH_ROOT / "data" / "econometrics_v2",
)
ECONOMETRICS_OUTPUT_DIR = _configured_path(
    "CITESCOPE_ECONOMETRICS_OUTPUT_DIR",
    RESEARCH_ROOT / "outputs" / "econometrics_eda_v2",
)


def topic_output_dir(topic_slug: str = "scope_condo_nonbranded") -> Path:
    """Return the generated-output directory for one topic study."""
    return ECONOMETRICS_OUTPUT_DIR / "topic_sensitivity" / topic_slug
