"""Test config: put project root on sys.path and isolate storage in a temp dir."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    """Point all storage paths at a per-test temp dir (no pollution of data/)."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr(config, "BATCHES_DIR", tmp_path / "batches")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "audit.db")
    config.ensure_dirs()
    yield
