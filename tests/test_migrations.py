"""Chuỗi migration phải có mã duy nhất và đúng một head (nếu không `flask db upgrade` báo lỗi)."""
from __future__ import annotations

import re
from pathlib import Path

VERSIONS = Path(__file__).resolve().parent.parent / "migrations" / "versions"


def _read(pattern, text):
    m = re.search(pattern + r"\s*=\s*['\"]?([0-9a-f]+|None)['\"]?", text, re.M)
    return m.group(1) if m else None


def test_revisions_unique_and_single_head():
    revs, downs = [], set()
    for path in VERSIONS.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        revs.append(_read(r"^revision", text))
        downs.add(_read(r"^down_revision", text))
    assert len(revs) == len(set(revs)), "trùng revision id"
    heads = set(revs) - downs
    assert len(heads) == 1, f"nhiều head: {heads}"
