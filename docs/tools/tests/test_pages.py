"""Checks over the written pages that the site build does not make."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGES = sorted((Path(__file__).resolve().parents[2] / "src" / "content" / "docs").rglob("*.md*"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_figure_is_shown_both_as_a_figure_and_as_an_interactive_fallback(page):
    """A fallback prints on paper; with the same figure on the page it would print twice."""
    text = page.read_text()
    figures = {f"{s}/{n}" for s, n in re.findall(r'<Figure\s+src="([^"]+)"\s+name="([^"]+)"', text)}
    fallbacks = set(re.findall(r'fallback="([^"]+)"', text))
    assert not figures & fallbacks
