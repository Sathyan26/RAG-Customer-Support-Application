"""Text normalization: turn messy ingested text into clean, comparable text.

Each function here does one thing and is unit-tested in isolation
(`tests/test_normalize.py`); `clean_text()` composes them into the pipeline
step actually used by `cleaning/pipeline.py`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def strip_html(text: str) -> str:
    """Remove HTML tags, unescaping entities as BeautifulSoup parses them."""
    if "<" not in text and "&" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text()


def normalize_unicode(text: str) -> str:
    """NFKC-normalize so visually-identical characters compare equal."""
    return unicodedata.normalize("NFKC", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, strip trailing/leading whitespace on
    each line (not just mid-line runs -- a line ending in "  \\n" and one
    ending in "\\n" must normalize identically for dedup hashing to work),
    and collapse excess blank lines."""
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def fix_shouting(text: str) -> str:
    """Sentence-case a line that's overwhelmingly uppercase.

    Real support exports occasionally have all-caps fields; leaving them
    as-is degrades both retrieval (embedding models are case-sensitive to a
    degree) and the generated answer's tone. We only touch lines that are
    almost entirely uppercase, so ordinary text with acronyms is untouched.
    """
    lines = text.split("\n")
    fixed = []
    for line in lines:
        letters = [c for c in line if c.isalpha()]
        if len(letters) >= 8 and sum(1 for c in letters if c.isupper()) / len(letters) > 0.9:
            line = line.capitalize()
        fixed.append(line)
    return "\n".join(fixed)


def clean_text(text: str) -> str:
    """The full normalization pipeline applied to one document's raw text."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = fix_shouting(text)
    text = collapse_whitespace(text)
    return text


def content_hash(text: str) -> str:
    """Stable hash of *cleaned* text, used for exact-duplicate detection.

    Hashing the cleaned rather than raw text is what lets two records that
    differ only in HTML wrapping/whitespace/casing (see the sample dataset's
    injected noise) collapse to the same hash after cleaning.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_low_quality(text: str, min_chars: int = 8) -> bool:
    """Heuristic filter for near-empty or junk records that slipped through
    ingestion (e.g. a response field that was just whitespace or a single
    punctuation character)."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    return alpha_chars / max(len(stripped), 1) < 0.3
