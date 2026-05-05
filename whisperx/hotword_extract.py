"""
Auto-extract a hotwords list from a free-form text prompt.

Whisper's `--initial_prompt` is fed as `<|startofprev|>` tokens, which makes
the model treat the prompt as a transcript that already happened. When the
prompt content overlaps with the start of the audio, Whisper happily skips
those words on the assumption they were "already said".

For vocabulary biasing — proper nouns, units, technical terms — the right
mechanism in faster-whisper is `hotwords`. This module turns a long prompt
text into a compact, deduplicated hotwords string by keeping only:

  * proper nouns (capitalized words that are not the first word of a sentence)
  * tokens carrying a unit / currency / degree symbol (m², €, %, °, ²)
  * "all-caps" acronyms of length >= 2 (e.g. "GPS", "HDMI")

Common French / English sentence-initial words ("Cette", "Construite", "The"…)
are filtered out. The resulting hotwords cap at `max_terms` to stay well below
Whisper's prompt budget.
"""
from __future__ import annotations

import re
from typing import Iterable

# Unit / symbol characters that mark a token as worth biasing toward.
_UNIT_CHARS = "²³€$£¥%°ǵ"

# Punctuation we strip from token edges before classification.
_PUNCT_EDGE = ".,;:!?…\"'()[]{}«»"

# Sentence boundary regex (keeps it lightweight — no NLTK dep).
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# A "word-ish" token: letters (incl. accented), digits, and the unit chars.
_WORD = re.compile(rf"[A-Za-zÀ-ÖØ-öø-ÿ0-9{re.escape(_UNIT_CHARS)}'’\-]+")


def _strip_edges(token: str) -> str:
    return token.strip(_PUNCT_EDGE)


def _is_acronym(token: str) -> bool:
    return len(token) >= 2 and token.isupper() and any(c.isalpha() for c in token)


def _has_unit(token: str) -> bool:
    return any(c in _UNIT_CHARS for c in token)


def _is_capitalized_word(token: str) -> bool:
    if len(token) < 2 or not token[0].isalpha():
        return False
    return token[0].isupper() and any(c.islower() for c in token[1:])


def extract_hotwords(text: str, max_terms: int = 30) -> str:
    """
    Return a comma-separated hotwords string built from `text`.

    The function preserves first-occurrence order so the most "narratively
    early" terms — typically the most relevant — survive the cap.
    """
    if not text or not text.strip():
        return ""

    seen: dict[str, None] = {}  # ordered set
    sentences = _SENT_SPLIT.split(text.strip())

    for sentence in sentences:
        tokens = _WORD.findall(sentence)
        for idx, raw in enumerate(tokens):
            tok = _strip_edges(raw)
            if not tok:
                continue
            # Drop single-letter / single-digit noise, but keep single unit
            # symbols (€, %, °…) on their own.
            if len(tok) == 1 and not _has_unit(tok):
                continue
            keep = False
            # Acronyms anywhere
            if _is_acronym(tok):
                keep = True
            # Tokens carrying a unit/currency symbol
            elif _has_unit(tok):
                keep = True
            # Capitalized mid-sentence → proper noun candidate
            elif idx > 0 and _is_capitalized_word(tok):
                keep = True
            if keep and tok not in seen:
                seen[tok] = None
                if len(seen) >= max_terms:
                    return ", ".join(seen)

    return ", ".join(seen)


def merge_hotwords(*sources: Iterable[str | None]) -> str | None:
    """
    Merge several hotwords strings, preserving order, deduplicating
    case-insensitively. Returns `None` when the merged result is empty,
    so callers can pass it straight to faster-whisper without changing
    the "no hotwords" sentinel.
    """
    seen: dict[str, str] = {}
    for src in sources:
        if not src:
            continue
        if isinstance(src, str):
            parts = [p.strip() for p in src.split(",")]
        else:
            parts = [str(p).strip() for p in src]
        for p in parts:
            if not p:
                continue
            key = p.lower()
            if key not in seen:
                seen[key] = p
    return ", ".join(seen.values()) if seen else None
