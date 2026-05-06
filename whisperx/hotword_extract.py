"""
Auto-extract a hotwords list from a free-form text prompt.

Whisper's `--initial_prompt` is fed as `<|startofprev|>` tokens, which makes
the model treat the prompt as a transcript that already happened. When the
prompt content overlaps with the start of the audio, Whisper happily skips
those words on the assumption they were "already said".

For vocabulary biasing — proper nouns, units, technical terms, numbers —
the right mechanism in faster-whisper is `hotwords`. This module turns a
long prompt text into a compact, deduplicated hotwords string.

Three extraction modes are available:

  * **names**  — keep only proper nouns (capitalized mid-sentence words),
    unit/currency/degree-bearing tokens (m², €, %, °, ²) and ALL-CAPS
    acronyms of length >= 2.
  * **formatted** *(default)* — `names` plus tokens carrying digits and
    bigrams of the form ``<digits> <digits>`` so that spaced numbers like
    "1 590" or "1 231" are biased toward the spaced spelling. This is the
    sweet spot for transcripts where exact figures and units matter.
  * **all** — `formatted` plus every word of length >= 2. Closest to the
    biasing strength of `--initial_prompt`, but accepts a (small) risk
    that faster-whisper's `<|startofprev|>` semantics make Whisper skip
    bits of the first sentence — same trade-off as `--initial_prompt`.

Common French / English sentence-initial words ("Cette", "Construite",
"The"…) are filtered out in `names` and `formatted` modes. The output
is capped at `max_terms` to stay well below Whisper's prompt budget.
"""
from __future__ import annotations

import re
from typing import Iterable, Literal

# Unit / symbol characters that mark a token as worth biasing toward.
_UNIT_CHARS = "²³€$£¥%°ǵ"

# Punctuation we strip from token edges before classification.
_PUNCT_EDGE = ".,;:!?…\"'()[]{}«»"

# Sentence boundary regex (keeps it lightweight — no NLTK dep).
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# A "word-ish" token: letters (incl. accented), digits, and the unit chars.
_WORD = re.compile(rf"[A-Za-zÀ-ÖØ-öø-ÿ0-9{re.escape(_UNIT_CHARS)}'’\-]+")

Mode = Literal["names", "formatted", "all"]


def _strip_edges(token: str) -> str:
    return token.strip(_PUNCT_EDGE)


def _is_acronym(token: str) -> bool:
    return len(token) >= 2 and token.isupper() and any(c.isalpha() for c in token)


def _has_unit(token: str) -> bool:
    return any(c in _UNIT_CHARS for c in token)


def _has_digit(token: str) -> bool:
    return any(c.isdigit() for c in token)


def _is_all_digits(token: str) -> bool:
    return len(token) > 0 and all(c.isdigit() for c in token)


def _is_capitalized_word(token: str) -> bool:
    if len(token) < 2 or not token[0].isalpha():
        return False
    return token[0].isupper() and any(c.islower() for c in token[1:])


def extract_hotwords(
    text: str,
    max_terms: int = 30,
    mode: Mode = "formatted",
) -> str:
    """
    Return a comma-separated hotwords string built from `text`.

    `mode` controls how aggressive the extraction is:

      - "names":     proper nouns + unit-bearing tokens + acronyms only.
      - "formatted": adds digit-bearing tokens and "<digits> <digits>"
                     bigrams so spaced numbers survive (default).
      - "all":       adds every word of length >= 2.

    First-occurrence order is preserved so the most "narratively early"
    terms — typically the most relevant — survive the cap.
    """
    if not text or not text.strip():
        return ""

    if mode not in ("names", "formatted", "all"):
        raise ValueError(
            f"Unknown auto_hotwords mode {mode!r}; expected one of "
            "'names', 'formatted', 'all'."
        )

    seen: dict[str, None] = {}  # ordered set, key = case-insensitive form

    def _try_add(value: str) -> bool:
        """Add `value` if not seen; return True if cap was hit."""
        key = value.lower()
        if key in seen:
            return False
        seen[key] = None
        # We store the original-cased value via a parallel structure below
        # by remembering insertion order; here `seen` is just the keyset.
        ordered_values.append(value)
        return len(seen) >= max_terms

    ordered_values: list[str] = []

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
            if _is_acronym(tok):
                keep = True
            elif _has_unit(tok):
                keep = True
            elif idx > 0 and _is_capitalized_word(tok):
                keep = True
            elif mode in ("formatted", "all") and _has_digit(tok):
                keep = True
            elif mode == "all" and len(tok) >= 2 and tok[0].isalpha():
                # Skip the sentence-initial word in "all" mode too — it is
                # almost always a stop-word ("Cette", "The"…) and biasing
                # toward it is useless.
                if idx > 0:
                    keep = True

            if keep and _try_add(tok):
                return ", ".join(ordered_values)

        # Numeric bigrams: "1 590", "1 231" — only in formatted/all.
        if mode in ("formatted", "all"):
            stripped = [_strip_edges(t) for t in tokens]
            for i in range(len(stripped) - 1):
                a, b = stripped[i], stripped[i + 1]
                if _is_all_digits(a) and _is_all_digits(b):
                    bigram = f"{a} {b}"
                    if _try_add(bigram):
                        return ", ".join(ordered_values)

    return ", ".join(ordered_values)


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
