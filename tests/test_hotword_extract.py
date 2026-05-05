"""Tests for whisperx.hotword_extract."""

from whisperx.hotword_extract import extract_hotwords, merge_hotwords


def test_french_real_estate_text_keeps_proper_nouns_and_units():
    text = (
        "Cette villa de luxe se situe au coeur du domaine du Golf de Biscarrosse. "
        "La propriété est affichée au prix de 1 197 000 €. "
        "Elle offre une surface habitable de 170 m² sur une parcelle de 1112 m²."
    )
    result = extract_hotwords(text)
    parts = [p.strip() for p in result.split(",")]
    assert "Golf" in parts
    assert "Biscarrosse" in parts
    assert "m²" in parts
    assert "€" in parts
    # First-word-of-sentence noise must be filtered out.
    assert "Cette" not in parts
    assert "La" not in parts
    assert "Elle" not in parts


def test_english_text_keeps_acronyms_and_proper_nouns():
    text = (
        "WhisperX uses PyAnnote and a GPU. "
        "Performance was 95% accurate at 14:00 UTC."
    )
    result = extract_hotwords(text)
    parts = [p.strip() for p in result.split(",")]
    assert "PyAnnote" in parts
    assert "GPU" in parts
    assert "UTC" in parts
    assert "95%" in parts
    assert "WhisperX" not in parts  # first word of sentence


def test_max_terms_caps_output():
    text = ". ".join(f"Sentence start {i} ProperNoun{i}" for i in range(50))
    result = extract_hotwords(text, max_terms=5)
    parts = [p for p in result.split(", ") if p]
    assert len(parts) == 5
    assert parts == [f"ProperNoun{i}" for i in range(5)]


def test_empty_text_returns_empty_string():
    assert extract_hotwords("") == ""
    assert extract_hotwords("   ") == ""


def test_merge_hotwords_dedupes_case_insensitively():
    assert merge_hotwords("Foo, Bar", "bar, baz") == "Foo, Bar, baz"


def test_merge_hotwords_returns_none_when_all_empty():
    assert merge_hotwords(None, "", None) is None


def test_merge_hotwords_preserves_first_occurrence_order():
    assert merge_hotwords("A, B", "C, A, D") == "A, B, C, D"


def test_top_level_reexports():
    """Server callers `import whisperx` and expect the helpers there."""
    import whisperx
    assert whisperx.extract_hotwords("Hello WhisperX from Paris.") == "WhisperX, Paris"
    assert whisperx.merge_hotwords("a, b", "B, c") == "a, b, c"
