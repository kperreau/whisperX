"""Validate interpolation in the native alignment path."""

import numpy as np
import pytest

from whisperx_ext.alignment import align_segment


@pytest.mark.parametrize("interpolate_words", [True, False])
def test_native_alignment_can_preserve_missing_word_timestamps(interpolate_words):
    emission = np.full((20, 4), -5, dtype=np.float32)
    emission[:, 0] = -1
    for token, start, end in [(1, 2, 5), (2, 8, 10), (3, 14, 17)]:
        emission[start:end, token] = 0
        emission[start:end, 0] = -5
    result = align_segment(
        emission, "a ? b", "a|b", {"a": 1, "|": 2, "b": 3},
        [0, 1, 4], [(0, 5)], 0, 0, 5, False, False,
        interpolate_words=interpolate_words,
    )
    assert result is not None
    words = {word.word: word for word in result[0].words}
    assert np.isfinite(words["a"].start) and np.isfinite(words["b"].end)
    assert bool(np.isnan(words["?"].start)) is (not interpolate_words)
    assert bool(np.isnan(words["?"].end)) is (not interpolate_words)
    assert np.isfinite(result[0].start) and np.isfinite(result[0].end)
