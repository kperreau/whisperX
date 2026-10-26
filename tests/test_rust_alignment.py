"""Validate interpolation in the native alignment path."""

import numpy as np
import pytest

from whisperx_ext.alignment import align_segment


@pytest.mark.parametrize('changes', [
    {'blank_id': 4}, {'model_dictionary': {'a': 4}}, {'clean_cdx': [4]},
    {'clean_cdx': []}, {'sentence_spans': [(0, 4)]}, {'sentence_spans': [(1, 0)]},
    {'duration': 0}, {'t1': float('nan')},
])
def test_native_alignment_rejects_invalid_indices_without_panicking(changes):
    arguments = dict(emission=np.zeros((2, 2), np.float32), text='a', text_clean='a',
                     model_dictionary={'a': 1}, clean_cdx=[0], sentence_spans=[(0, 1)],
                     blank_id=0, t1=0., duration=1., no_spaces=False, return_char_alignments=False)
    arguments.update(changes)
    with pytest.raises(ValueError):
        align_segment(**arguments)


def test_native_empty_transcript_has_no_alignment():
    assert align_segment(np.zeros((2, 2), np.float32), '', '', {}, [], [], 0,
                         0., 1., False, False) is None


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
