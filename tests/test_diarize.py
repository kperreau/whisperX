import numpy as np
import pandas as pd
import pytest

from whisperx.diarize import IntervalTree, assign_word_speakers


def test_overlap_query_handles_nested_and_unsorted_intervals():
    tree = IntervalTree([(4, 6, "B"), (0, 10, "A"), (2, 3, "C")])
    assert tree.query(5, 7) == [("A", 2), ("B", 1)]
    assert tree.query(10, 11) == []
    assert tree.query(3, 3) == []


def test_overlap_queries_match_brute_force():
    rng = np.random.default_rng(42)
    starts = rng.uniform(0, 100, 200)
    intervals = [(start, start + length, str(i))
                 for i, (start, length) in enumerate(zip(starts, rng.uniform(0, 20, 200)))]
    tree = IntervalTree(intervals)
    for start, length in zip(rng.uniform(0, 120, 100), rng.uniform(0.1, 10, 100)):
        end = start + length
        expected = [(speaker, min(stop, end) - max(begin, start))
                    for begin, stop, speaker in sorted(intervals)
                    if begin < end and stop > start]
        assert tree.query(start, end) == expected


@pytest.mark.parametrize("time, expected", [
    (-1, "long"), (99, "long"), (100.1, "long"), (102.9, "short"), (105, "short"),
])
def test_nearest_speaker_uses_interval_edges(time, expected):
    tree = IntervalTree([(0, 100, "long"), (103, 104, "short")])
    assert tree.find_nearest(time) == expected


def test_empty_interval_tree():
    tree = IntervalTree([])
    assert tree.query(0, 1) == []
    assert tree.find_nearest(1) is None


def test_fill_nearest_assigns_speaker_to_segments_and_words():
    diarization = pd.DataFrame([
        {"start": 0, "end": 100, "speaker": "long"},
        {"start": 103, "end": 104, "speaker": "short"},
    ])
    result = {"segments": [{"start": 100.1, "end": 100.2, "text": "hello",
                            "words": [{"word": "hello", "start": 100.1, "end": 100.2}]}]}
    assign_word_speakers(diarization, result, fill_nearest=True)
    assert result["segments"][0]["speaker"] == "long"
    assert result["segments"][0]["words"][0]["speaker"] == "long"
