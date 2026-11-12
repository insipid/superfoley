import numpy as np
import pytest

from superfoley.audio import extract_audio, load_wav
from superfoley.vad import speech_ranges
from tests.fixtures.make_fixture import load_truth_ranges, make_fixture


def overlap(a, b):
    """Total overlap in seconds between two lists of ranges."""
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def dur(r):
    return sum(e - s for s, e in r)


def test_silence_has_no_speech():
    assert speech_ranges(np.zeros(16000)) == []
    assert speech_ranges(np.zeros(0)) == []


def test_partial_chunk_ok():
    assert speech_ranges(np.zeros(1000)) == []


@pytest.mark.silero
@pytest.mark.parametrize("seed", [0, 1])
def test_fixture_speech(tmp_path, seed):
    fx = make_fixture(tmp_path, seed)
    x = load_wav(extract_audio(fx.video, tmp_path / "a.wav"))
    got = speech_ranges(x, 0.5)
    speech = load_truth_ranges(fx.truth, "speech")
    other = load_truth_ranges(fx.truth, "sound") + load_truth_ranges(fx.truth, "silence")
    assert overlap(got, speech) >= 0.9 * dur(speech)
    assert overlap(got, other) < 0.1 * dur(other)
    assert got == sorted(got)
