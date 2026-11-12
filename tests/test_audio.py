import numpy as np
import pytest

from superfoley.audio import (NoAudioError, extract_audio, load_wav, rms_db_frames,
                              silence_ranges)
from tests.fixtures.make_fixture import load_truth_ranges

SR = 16000
FRAME = 0.02


def sine(db, secs, freq=1000.0):
    t = np.arange(int(secs * SR)) / SR
    return (10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_digital_zero_hits_floor():
    db = rms_db_frames(np.zeros(SR), SR)
    assert len(db) == 50 and np.all(db == -120.0)


def test_sine_level():
    db = rms_db_frames(sine(-20, 1.0), SR)
    assert np.allclose(db, -20, atol=0.1)


def test_partial_frame_and_empty():
    assert len(rms_db_frames(np.zeros(SR + 100), SR)) == 51
    assert len(rms_db_frames(np.zeros(0), SR)) == 0


def test_silence_ranges_mixed():
    x = np.concatenate([np.zeros(SR), sine(-20, 1.0), np.zeros(SR // 2)])
    assert silence_ranges(x, -45, SR) == [(0.0, 1.0), (2.0, 2.5)]
    assert silence_ranges(sine(-20, 1.0), -45, SR) == []
    assert silence_ranges(np.zeros(SR), -45, SR) == [(0.0, 1.0)]
    assert silence_ranges(np.zeros(0), -45, SR) == []


def test_silence_threshold_respected():
    x = np.concatenate([sine(-50, 1.0), sine(-30, 1.0)])
    assert silence_ranges(x, -45, SR) == [(0.0, 1.0)]
    assert silence_ranges(x, -60, SR) == []


def test_fixture_silence_matches_truth(fixture, tmp_path):
    wav = extract_audio(fixture.video, tmp_path / "a.wav")
    x = load_wav(wav)
    assert x.dtype == np.float32 and np.abs(x).max() <= 1.0
    got = silence_ranges(x, -45, SR)
    truth = load_truth_ranges(fixture.truth, "silence")
    tol = FRAME + 1e-6
    # each truth silence segment is covered by one detected range, edges within a frame
    for ts, te in truth:
        hit = [g for g in got if g[0] < te and g[1] > ts and g[1] - g[0] >= 0.5 * (te - ts)]
        assert hit, (ts, te)
        gs, ge = hit[0]
        assert abs(gs - ts) <= tol or ts == 0.0 and gs == 0.0
        assert abs(ge - te) <= tol or te >= fixture.truth["duration"] - tol
    # Silence must not cover speech, except for quiet pauses/tails inside the
    # synthetic speech (room tone at -55 dBFS is below -45 dB): each overlap is
    # a short pause and the total stays a small share of speech time.
    speech = load_truth_ranges(fixture.truth, "speech")
    covered = 0.0
    for ss, se in speech:
        for gs, ge in got:
            ov = max(0.0, min(se, ge) - max(ss, gs))
            assert ov <= 0.5
            covered += ov
    assert covered < 0.15 * sum(e - s for s, e in speech)


def test_no_audio_raises(tmp_path):
    import subprocess
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x64:rate=10:duration=1", "-c:v", "libx264",
                    str(v)], check=True)
    with pytest.raises(NoAudioError):
        extract_audio(v, tmp_path / "o.wav")
