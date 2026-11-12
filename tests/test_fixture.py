import hashlib
import json
import subprocess

import numpy as np
import pytest

from tests.fixtures.make_fixture import FPS, SR, build_audio, load_truth_ranges


def _ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def test_audio_deterministic():
    a, _ = build_audio(0)
    b, _ = build_audio(0)
    assert hashlib.sha256(a.tobytes()).digest() == hashlib.sha256(b.tobytes()).digest()
    c, _ = build_audio(1)
    assert a.tobytes() != c.tobytes()


def test_truth_wellformed(fixture):
    truth = json.loads(fixture.truth_path.read_text())
    assert truth == fixture.truth
    segs = truth["segments"]
    assert {s["kind"] for s in segs} == {"speech", "sound", "silence"}
    prev_end = 0.0
    for s in segs:
        assert s["kind"] in {"speech", "sound", "silence"}
        assert s["start"] >= prev_end - 1e-9, "sorted and non-overlapping"
        assert s["end"] > s["start"]
        prev_end = s["end"]
    assert segs[0]["start"] >= 0
    assert prev_end <= truth["duration"] + 1e-9
    # speech never overlaps sound
    for a0, a1 in load_truth_ranges(truth, "speech"):
        for b0, b1 in load_truth_ranges(truth, "sound"):
            assert a1 <= b0 or b1 <= a0


def test_truth_matches_audio_levels(fixture):
    pcm, _ = build_audio(0)
    mono = pcm.astype(np.float64).mean(axis=1) / 32768
    for s in fixture.truth["segments"]:
        x = mono[int(s["start"] * SR):int(s["end"] * SR)]
        rms_db = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
        if s["kind"] == "silence":
            assert rms_db < -65
        else:
            assert rms_db > -40


def test_ffprobe(fixture):
    info = _ffprobe(fixture.video)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert v["codec_name"] == "h264"
    assert (v["width"], v["height"]) == (320, 240)
    assert v["r_frame_rate"] == v["avg_frame_rate"] == f"{FPS}/1"
    assert a["codec_name"] == "aac"
    assert a["channels"] == 2 and int(a["sample_rate"]) == SR
    dur = fixture.truth["duration"]
    assert abs(float(info["format"]["duration"]) - dur) < 1 / FPS + 0.03
    assert int(v["nb_frames"]) == round(dur * FPS)


# ------------------------------------------------ Silero-dependent checks


def _silero_probs(wav_pcm):
    from pysilero_vad import SileroVoiceActivityDetector

    b = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-",
         "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
        input=wav_pcm.tobytes(), capture_output=True, check=True).stdout
    vad = SileroVoiceActivityDetector()
    return np.array([vad.process_chunk(b[i:i + 1024]) for i in range(0, len(b) - 1023, 1024)])


@pytest.mark.silero
@pytest.mark.parametrize("seed", [0, 1])
def test_silero_flags_synthetic_speech_only(seed):
    pcm, truth = build_audio(seed)
    probs = _silero_probs(pcm)
    chunk = 512 / 16000
    for s in truth["segments"]:
        p = probs[int(s["start"] / chunk):int(s["end"] / chunk)]
        frac = float((p > 0.5).mean())
        if s["kind"] == "speech":
            assert frac > 0.6, (s, frac)
        else:
            assert frac < 0.3, (s, frac)
