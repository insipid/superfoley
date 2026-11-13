"""T7 edge cases, one test per case."""

import json
import shutil
import wave

import numpy as np
import pytest

from superfoley.cli import main
from tests.fixtures.make_fixture import SR
from tests.helpers import decodes_cleanly, ff, ffprobe, frame_pts, fmt_duration, stream, stream_end

FRAME = 1 / 25


def _segments(out, stem):
    return json.loads((out / stem / "segments.json").read_text())


def _no_supercut(out, stem):
    assert not list((out / stem).glob("*supercut.mp4"))
    assert not list((out / stem).glob("clips/*"))


def _silent_video(path, audio):
    """3-5 s testsrc video with a WAV audio track."""
    ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25", "-i", audio, "-shortest",
       "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", path)


def _wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.round(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def test_no_audio_track(small_clips, tmp_path, capsys):
    assert main([str(small_clips["noaudio"]), "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert "no audio track" in err and "Traceback" not in err and len(err.strip().splitlines()) == 1


@pytest.mark.silero
def test_variable_frame_rate(small_clips, tmp_path):
    src = small_clips["vfr"]
    assert len({round(d, 4) for d in np.diff(frame_pts(src))}) > 1, "test clip is not VFR"
    out = tmp_path / "o"
    assert main([str(src), "--out", str(out)]) == 0
    sc = out / "vfr" / "vfr.supercut.mp4"
    assert decodes_cleanly(sc)
    info = ffprobe(sc)
    v, a = stream(info, "video"), stream(info, "audio")
    assert v and a
    assert v["r_frame_rate"] == v["avg_frame_rate"]  # constant frame rate
    pts = frame_pts(sc)
    deltas = np.diff(pts)
    assert deltas.max() - deltas.min() < 1e-3
    n = len(_segments(out, "vfr")["segments"])
    assert n >= 1
    # audio and video end together within one frame (+ the 1-frame start offset)
    assert abs(stream_end(v) - stream_end(a)) <= (1 + n) * FRAME
    kept = sum(s["duration"] for s in _segments(out, "vfr")["segments"])
    assert abs(fmt_duration(sc) - kept) <= n * FRAME + 0.05


@pytest.mark.silero
def test_av1_input(small_clips, tmp_path):
    src = small_clips["av1"]
    if src is None:
        pytest.skip("no AV1 encoder (libsvtav1/libaom-av1) in this ffmpeg")
    assert stream(ffprobe(src), "video")["codec_name"] == "av1"
    out = tmp_path / "o"
    assert main([str(src), "--out", str(out)]) == 0
    sc = out / "av1" / "av1.supercut.mp4"
    info = ffprobe(sc)
    assert stream(info, "video")["codec_name"] == "h264" and stream(info, "audio")
    assert decodes_cleanly(sc)


def test_all_silence_no_segments(tmp_path, capsys):
    _wav(tmp_path / "z.wav", np.zeros(SR * 3))
    vid = tmp_path / "silent.mp4"
    _silent_video(vid, tmp_path / "z.wav")
    out = tmp_path / "o"
    assert main([str(vid), "--out", str(out)]) == 0
    doc = _segments(out, "silent")
    assert doc["segments"] == []
    assert (out / "silent" / "segments.csv").read_text().strip() == "index,start,end,duration"
    _no_supercut(out, "silent")
    assert "warning" in capsys.readouterr().err


def test_all_segments_too_short(tmp_path, capsys):
    # four 50 ms noise bursts in digital silence; with --pad-ms 0 each is below the
    # default 150 ms --min-dur-ms and is dropped
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 5)
    for t in (0.5, 1.5, 2.5, 3.5):
        i = int(t * SR)
        x[i:i + SR // 20] = rng.standard_normal(SR // 20) * 0.3
    _wav(tmp_path / "b.wav", x)
    vid = tmp_path / "bursts.mp4"
    _silent_video(vid, tmp_path / "b.wav")
    out = tmp_path / "o"
    assert main([str(vid), "--out", str(out), "--pad-ms", "0"]) == 0
    assert _segments(out, "bursts")["segments"] == []
    _no_supercut(out, "bursts")
    assert "warning" in capsys.readouterr().err
    # sanity: the bursts are real sounds that survive with a smaller --min-dur-ms
    assert main([str(vid), "--out", str(out), "--pad-ms", "0", "--min-dur-ms", "20",
                 "--dry-run"]) == 0
    assert len(_segments(out, "bursts")["segments"]) == 4


@pytest.mark.silero
def test_odd_filenames(fixture, tmp_path):
    d = tmp_path / "Ünï cödé 'dir' & more"
    d.mkdir()
    vid = d / "It's a \"test\" clip – ñ 日本 [1] %s.mp4"
    shutil.copy(fixture.video, vid)
    out = tmp_path / "out dir's ünï"
    assert main([str(vid), "--out", str(out)]) == 0
    sc = out / vid.stem / f"{vid.stem}.supercut.mp4"
    assert sc.exists() and decodes_cleanly(sc)
    assert stream(ffprobe(sc), "audio") and stream(ffprobe(sc), "video")


def test_missing_input(tmp_path, capsys):
    assert main([str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert "not found" in err and "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("kind", ["file", "under_file", "unwritable"])
def test_bad_output_dir(small_clips, fixture, tmp_path, capsys, kind):
    if kind == "file":
        out = tmp_path / "out"
        out.write_text("x")
    elif kind == "under_file":
        (tmp_path / "f").write_text("x")
        out = tmp_path / "f" / "out"
    else:
        import os
        if os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        out = tmp_path / "ro"
        out.mkdir()
        out.chmod(0o500)
    try:
        assert main([str(fixture.video), "--out", str(out), "--dry-run"]) == 1
    finally:
        if out.is_dir():
            out.chmod(0o700)
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
