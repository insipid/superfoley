import json

import pytest

from superfoley.cli import main
from superfoley.probe import probe
from superfoley.render import clip_command, clip_name
from tests.helpers import decodes_cleanly, ffprobe, fmt_duration, stream, stream_end

FRAME = 1 / 25


def _render(video, out, *extra):
    assert main([str(video), "--out", str(out), *extra]) == 0
    d = out / video.stem
    segs = json.loads((d / "segments.json").read_text())["segments"]
    return d, segs


@pytest.mark.silero
def test_supercut(fixture, tmp_path):
    d, segs = _render(fixture.video, tmp_path)
    clips = sorted((d / "clips").glob("*.mp4"))
    assert len(clips) == len(segs) > 0
    assert [c.name for c in clips] == [clip_name(i, len(segs)) for i in range(1, len(segs) + 1)]
    assert clips[0].name == "0001.mp4"
    assert all(decodes_cleanly(c) for c in clips)

    sc = d / "fixture.supercut.mp4"
    assert decodes_cleanly(sc)
    info = ffprobe(sc)
    v, a = stream(info, "video"), stream(info, "audio")
    assert v and a
    assert (v["width"], v["height"]) == (320, 240)
    assert v["r_frame_rate"] == v["avg_frame_rate"] == "25/1"
    assert a["codec_name"] == "aac" and a["channels"] == 2 and int(a["sample_rate"]) == 48000

    # Tolerance: supercut duration within ONE video frame (40 ms) in total of the sum of
    # the clip durations (stricter than one frame per clip); each clip's audio and
    # video ends agree within one frame.
    total = sum(fmt_duration(c) for c in clips)
    assert abs(fmt_duration(sc) - total) <= FRAME + 1e-3
    for c in clips:
        ci = ffprobe(c)
        assert abs(stream_end(stream(ci, "video")) - stream_end(stream(ci, "audio"))) <= FRAME
    # and the clips match the requested segments to within a frame
    for c, s in zip(clips, segs):
        assert abs(fmt_duration(c) - s["duration"]) <= FRAME + 0.022


@pytest.mark.silero
def test_scale_and_no_fade_zero(fixture, tmp_path):
    d, segs = _render(fixture.video, tmp_path, "--scale", "120", "--fade-ms", "0",
                      "--crf", "30")
    v = stream(ffprobe(d / "fixture.supercut.mp4"), "video")
    assert v["height"] == 120 and v["width"] == 160
    assert len(list((d / "clips").glob("*.mp4"))) == len(segs)


def test_clip_command_template():
    info = probe_fixture_like()
    cmd = clip_command("in.mp4", "out.mp4", 1.5, 2.0, info, crf=20, scale=None, fade_ms=10)
    assert cmd.index("-ss") < cmd.index("-i")
    assert "afade=t=in:st=0:d=0.0100,afade=t=out:st=1.9900:d=0.0100" in cmd
    assert "-vf" not in cmd
    short = clip_command("in.mp4", "out.mp4", 0, 0.04, info, fade_ms=30)
    assert "-af" not in short
    assert "-af" not in clip_command("in.mp4", "out.mp4", 0, 1, info, fade_ms=0)
    assert "scale=-2:720" in clip_command("in.mp4", "o.mp4", 0, 1, info, scale=720)


def probe_fixture_like():
    from fractions import Fraction

    from superfoley.probe import MediaInfo
    return MediaInfo(duration=10, fps=Fraction(25), width=320, height=240,
                     has_audio=True, sample_rate=48000, channels=2)
