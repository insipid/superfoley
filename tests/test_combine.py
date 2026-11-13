"""supersupercut.mp4: all supercuts of a folder joined into one browser-friendly file."""

import os
import shutil
import struct
import subprocess

import pytest

from superfoley.cli import main
from superfoley.render import COMBINED_NAME
from tests.helpers import decodes_cleanly, ff, ffprobe, fmt_duration, stream


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    monkeypatch.setenv("SUPERFOLEY_IN", str(in_dir))
    monkeypatch.setenv("SUPERFOLEY_OUT", str(out_dir))
    return in_dir, out_dir


@pytest.fixture(scope="module")
def mixed(fixture, tmp_path_factory):
    """Three inputs that differ in size, frame rate and audio format."""
    d = tmp_path_factory.mktemp("mixed")
    shutil.copy(fixture.video, d / "a_landscape.mp4")                      # 320x240 @25, stereo 48k
    ff("-i", fixture.video, "-vf", "scale=240:320,fps=30", "-c:v", "libx264", "-preset",
       "veryfast", "-c:a", "copy", d / "b_portrait.mp4")                    # 240x320 @30
    ff("-i", fixture.video, "-vf", "scale=160:120", "-c:v", "libx264", "-preset", "veryfast",
       "-ac", "1", "-ar", "44100", "-c:a", "aac", d / "c_small_mono.mp4")  # 160x120, mono 44.1k
    return d


def _atoms(path):
    out, size = [], os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos < size:
            f.seek(pos)
            s, t = struct.unpack(">I4s", f.read(8))
            if s == 1:
                s = struct.unpack(">Q", f.read(8))[0]
            out.append(t.decode())
            pos += s
    return out


def _frame_sizes(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "frame=width,height", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return {tuple(ln.strip(",").split(",")) for ln in r.stdout.split()}


def _dts_gaps(path, limit=0.2):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "packet=dts_time", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    t = [float(x) for x in r.stdout.split()]
    return [b - a for a, b in zip(t, t[1:]) if b - a > limit or b <= a]


@pytest.mark.silero
def test_combined_file_is_clean_and_complete(dirs, mixed, capsys):
    in_dir, out_dir = dirs
    for f in mixed.iterdir():
        shutil.copy(f, in_dir / f.name)
    assert main([]) == 0
    comb = out_dir / COMBINED_NAME
    assert comb.exists() and decodes_cleanly(comb)
    assert not list(out_dir.rglob("*.part"))

    supercuts = [out_dir / s / f"{s}.supercut.mp4"
                 for s in ("a_landscape", "b_portrait", "c_small_mono")]
    assert all(p.exists() for p in supercuts)

    info = ffprobe(comb)
    v, a = stream(info, "video"), stream(info, "audio")
    assert (v["width"], v["height"]) == (320, 240)             # first supercut's frame
    assert v["r_frame_rate"] == "25/1"                         # constant, first's rate
    num, den = map(int, v["avg_frame_rate"].split("/"))
    assert abs(num / den - 25) < 0.5
    assert a["codec_name"] == "aac" and int(a["sample_rate"]) == 48000
    assert _frame_sizes(comb) == {("320", "240")}              # no size changes inside

    total = sum(fmt_duration(p) for p in supercuts)
    assert abs(fmt_duration(comb) - total) <= 0.3 + 0.05 * len(supercuts)
    assert abs(float(v["duration"]) - float(a["duration"])) <= 0.15
    assert _dts_gaps(comb) == []

    atoms = _atoms(comb)
    assert atoms.index("moov") < atoms.index("mdat")           # index first: web friendly

    listing = (out_dir / "supersupercut.txt").read_text().splitlines()
    assert listing == [f"{s}/{s}.supercut.mp4" for s in ("a_landscape", "b_portrait",
                                                         "c_small_mono")]


@pytest.mark.silero
def test_combined_is_rebuilt_only_when_needed(dirs, fixture, capsys):
    in_dir, out_dir = dirs
    for n in ("a.mp4", "b.mp4"):
        shutil.copy(fixture.video, in_dir / n)
    assert main(["--crf", "35"]) == 0
    comb = out_dir / COMBINED_NAME
    first = comb.stat().st_mtime_ns
    capsys.readouterr()

    assert main(["--crf", "35"]) == 0
    assert "up to date" in capsys.readouterr().err
    assert comb.stat().st_mtime_ns == first

    new = out_dir / "a" / "a.supercut.mp4"                      # a supercut changed
    os.utime(new, (comb.stat().st_atime + 5, comb.stat().st_mtime + 5))
    assert main(["--crf", "35"]) == 0
    assert comb.stat().st_mtime_ns != first

    shutil.copy(fixture.video, in_dir / "c.mp4")                # a new video joins the list
    assert main(["--crf", "35"]) == 0
    assert "c/c.supercut.mp4" in (out_dir / "supersupercut.txt").read_text()


@pytest.mark.silero
def test_no_combine_dry_run_and_single_video(dirs, fixture, capsys):
    in_dir, out_dir = dirs
    for n in ("a.mp4", "b.mp4"):
        shutil.copy(fixture.video, in_dir / n)
    assert main(["--dry-run"]) == 0
    assert not (out_dir / COMBINED_NAME).exists()
    assert main(["--no-combine", "--crf", "35"]) == 0
    assert not (out_dir / COMBINED_NAME).exists()

    (in_dir / "b.mp4").unlink()
    shutil.rmtree(out_dir)
    capsys.readouterr()
    assert main(["--crf", "35"]) == 0
    assert "needs at least two" in capsys.readouterr().err
    assert not (out_dir / COMBINED_NAME).exists()


@pytest.mark.silero
def test_named_file_never_builds_combined(dirs, fixture):
    in_dir, out_dir = dirs
    for n in ("a.mp4", "b.mp4"):
        shutil.copy(fixture.video, in_dir / n)
    assert main(["a.mp4", "--crf", "35"]) == 0
    assert not (out_dir / COMBINED_NAME).exists()
