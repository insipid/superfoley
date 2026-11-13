"""Input/output folders, batch mode, skip-if-done and supercut naming."""

import json
import shutil

import pytest

from superfoley.cli import _is_done, main
from tests.helpers import decodes_cleanly


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    monkeypatch.setenv("SUPERFOLEY_IN", str(in_dir))
    monkeypatch.setenv("SUPERFOLEY_OUT", str(out_dir))
    return in_dir, out_dir


def _add(in_dir, fixture, name):
    shutil.copy(fixture.video, in_dir / name)


def test_out_defaults_from_env(dirs):
    from superfoley.cli import build_parser
    assert build_parser().parse_args([]).out == dirs[1]


@pytest.mark.silero
def test_relative_input_is_looked_up_in_input_folder(dirs, fixture):
    in_dir, out_dir = dirs
    _add(in_dir, fixture, "clip.mp4")
    assert main(["clip.mp4", "--dry-run"]) == 0
    assert (out_dir / "clip" / "segments.json").exists()


@pytest.mark.silero
def test_absolute_input_is_used_as_given(dirs, fixture, tmp_path):
    _, out_dir = dirs
    elsewhere = tmp_path / "elsewhere.mp4"
    shutil.copy(fixture.video, elsewhere)
    assert main([str(elsewhere), "--dry-run"]) == 0
    assert (out_dir / "elsewhere" / "segments.json").exists()


def test_missing_relative_input_reports_input_folder_path(dirs, capsys):
    assert main(["nope.mp4"]) == 1
    assert str(dirs[0] / "nope.mp4") in capsys.readouterr().err


@pytest.mark.silero
def test_batch_processes_videos_only_and_skips_done(dirs, fixture, capsys):
    in_dir, out_dir = dirs
    _add(in_dir, fixture, "a.mp4")
    _add(in_dir, fixture, "b.MP4")
    (in_dir / "notes.txt").write_text("not a video")
    _add(in_dir, fixture, ".hidden.mp4")

    assert main(["--dry-run"]) == 0
    assert "2 processed, 0 skipped, 0 failed" in capsys.readouterr().err
    assert sorted(p.name for p in out_dir.iterdir()) == ["a", "b"]

    seg = out_dir / "a" / "segments.json"
    seg.write_text(seg.read_text() + " ")          # marker: changes if rewritten
    before = seg.read_text()
    assert main(["--dry-run"]) == 0
    assert "0 processed, 2 skipped, 0 failed" in capsys.readouterr().err
    assert seg.read_text() == before

    assert main(["--dry-run", "--force"]) == 0
    assert "2 processed, 0 skipped, 0 failed" in capsys.readouterr().err
    assert seg.read_text() != before


@pytest.mark.silero
def test_explicit_file_always_reruns(dirs, fixture, capsys):
    in_dir, out_dir = dirs
    _add(in_dir, fixture, "a.mp4")
    assert main(["a.mp4", "--dry-run"]) == 0
    seg = out_dir / "a" / "segments.json"
    seg.write_text(seg.read_text() + " ")
    before = seg.read_text()
    assert main(["a.mp4", "--dry-run"]) == 0
    assert seg.read_text() != before


@pytest.mark.silero
def test_supercut_name_and_dry_run_then_real_run(dirs, fixture, capsys):
    in_dir, out_dir = dirs
    _add(in_dir, fixture, "video.mp4")
    assert main(["--dry-run"]) == 0                 # dry-run output must not count as done
    assert main([]) == 0
    sc = out_dir / "video" / "video.supercut.mp4"
    assert sc.exists() and decodes_cleanly(sc)
    assert not (out_dir / "video" / "supercut.mp4").exists()
    assert not list(out_dir.rglob("*.part"))
    capsys.readouterr()
    assert main([]) == 0
    assert "0 processed, 1 skipped, 0 failed" in capsys.readouterr().err


def test_empty_input_folder(dirs, capsys):
    assert main([]) == 0
    assert "no video files" in capsys.readouterr().err


def test_stem_clash_is_an_error(dirs, fixture, capsys):
    in_dir, out_dir = dirs
    _add(in_dir, fixture, "same.mp4")
    _add(in_dir, fixture, "same.mkv")
    assert main(["--dry-run"]) == 1
    assert "same.mkv, same.mp4" in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.silero
def test_one_failure_does_not_stop_the_batch(dirs, fixture, small_clips, capsys):
    in_dir, out_dir = dirs
    shutil.copy(small_clips["noaudio"], in_dir / "a_noaudio.mp4")
    _add(in_dir, fixture, "b_good.mp4")
    assert main(["--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "no audio track" in err
    assert "1 processed, 0 skipped, 1 failed" in err
    assert (out_dir / "b_good" / "segments.json").exists()


def test_is_done_rules(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    assert not _is_done(d, "x", dry_run=False) and not _is_done(d, "x", dry_run=True)

    def seg(segments, dry):
        (d / "segments.json").write_text(json.dumps(
            {"segments": segments, "params": {"dry_run": dry}}))

    seg([{"index": 1}], dry=False)                  # crashed mid-render: no supercut yet
    assert not _is_done(d, "x", dry_run=False) and _is_done(d, "x", dry_run=True)
    seg([], dry=False)                              # finished, nothing to keep
    assert _is_done(d, "x", dry_run=False)
    seg([], dry=True)                               # only a dry run saw nothing
    assert not _is_done(d, "x", dry_run=False)
    (d / "x.supercut.mp4").write_bytes(b"x")
    assert _is_done(d, "x", dry_run=False)
    (d / "x.supercut.mp4").unlink()
    (d / "supercut.mp4").write_bytes(b"x")          # output from before the rename
    assert _is_done(d, "x", dry_run=False)
