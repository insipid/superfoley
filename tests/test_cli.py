from pathlib import Path

import pytest

from superfoley.cli import build_parser, main


def test_defaults():
    a = build_parser().parse_args(["in.mp4"])
    assert a.input == Path("in.mp4")
    assert a.out == Path("./out")
    assert a.silence_db == -45
    assert a.speech_pad_ms == 150
    assert a.min_gap_ms == 200
    assert a.min_dur_ms == 150
    assert a.pad_ms == 100
    assert a.fade_ms == 10
    assert a.vad_threshold == 0.5
    assert a.crf == 20
    assert a.scale is None
    assert a.dry_run is False


def test_overrides():
    a = build_parser().parse_args([
        "/in/v.mp4", "--out", "/out", "--silence-db", "-50.5",
        "--speech-pad-ms", "1", "--min-gap-ms", "2", "--min-dur-ms", "3",
        "--pad-ms", "4", "--fade-ms", "5", "--vad-threshold", "0.7",
        "--crf", "28", "--scale", "720", "--dry-run",
    ])
    assert a.out == Path("/out")
    assert a.silence_db == -50.5
    assert (a.speech_pad_ms, a.min_gap_ms, a.min_dur_ms, a.pad_ms, a.fade_ms) == (1, 2, 3, 4, 5)
    assert a.vad_threshold == 0.7
    assert a.crf == 28
    assert a.scale == 720
    assert a.dry_run is True


def test_input_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_not_implemented():
    with pytest.raises(NotImplementedError):
        main(["in.mp4"])
