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


def test_usage_errors_exit_2():
    for argv in (["x.mp4", "--crf", "99"], ["x.mp4", "--pad-ms", "-1"],
                 ["x.mp4", "--scale", "0"], ["x.mp4", "--bogus"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2


# ------------------------------------------------------------ end to end

from superfoley.segments import segments_to_json  # noqa: E402,F401
from tests.fixtures.make_fixture import load_truth_ranges  # noqa: E402

import csv  # noqa: E402
import json  # noqa: E402

TOL = 0.150


@pytest.mark.silero
def test_end_to_end_segments(fixture, tmp_path, capsys):
    out = tmp_path / "out"
    assert main([str(fixture.video), "--out", str(out), "--dry-run"]) == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == str(out / "fixture")
    assert "4 segments" in cap.err and "%" in cap.err

    d = out / "fixture"
    doc = json.loads((d / "segments.json").read_text())
    segs = [(s["start"], s["end"]) for s in doc["segments"]]
    D = fixture.truth["duration"]
    assert abs(doc["duration"] - D) < 0.05
    assert doc["params"]["silence_db"] == -45 and doc["params"]["pad_ms"] == 100
    assert segs == sorted(segs)
    assert all(0 <= s < e <= doc["duration"] + 1e-6 for s, e in segs)
    assert all(a[1] <= b[0] for a, b in zip(segs, segs[1:]))
    rows = list(csv.reader((d / "segments.csv").open()))
    assert rows[0] == ["index", "start", "end", "duration"] and len(rows) == len(segs) + 1

    # same criteria as the segments end-to-end test: one segment per truth sound,
    # edges bordering silence/boundary within +-150 ms
    truth = fixture.truth["segments"]
    sounds = [i for i, t in enumerate(truth) if t["kind"] == "sound"]
    assert len(segs) == len(sounds)
    for i in sounds:
        t = truth[i]
        (hit,) = [g for g in segs if g[0] < t["end"] and g[1] > t["start"]]
        if i == 0 or truth[i - 1]["kind"] != "speech":
            assert abs(hit[0] - t["start"]) <= TOL
        if i + 1 == len(truth) or truth[i + 1]["kind"] != "speech":
            assert abs(hit[1] - t["end"]) <= TOL


@pytest.mark.silero
def test_dry_run_writes_no_video(fixture, tmp_path):
    out = tmp_path / "o"
    assert main([str(fixture.video), "--out", str(out), "--dry-run"]) == 0
    files = sorted(p.name for p in (out / "fixture").rglob("*") if p.is_file())
    assert files == ["segments.csv", "segments.json"]
