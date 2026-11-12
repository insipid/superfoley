"""Shared helpers for media tests."""

import json
import subprocess
from pathlib import Path


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def stream(info, kind):
    return next((s for s in info["streams"] if s["codec_type"] == kind), None)


def stream_end(s):
    """start_time + duration of a stream, in seconds."""
    return float(s.get("start_time", 0)) + float(s["duration"])


def fmt_duration(path):
    return float(ffprobe(path)["format"]["duration"])


def decodes_cleanly(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                       capture_output=True, text=True)
    return r.returncode == 0 and not r.stderr.strip()


def has_encoder(name):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                         text=True).stdout
    return any(ln.split()[1:2] == [name] for ln in out.splitlines())


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def frame_pts(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=pts_time", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout
    return [float(x.strip(",")) for x in out.split()]
