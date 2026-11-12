"""ffprobe helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class ProbeError(RuntimeError):
    """The input could not be read as a video."""


@dataclass
class MediaInfo:
    duration: float
    fps: Fraction
    width: int
    height: int
    has_audio: bool
    sample_rate: int = 48000
    channels: int = 2


def _fraction(s: str | None) -> Fraction:
    try:
        f = Fraction(s or "0")
    except (ValueError, ZeroDivisionError):
        return Fraction(0)
    return f if f > 0 else Fraction(0)


def probe(path: str | Path) -> MediaInfo:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
         str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ProbeError(f"cannot read {path} as a media file")
    info = json.loads(r.stdout)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise ProbeError(f"{path} has no video stream")
    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate")) or Fraction(30)
    dur = None
    for src in (info.get("format", {}), a or {}, v):
        try:
            dur = float(src["duration"])
            break
        except (KeyError, ValueError, TypeError):
            continue
    if not dur or dur <= 0:
        raise ProbeError(f"cannot determine the duration of {path}")
    return MediaInfo(
        duration=dur, fps=fps, width=int(v.get("width", 0)), height=int(v.get("height", 0)),
        has_audio=a is not None,
        sample_rate=int(a.get("sample_rate", 48000)) if a else 48000,
        channels=int(a.get("channels", 2)) if a else 2)
