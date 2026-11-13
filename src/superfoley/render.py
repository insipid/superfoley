"""Cut clips (re-encode H.264/AAC) and concatenate them into a supercut."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from .probe import MediaInfo, probe

Range = tuple[float, float]

COMBINED_NAME = "supersupercut.mp4"


class RenderError(RuntimeError):
    """ffmpeg failed while rendering."""


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
        raise RenderError(f"ffmpeg failed: {lines[-1] if lines else 'exit %d' % r.returncode}")


def clip_name(i: int, n: int) -> str:
    return f"{i:0{max(4, len(str(n)))}d}.mp4"


def clip_command(src: str | Path, dst: str | Path, start: float, dur: float,
                 info: MediaInfo, *, crf: int = 20, scale: int | None = None,
                 fade_ms: int = 10) -> list[str]:
    """ffmpeg command line for one clip.

    The duration is snapped to a whole number of source frames so the video
    stream is exactly that long; audio is cut to the same length.
    """
    fps = info.fps
    nframes = max(1, round(dur * float(fps)))
    clip_dur = nframes / float(fps)
    vf = []
    if scale:
        vf.append(f"scale=-2:{scale}")
    fade = fade_ms / 1000
    af = []
    if fade_ms > 0 and clip_dur >= 2 * fade:
        af = [f"afade=t=in:st=0:d={fade:.4f}",
              f"afade=t=out:st={clip_dur - fade:.4f}:d={fade:.4f}"]
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.6f}", "-i", str(src),
           "-t", f"{clip_dur:.6f}", "-map", "0:v:0"]
    if info.has_audio:
        cmd += ["-map", "0:a:0"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-r", f"{fps.numerator}/{fps.denominator}",
            "-fps_mode", "cfr", "-frames:v", str(nframes)]
    if info.has_audio:
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += ["-c:a", "aac", "-ar", str(info.sample_rate), "-ac", str(info.channels)]
    cmd += ["-shortest", "-movflags", "+faststart", str(dst)]
    return cmd


def supercut_name(src: str | Path) -> str:
    """``video.mp4`` -> ``video.supercut.mp4``."""
    return f"{Path(src).stem}.supercut.mp4"


def _concat_quote(p: Path) -> str:
    return "'" + str(p).replace("'", "'\\''") + "'"


def concat_clips(clips: Sequence[Path], dst: str | Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file {_concat_quote(Path(c).resolve())}\n" for c in clips),
                       encoding="utf-8")
        dst = Path(dst)
        # Write to a temporary name and rename on success, so an interrupted run never
        # leaves a half-written supercut that looks finished.
        part = dst.with_name(dst.name + ".part")
        try:
            _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                  "-c", "copy", "-movflags", "+faststart", "-f", "mp4", str(part)])
            os.replace(part, dst)
        finally:
            part.unlink(missing_ok=True)


def render(src: str | Path, segments: Sequence[Range], out_dir: str | Path, info: MediaInfo,
           *, crf: int = 20, scale: int | None = None, fade_ms: int = 10,
           progress: Callable[[str], None] | None = None) -> Path:
    """Write ``clips/NNNN.mp4`` and ``<stem>.supercut.mp4`` under ``out_dir``; return the supercut."""
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.mp4"):
        old.unlink()
    n = len(segments)
    if progress is None:
        def progress(msg: str) -> None:
            print(msg, file=sys.stderr, flush=True)
    clips = []
    for i, (s, e) in enumerate(segments, 1):
        progress(f"clip {i}/{n}")
        dst = clips_dir / clip_name(i, n)
        _run(clip_command(src, dst, s, e - s, info, crf=crf, scale=scale, fade_ms=fade_ms))
        clips.append(dst)
    progress("concatenating")
    out = out_dir / supercut_name(src)
    concat_clips(clips, out)
    return out


def _video_duration(path: str | Path) -> float:
    """Duration of the first video stream (the container duration can include longer audio)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.split()[0])
    except (IndexError, ValueError):
        return probe(path).duration


def normalize_command(src: str | Path, dst: str | Path, target: MediaInfo, *,
                      video_duration: float, crf: int = 20) -> list[str]:
    """Re-encode one supercut to ``target``'s size, frame rate and audio format (MPEG-TS).

    Inputs may differ; scaling is to fit with padding. MPEG-TS is used for the temporary
    parts because it has no index or edit lists, so the parts join with continuous
    timestamps. The part is cut to a whole number of output frames with audio padded or
    trimmed to match, so audio and video end together and joins leave no gap.
    """
    w, h, fps = target.width, target.height, target.fps
    nframes = max(1, round(video_duration * fps))
    length = nframes / fps
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps.numerator}/{fps.denominator},"
          f"tpad=stop_mode=clone:stop_duration=1,format=yuv420p")
    ch = "mono" if target.channels == 1 else "stereo"
    af = (f"aresample={target.sample_rate}:async=1:first_pts=0,"
          f"aformat=channel_layouts={ch},apad")
    return ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0",
            "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-frames:v", str(nframes),
            "-af", af, "-c:a", "aac", "-ar", str(target.sample_rate),
            "-t", f"{float(length):.6f}", "-f", "mpegts", str(dst)]


def combine(supercuts: Sequence[Path], dst: str | Path, *, crf: int = 20,
            progress: Callable[[str], None] | None = None) -> Path:
    """Join several supercuts, in order, into one browser-friendly file.

    Output is H.264/AAC at the first supercut's frame size, constant frame rate and audio
    format, with the index at the front. Each input is normalised on its own (one ffmpeg at
    a time, so memory stays small however many there are) and the parts are then joined
    without re-encoding.
    """
    dst = Path(dst)
    supercuts = [Path(p) for p in supercuts]
    n = len(supercuts)
    if progress is None:
        def progress(msg: str) -> None:
            print(msg, file=sys.stderr, flush=True)
    target = probe(supercuts[0])
    part = dst.with_name(dst.name + ".part")
    with tempfile.TemporaryDirectory(prefix=".combine-", dir=dst.parent) as td:
        parts = []
        for i, src in enumerate(supercuts, 1):
            progress(f"combining {i}/{n}")
            ts = Path(td) / f"{i:05d}.ts"
            _run(normalize_command(src, ts, target, video_duration=_video_duration(src),
                                   crf=crf))
            parts.append(ts)
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file {_concat_quote(t.resolve())}\n" for t in parts),
                       encoding="utf-8")
        try:
            _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                  "-c", "copy", "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart",
                  "-f", "mp4", str(part)])
            os.replace(part, dst)
        finally:
            part.unlink(missing_ok=True)
    return dst
