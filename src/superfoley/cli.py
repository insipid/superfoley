"""Command line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from .audio import NoAudioError, extract_audio, load_wav, silence_ranges
from .probe import ProbeError, probe
from .segments import compute_segments, segments_to_csv, segments_to_json
from .vad import speech_ranges


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="superfoley",
        description="Cut a video down to only its non-speech, non-silent sounds.",
    )
    p.add_argument("input", type=Path, help="input video file")
    p.add_argument("--out", type=Path, default=Path("./out"),
                   help="output directory (default: ./out)")
    p.add_argument("--silence-db", type=float, default=-45.0,
                   help="frames quieter than this RMS level (dBFS) are silence (default: -45)")
    p.add_argument("--speech-pad-ms", type=int, default=150,
                   help="widen detected speech by this much on each side (default: 150)")
    p.add_argument("--min-gap-ms", type=int, default=200,
                   help="merge kept segments separated by less than this (default: 200)")
    p.add_argument("--min-dur-ms", type=int, default=150,
                   help="drop kept segments shorter than this (default: 150)")
    p.add_argument("--pad-ms", type=int, default=100,
                   help="pad each kept segment by this much on each side (default: 100)")
    p.add_argument("--fade-ms", type=int, default=10,
                   help="audio fade in/out per clip (default: 10)")
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="Silero speech probability threshold (default: 0.5)")
    p.add_argument("--crf", type=int, default=20,
                   help="H.264 CRF for rendered clips (default: 20)")
    p.add_argument("--scale", type=int, default=None, metavar="HEIGHT",
                   help="output height in pixels (default: keep source resolution)")
    p.add_argument("--dry-run", action="store_true",
                   help="write the segment list only, do not render video")
    return p


class CliError(Exception):
    """A real failure: print one line and exit 1."""


def _check_args(p: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    for name in ("speech_pad_ms", "min_gap_ms", "min_dur_ms", "pad_ms", "fade_ms"):
        if getattr(a, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0")
    if not 0 <= a.vad_threshold <= 1:
        p.error("--vad-threshold must be between 0 and 1")
    if not 0 <= a.crf <= 51:
        p.error("--crf must be between 0 and 51")
    if a.scale is not None and a.scale <= 0:
        p.error("--scale must be a positive height")


def _fail_text(e: BaseException) -> str:
    if isinstance(e, subprocess.CalledProcessError):
        return f"{Path(str(e.cmd[0])).name} failed (exit {e.returncode})"
    if isinstance(e, OSError) and e.strerror:
        return f"{e.strerror}: {e.filename}" if e.filename else e.strerror
    return str(e)


def _pipeline(args: argparse.Namespace) -> int:
    src = args.input
    if not src.exists():
        raise CliError(f"input file not found: {src}")
    if not src.is_file():
        raise CliError(f"input is not a file: {src}")
    info = probe(src)
    if not info.has_audio:
        raise CliError(f"{src} has no audio track")

    out_dir = args.out / src.stem
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CliError(f"cannot create output directory {out_dir}: {e.strerror or e}")

    with tempfile.TemporaryDirectory(prefix="superfoley-") as td:
        wav = extract_audio(src, Path(td) / "audio.wav")
        x = load_wav(wav)
    duration = info.duration
    speech = speech_ranges(x, threshold=args.vad_threshold)
    silence = silence_ranges(x, args.silence_db)
    segments = compute_segments(
        duration, speech, silence,
        speech_pad=args.speech_pad_ms / 1000, min_gap=args.min_gap_ms / 1000,
        min_dur=args.min_dur_ms / 1000, pad=args.pad_ms / 1000)

    params = {"silence_db": args.silence_db, "speech_pad_ms": args.speech_pad_ms,
              "min_gap_ms": args.min_gap_ms, "min_dur_ms": args.min_dur_ms,
              "pad_ms": args.pad_ms, "fade_ms": args.fade_ms,
              "vad_threshold": args.vad_threshold, "crf": args.crf,
              "scale": args.scale, "dry_run": args.dry_run}
    try:
        segments_to_json(out_dir / "segments.json", segments, duration, params)
        segments_to_csv(out_dir / "segments.csv", segments)
    except OSError as e:
        raise CliError(f"cannot write to {out_dir}: {e.strerror or e}")

    kept = sum(e - s for s, e in segments)
    print(f"{len(segments)} segments, {kept:.1f}s kept of {duration:.1f}s "
          f"({100 * kept / duration:.1f}%)", file=sys.stderr)
    if not segments:
        print("warning: no non-speech, non-silent segments found; nothing to render",
              file=sys.stderr)
    elif not args.dry_run:
        pass  # rendering arrives in T6
    print(out_dir)
    return 0


def run(args: argparse.Namespace) -> int:
    try:
        return _pipeline(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
    except NoAudioError as e:
        print(f"error: {e}", file=sys.stderr)
    except ProbeError as e:
        print(f"error: {e}", file=sys.stderr)
    except FileNotFoundError as e:
        print("error: ffmpeg/ffprobe not found on PATH" if e.filename in ("ffmpeg", "ffprobe")
              else f"error: {_fail_text(e)}", file=sys.stderr)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"error: {_fail_text(e)}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_args(parser, args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
