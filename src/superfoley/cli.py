"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .audio import NoAudioError, extract_audio, load_wav, silence_ranges
from .probe import ProbeError, probe
from .render import COMBINED_NAME, RenderError, combine, render
from .segments import compute_segments, segments_to_csv, segments_to_json
from .vad import speech_ranges


VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ts",
              ".flv", ".wmv", ".3gp"}


def _env_dir(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="superfoley",
        description="Cut a video down to only its non-speech, non-silent sounds.",
        epilog="Folders: SUPERFOLEY_IN sets the input folder (the Docker image sets /in). "
               "A relative INPUT is looked up there, and with no INPUT every video in it "
               "is processed, skipping any that already have output. SUPERFOLEY_OUT sets "
               "the default --out (the Docker image sets /out). Folder mode also joins all the "
               "supercuts into one supersupercut.mp4 (use --no-combine to skip).",
    )
    p.add_argument("input", type=Path, nargs="?", default=None,
                   help="input video file; if omitted, process every video in the input folder")
    p.add_argument("--out", type=Path, default=_env_dir("SUPERFOLEY_OUT") or Path("./out"),
                   help="output directory (default: $SUPERFOLEY_OUT or ./out)")
    p.add_argument("--force", action="store_true",
                   help="when processing a whole folder, redo videos that already have output")
    p.add_argument("--no-combine", action="store_true",
                   help=f"when processing a whole folder, don't also build {COMBINED_NAME}")
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
    if a.input is None and _env_dir("SUPERFOLEY_IN") is None:
        p.error("no input file given (and SUPERFOLEY_IN is not set, so there is no folder to scan)")
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


def _pipeline(args: argparse.Namespace, src: Path) -> int:
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
        render(src, segments, out_dir, info, crf=args.crf, scale=args.scale,
               fade_ms=args.fade_ms)
    print(out_dir)
    return 0


def _is_done(out_dir: Path, stem: str, dry_run: bool) -> bool:
    """Has this video already been processed in a way that satisfies this run?"""
    seg = out_dir / "segments.json"
    if dry_run:
        return seg.exists()
    if (out_dir / f"{stem}.supercut.mp4").exists() or (out_dir / "supercut.mp4").exists():
        return True
    # A finished run that found nothing to keep never writes a supercut.
    try:
        data = json.loads(seg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("segments") == [] and not data.get("params", {}).get("dry_run", True)


def _resolve_input(arg: Path, in_dir: Path | None) -> Path:
    """Absolute paths are used as given; relative ones are looked up in the input folder."""
    if arg.is_absolute() or in_dir is None:
        return arg
    return in_dir / arg


def _guarded(args: argparse.Namespace, src: Path) -> int:
    try:
        return _pipeline(args, src)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
    except NoAudioError as e:
        print(f"error: {e}", file=sys.stderr)
    except (ProbeError, RenderError) as e:
        print(f"error: {e}", file=sys.stderr)
    except FileNotFoundError as e:
        print("error: ffmpeg/ffprobe not found on PATH" if e.filename in ("ffmpeg", "ffprobe")
              else f"error: {_fail_text(e)}", file=sys.stderr)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"error: {_fail_text(e)}", file=sys.stderr)
    return 1


def _supercut_for(out: Path, stem: str) -> Path | None:
    for name in (f"{stem}.supercut.mp4", "supercut.mp4"):   # second: pre-rename output
        if (out / stem / name).exists():
            return out / stem / name
    return None


def _combine_outputs(args: argparse.Namespace, files: list[Path]) -> bool:
    """Build ``supersupercut.mp4`` from every supercut of the folder; False on failure."""
    inputs = [p for f in files if (p := _supercut_for(args.out, f.stem))]
    if len(inputs) < 2:
        print(f"not building {COMBINED_NAME}: needs at least two supercuts "
              f"(found {len(inputs)})", file=sys.stderr)
        return True
    dst = args.out / COMBINED_NAME
    listing = dst.with_suffix(".txt")
    names = [str(p.relative_to(args.out)) for p in inputs]
    try:
        fresh = (dst.exists() and listing.exists()
                 and listing.read_text(encoding="utf-8").splitlines() == names
                 and all(p.stat().st_mtime <= dst.stat().st_mtime for p in inputs))
    except OSError:
        fresh = False
    if fresh:
        print(f"{COMBINED_NAME} is up to date", file=sys.stderr)
        return True
    print(f"building {COMBINED_NAME} from {len(inputs)} supercuts", file=sys.stderr)
    try:
        combine(inputs, dst, crf=args.crf)
        listing.write_text("\n".join(names) + "\n", encoding="utf-8")
    except (CliError, ProbeError, RenderError, OSError, subprocess.CalledProcessError) as e:
        print(f"error: could not build {COMBINED_NAME}: {_fail_text(e)}", file=sys.stderr)
        return False
    print(dst)
    return True


def _batch(args: argparse.Namespace, in_dir: Path) -> int:
    try:
        files = sorted(f for f in in_dir.iterdir()
                       if f.is_file() and not f.name.startswith(".")
                       and f.suffix.lower() in VIDEO_EXTS)
    except OSError as e:
        print(f"error: cannot read input folder {in_dir}: {e.strerror or e}", file=sys.stderr)
        return 1
    if not files:
        print(f"warning: no video files found in {in_dir}", file=sys.stderr)
        return 0
    by_stem: dict[str, list[str]] = {}
    for f in files:
        by_stem.setdefault(f.stem, []).append(f.name)
    clashes = [names for names in by_stem.values() if len(names) > 1]
    if clashes:
        for names in clashes:
            print(f"error: {', '.join(names)} would share one output folder; rename one",
                  file=sys.stderr)
        return 1
    done = failed = 0
    for i, f in enumerate(files, 1):
        if not args.force and _is_done(args.out / f.stem, f.stem, args.dry_run):
            print(f"[{i}/{len(files)}] skip {f.name} (already done)", file=sys.stderr)
            continue
        print(f"[{i}/{len(files)}] {f.name}", file=sys.stderr)
        if _guarded(args, f) == 0:
            done += 1
        else:
            failed += 1
    skipped = len(files) - done - failed
    print(f"{done} processed, {skipped} skipped, {failed} failed", file=sys.stderr)
    combined_ok = True
    if not (args.dry_run or args.no_combine):
        combined_ok = _combine_outputs(args, files)
    return 1 if failed or not combined_ok else 0


def run(args: argparse.Namespace) -> int:
    in_dir = _env_dir("SUPERFOLEY_IN")
    if args.input is None:
        return _batch(args, in_dir)
    return _guarded(args, _resolve_input(args.input, in_dir))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_args(parser, args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
