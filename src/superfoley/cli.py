"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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


def run(args: argparse.Namespace) -> int:
    raise NotImplementedError("pipeline not implemented yet")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
