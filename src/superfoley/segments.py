"""Pure interval arithmetic and the segment combinator.

A range is a ``(start, end)`` tuple of float seconds. All range-list functions
return sorted, non-overlapping, non-empty ranges.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

Range = tuple[float, float]

EPS = 1e-9


def merge(ranges: Iterable[Range]) -> list[Range]:
    """Sort, drop empty/inverted ranges, and merge overlapping or touching ones."""
    out: list[Range] = []
    for s, e in sorted((float(s), float(e)) for s, e in ranges if e > s):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def dilate(ranges: Iterable[Range], pad: float, lo: float = -math.inf,
           hi: float = math.inf) -> list[Range]:
    """Grow each range by ``pad`` on both sides, clamp to [lo, hi], merge."""
    return merge((max(s - pad, lo), min(e + pad, hi)) for s, e in merge(ranges))


def pad_and_clamp(ranges: Iterable[Range], pad: float, lo: float, hi: float) -> list[Range]:
    """Pad each range by ``pad`` per side, clamp to [lo, hi], re-merge overlaps."""
    return dilate(ranges, pad, lo, hi)


def complement(ranges: Iterable[Range], lo: float, hi: float) -> list[Range]:
    """The parts of [lo, hi] not covered by ``ranges``."""
    out: list[Range] = []
    pos = lo
    for s, e in merge(ranges):
        if e <= lo:
            continue
        if s >= hi:
            break
        if s > pos:
            out.append((pos, s))
        pos = max(pos, e)
    if pos < hi:
        out.append((pos, hi))
    return out


def subtract(a: Iterable[Range], b: Iterable[Range]) -> list[Range]:
    """Parts of ``a`` not covered by ``b``."""
    out: list[Range] = []
    bs = merge(b)
    for s, e in merge(a):
        pos = s
        for bs_, be in bs:
            if be <= pos:
                continue
            if bs_ >= e:
                break
            if bs_ > pos:
                out.append((pos, bs_))
            pos = max(pos, be)
        if pos < e:
            out.append((pos, e))
    return out


def merge_gaps(ranges: Iterable[Range], min_gap: float) -> list[Range]:
    """Merge neighbours separated by a gap shorter than ``min_gap``."""
    out: list[Range] = []
    for s, e in merge(ranges):
        if out and s - out[-1][1] < min_gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def drop_short(ranges: Iterable[Range], min_dur: float) -> list[Range]:
    """Drop ranges shorter than ``min_dur``."""
    return [(s, e) for s, e in merge(ranges) if e - s >= min_dur - EPS]


def compute_segments(duration: float, speech: Sequence[Range], silence: Sequence[Range], *,
                     speech_pad: float, min_gap: float, min_dur: float,
                     pad: float) -> list[Range]:
    """Non-speech, non-silent segments of a clip of ``duration`` seconds.

    Order of operations (all values in seconds):
      1. keep = [0, duration] - dilate(speech, speech_pad) - silence
      2. merge_gaps(keep, min_gap): join segments separated by < min_gap
         (this deliberately keeps up to min_gap of speech/silence between them)
      3. drop_short(.., min_dur): remove segments shorter than min_dur
      4. pad_and_clamp(.., pad, 0, duration): grow each by pad, clamp, re-merge.
    Step 4 runs after speech dilation, so padding can bring back up to ``pad``
    seconds of dilated speech/silence at each segment edge. That is intended.
    """
    if duration <= 0:
        return []
    keep = subtract([(0.0, duration)],
                    dilate(speech, speech_pad, 0.0, duration) + list(silence))
    keep = merge_gaps(keep, min_gap)
    keep = drop_short(keep, min_dur)
    return pad_and_clamp(keep, pad, 0.0, duration)


def _rows(segments: Sequence[Range]) -> list[dict]:
    return [{"index": i, "start": round(s, 3), "end": round(e, 3),
             "duration": round(e - s, 3)} for i, (s, e) in enumerate(segments, 1)]


def segments_to_json(path: str | Path, segments: Sequence[Range], duration: float,
                     params: dict) -> None:
    doc = {"duration": round(duration, 3), "params": params, "segments": _rows(segments)}
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def segments_to_csv(path: str | Path, segments: Sequence[Range]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["index", "start", "end", "duration"])
        for r in _rows(segments):
            w.writerow([r["index"], f"{r['start']:.3f}", f"{r['end']:.3f}",
                        f"{r['duration']:.3f}"])
