"""Deterministic synthetic test video with known ground truth.

Importable:  ``make_fixture(out_dir, seed=0)`` -> ``Fixture``
CLI:         ``python tests/fixtures/make_fixture.py [out_dir] [--seed N]``
             (default out_dir: tests/fixtures/generated/)

Outputs in out_dir:
    fixture.mp4   320x240, 25 fps constant, h264 + aac stereo 48 kHz
    audio.wav     the exact 48 kHz stereo int16 audio that was muxed (pre-AAC)
    truth.json    {"duration", "sample_rate", "fps", "seed", "segments": [
                      {"start", "end", "kind"}, ...]}  kind in speech/sound/silence

Truth semantics:
  * segments are sorted, contiguous, non-overlapping and cover [0, duration];
  * "speech" never overlaps "sound";
  * a "sound" segment spans from its first to its last event; gaps between
    events inside it are < ~150 ms (they are not separate silence segments);
  * "silence" is digital zero or white noise at about -70 dBFS RMS.

Speech is synthetic (glottal pulse train through formant filters with ~4 Hz
syllabic modulation and pauses). Whether Silero VAD fires on it is checked by
tests marked ``silero``; ``load_truth_ranges`` lets tests use truth.json
instead of a detector.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 48_000
FPS = 25
WIDTH, HEIGHT = 320, 240
FRAME = SR // FPS  # 1920 samples; every segment is a whole number of frames

# (kind, variant, seconds). Multiples of 0.04 s. Speech never touches sound
# directly except through silence-free abutment, which is allowed (no overlap).
TIMELINE: list[tuple[str, str, float]] = [
    ("silence", "zero", 1.52),
    ("speech", "", 3.52),
    ("silence", "noise", 1.04),
    ("sound", "clicks", 3.04),
    ("speech", "", 2.52),
    ("sound", "rings", 3.52),
    ("silence", "noise", 1.52),
    ("sound", "scrape", 3.04),
    ("silence", "noise", 1.04),
    ("speech", "", 2.04),
    ("sound", "hum", 4.04),
    ("silence", "noise", 1.52),
]

# ---------------------------------------------------------------- synthesis

_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
           (530, 1840, 2480), (660, 1720, 2410), (570, 840, 2410)]


def _formant_gain(n: int, formants: list[float]) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    h = np.ones_like(f)
    for fc, bw in zip(list(formants) + [3800.0], (80, 100, 140, 200)):
        h /= np.sqrt((1 - (f / fc) ** 2) ** 2 + ((f / fc) * (bw / fc)) ** 2)
    return h


def _syllable(dur: float, rng: np.random.Generator, f_base: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = f_base * (1 + 0.05 * np.sin(2 * np.pi * 5 * t)
                   + 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6))
                   + rng.uniform(-0.1, 0.1) * t / dur)
    frac = (np.cumsum(f0) / SR) % 1.0
    flow = np.where(frac < 0.6, np.sin(np.pi * frac / 0.6) ** 2, 0.0)
    x = np.diff(flow, prepend=0.0) * SR / 200 + 0.03 * rng.standard_normal(n)
    spec = np.fft.rfft(x)
    # formant glide: filter with two vowels and crossfade across the syllable
    ys = []
    for _ in range(2):
        vowel = _VOWELS[rng.integers(len(_VOWELS))]
        formants = [v * j for v, j in zip(vowel, rng.uniform(0.93, 1.07, 3))]
        ys.append(np.fft.irfft(spec * _formant_gain(n, formants), n))
    mix = np.linspace(0, 1, n)
    y = ys[0] * (1 - mix) + ys[1] * mix
    return y * np.hanning(n) ** 0.7 * rng.uniform(0.5, 1.0)


def _consonant(rng: np.random.Generator) -> np.ndarray:
    """Short band-limited noise burst (fricative / plosive-like)."""
    n = int(rng.uniform(0.03, 0.08) * SR)
    c = _bandpassed_noise(n, rng, rng.uniform(2500, 4500), rng.uniform(6000, 9000))
    return c * np.hanning(n) * rng.uniform(0.15, 0.35)


def synth_speech(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n_samples)
    pos = 0
    f_base = rng.uniform(110, 130)
    while pos < n_samples:
        if pos > 0 and rng.random() < 0.08:  # pause
            pos += int(rng.uniform(0.12, 0.3) * SR)
            f_base = rng.uniform(110, 130)  # new phrase, new pitch
            continue
        if rng.random() < 0.5:
            _add(out, _consonant(rng), pos)
            pos += int(0.03 * SR)
        s = _syllable(rng.uniform(0.17, 0.30), rng, f_base * rng.uniform(0.9, 1.12))
        _add(out, s, pos)
        pos += int(len(s) * rng.uniform(0.85, 1.0))  # ~4 syllables/s
    out = out / (np.abs(out).max() + 1e-9) * 0.5
    return out + rng.standard_normal(n_samples) * 10 ** (-55 / 20)  # room tone


def _decay_sine(freq: float, t60: float, n: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t + phase) * 10 ** (-3 * t / t60)


def _add(dst: np.ndarray, src: np.ndarray, at: int) -> None:
    end = min(len(dst), at + len(src))
    if end > at:
        dst[at:end] += src[: end - at]


def synth_clicks(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    pos = 0
    while pos < n - int(0.02 * SR):
        burst = rng.standard_normal(int(0.002 * SR)) * np.hanning(int(0.002 * SR))
        ring = _decay_sine(rng.uniform(1800, 4200), 0.25, int(0.2 * SR))
        _add(out, ring * 0.5, pos)
        _add(out, burst * 0.9, pos)
        pos += int(rng.uniform(0.07, 0.14) * SR)
    return out * rng.uniform(0.3, 0.5)


def synth_rings(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    pos = 0
    ratios = [1.0, 2.76, 5.40, 8.93]
    while pos < n - int(0.02 * SR):
        base = rng.uniform(500, 1400)
        for i, r in enumerate(ratios):
            if base * r < 20000:
                _add(out, _decay_sine(base * r, 0.9 / (1 + i), int(1.0 * SR)) / (1 + i), pos)
        _add(out, rng.standard_normal(int(0.003 * SR)) * 0.5, pos)
        pos += int(rng.uniform(0.32, 0.45) * SR)  # tails overlap: no silent gaps
    return out / (np.abs(out).max() + 1e-9) * 0.6


def _bandpassed_noise(n: int, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(spec, n)
    return y / (y.std() + 1e-9)


def synth_scrape(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    env = 0.55 + 0.25 * np.sin(2 * np.pi * 2.3 * t + rng.uniform(0, 6)) \
        + 0.15 * np.sin(2 * np.pi * 7.1 * t + rng.uniform(0, 6))
    noise = _bandpassed_noise(n, rng, 1000, 7000)
    stutter = 1 + 0.5 * (rng.random(n // 480 + 1).repeat(480)[:n] - 0.5)
    return noise * env * stutter * 0.12


def synth_hum(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    tone = sum(np.sin(2 * np.pi * 100 * k * t + k) / k for k in range(1, 6)) * 0.08
    wobble = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
    return (tone + _bandpassed_noise(n, rng, 60, 2500) * 0.05) * wobble


_SOUNDS = {"clicks": synth_clicks, "rings": synth_rings,
           "scrape": synth_scrape, "hum": synth_hum}


def _edge_fade(x: np.ndarray, ms: float = 5.0) -> np.ndarray:
    k = min(int(ms / 1000 * SR), len(x) // 2)
    w = np.linspace(0, 1, k)
    x = x.copy()
    x[:k] *= w
    x[-k:] *= w[::-1]
    return x


# ------------------------------------------------------------------ builder


def build_audio(seed: int = 0) -> tuple[np.ndarray, dict]:
    """Return (int16 array shaped (n, 2), truth dict). Pure function of seed."""
    rng = np.random.default_rng(seed)
    chans: list[np.ndarray] = []
    segments = []
    t = 0
    for kind, variant, secs in TIMELINE:
        n = round(secs * SR)
        assert n % FRAME == 0
        if kind == "speech":
            mono = _edge_fade(synth_speech(n, rng)) * 0.9
        elif kind == "sound":
            mono = _edge_fade(_SOUNDS[variant](n, rng), 3.0)
        elif variant == "zero":
            mono = np.zeros(n)
        else:  # ~ -70 dBFS RMS white noise
            mono = rng.standard_normal(n) * 10 ** (-70 / 20)
        pan = rng.uniform(0.8, 1.0, 2) if kind != "silence" else np.ones(2)
        chans.append(np.stack([mono * pan[0], mono * pan[1]], axis=1))
        segments.append({"start": t / SR, "end": (t + n) / SR, "kind": kind})
        t += n
    audio = np.concatenate(chans)
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    truth = {"duration": t / SR, "sample_rate": SR, "fps": FPS, "seed": seed,
             "segments": segments}
    return pcm, truth


def write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


@dataclass
class Fixture:
    video: Path
    wav: Path
    truth_path: Path
    truth: dict


def make_fixture(out_dir: str | Path, seed: int = 0) -> Fixture:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcm, truth = build_audio(seed)
    wav, video, truth_path = out / "audio.wav", out / "fixture.mp4", out / "truth.json"
    write_wav(wav, pcm)
    truth_path.write_text(json.dumps(truth, indent=2))
    dur = truth["duration"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}",
         "-i", str(wav),
         "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", str(FPS),
         "-pix_fmt", "yuv420p", "-r", str(FPS), "-fps_mode", "cfr",
         "-c:a", "aac", "-b:a", "128k", "-ar", str(SR), "-ac", "2",
         "-movflags", "+faststart", str(video)],
        check=True)
    return Fixture(video, wav, truth_path, truth)


def load_truth_ranges(truth: dict | Path | str, kind: str) -> list[tuple[float, float]]:
    """(start, end) pairs for one kind. Lets tests bypass Silero for speech."""
    if not isinstance(truth, dict):
        truth = json.loads(Path(truth).read_text())
    return [(s["start"], s["end"]) for s in truth["segments"] if s["kind"] == kind]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", nargs="?",
                    default=str(Path(__file__).parent / "generated"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    fx = make_fixture(a.out_dir, a.seed)
    print(f"wrote {fx.video}, {fx.wav}, {fx.truth_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
