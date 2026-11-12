"""Audio extraction and RMS-based silence detection."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

Range = tuple[float, float]
FLOOR_DB = -120.0


class NoAudioError(RuntimeError):
    """The input video has no audio track."""


def has_audio(video: str | Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True).stdout
    return bool(out.strip())


def extract_audio(video: str | Path, wav_path: str | Path) -> Path:
    """Decode the video's audio to 16 kHz mono 16-bit PCM WAV."""
    if not has_audio(video):
        raise NoAudioError(f"{video} has no audio track")
    wav_path = Path(wav_path)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
        check=True)
    return wav_path


def load_wav(path: str | Path) -> np.ndarray:
    """Read a 16-bit PCM WAV (mono, or first channel) as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch)[:, 0].copy() if ch > 1 else x


def rms_db_frames(samples: np.ndarray, sr: int = 16000, frame_ms: float = 20) -> np.ndarray:
    """dBFS RMS per non-overlapping frame (last partial frame included), floor -120."""
    n = max(1, int(round(sr * frame_ms / 1000)))
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0)
    nframes = -(-x.size // n)
    padded = np.zeros(nframes * n)
    padded[: x.size] = x
    counts = np.full(nframes, n, dtype=np.float64)
    counts[-1] = x.size - (nframes - 1) * n
    ms = (padded.reshape(nframes, n) ** 2).sum(axis=1) / counts
    return np.maximum(10 * np.log10(np.maximum(ms, 1e-30)), FLOOR_DB)


def silence_ranges(samples: np.ndarray, threshold_db: float, sr: int = 16000,
                   frame_ms: float = 20) -> list[Range]:
    """Merged (start, end) second ranges where frame RMS < threshold_db."""
    db = rms_db_frames(samples, sr, frame_ms)
    n = max(1, int(round(sr * frame_ms / 1000)))
    total = len(samples) / sr
    mask = np.concatenate([[False], db < threshold_db, [False]])
    d = np.diff(mask.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(float(s * n / sr), float(min(e * n / sr, total))) for s, e in zip(starts, ends)]
