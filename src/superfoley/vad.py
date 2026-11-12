"""Speech detection with Silero VAD (pysilero-vad, no PyTorch)."""

from __future__ import annotations

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

Range = tuple[float, float]


def speech_probs(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Per-chunk speech probability (512-sample chunks; last one zero-padded)."""
    if sr != 16000:
        raise ValueError("Silero VAD requires 16 kHz audio")
    n = SileroVoiceActivityDetector.chunk_samples()
    pcm = np.round(np.clip(np.asarray(samples, dtype=np.float64), -1, 1) * 32767).astype("<i2")
    pad = (-len(pcm)) % n
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, dtype="<i2")])
    vad = SileroVoiceActivityDetector()
    vad.reset()
    return np.array([vad.process_chunk(pcm[i:i + n].tobytes())
                     for i in range(0, len(pcm), n)])


def speech_ranges(samples: np.ndarray, threshold: float = 0.5, sr: int = 16000) -> list[Range]:
    """Merged (start, end) second ranges where chunk probability >= threshold.

    Chunks are 32 ms. Hysteresis: speech starts at probability >= threshold and
    continues until it drops below threshold - 0.15 (floored at 0.01).
    """
    probs = speech_probs(samples, sr)
    chunk = SileroVoiceActivityDetector.chunk_samples() / sr
    total = len(samples) / sr
    low = max(threshold - 0.15, 0.01)
    ranges: list[Range] = []
    start = None
    for i, p in enumerate(probs):
        if start is None and p >= threshold:
            start = i
        elif start is not None and p < low:
            ranges.append((start * chunk, i * chunk))
            start = None
    if start is not None:
        ranges.append((start * chunk, len(probs) * chunk))
    return [(s, min(e, total)) for s, e in ranges]
