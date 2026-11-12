# superfoley

Cut a video down to just its non-speech, non-silent sounds. Point it at someone doing mechanical work while narrating, and it keeps the clanks, clicks and scrapes and drops the talking and the quiet: an ASMR supercut.

How it works: find the speech (Silero VAD) and the silence (an RMS threshold), subtract both from the timeline, and cut what's left with ffmpeg. No sound classification.

> **Status:** planned, not built yet. See [docs/PLAN.md](docs/PLAN.md).

## Usage (planned)

```sh
docker compose run --rm superfoley /in/video.mp4 --out /out
```

Per video it writes `supercut.mp4`, `segments.json`, `segments.csv` and individual clips.

Useful flags: `--silence-db`, `--speech-pad-ms`, `--min-gap-ms`, `--min-dur-ms`, `--pad-ms`, `--fade-ms`, `--vad-threshold`, `--crf`, `--dry-run` (write the segment list only).

## Requirements

Docker. Built for an arm64 Mac, with no GPU needed.
