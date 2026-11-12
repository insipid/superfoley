# superfoley

Cut a video down to just its non-speech, non-silent sounds. Point it at someone doing mechanical work while narrating, and it keeps the clanks, clicks, hisses and scrapes and drops the talking and the quiet: an ASMR supercut.

How it works: find the speech (Silero VAD) and the silence (an RMS threshold), subtract both from the timeline, and cut what's left with ffmpeg. No sound classification.

## Usage

With Docker (put videos in `data/in/`):

```sh
docker compose build
docker compose run --rm superfoley /in/video.mp4 --out /out
```

Results appear in `data/out/video/`.

Without Docker (needs Python 3.11+ and ffmpeg):

```sh
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/superfoley path/to/video.mp4 --out out
```

Per video it writes, under `<out>/<video-name>/`:

- `supercut.mp4`: the kept sounds, joined, original audio and resolution
- `segments.json` and `segments.csv`: the kept time ranges
- `clips/0001.mp4`, ...: each kept sound as its own clip

## Options

| Flag | Default | What it does |
|---|---|---|
| `--silence-db` | -45 | Frames quieter than this (dBFS) count as silence |
| `--speech-pad-ms` | 150 | Widen detected speech by this much each side |
| `--min-gap-ms` | 200 | Join kept sounds closer together than this |
| `--min-dur-ms` | 150 | Drop kept sounds shorter than this |
| `--pad-ms` | 100 | Pad each kept sound each side |
| `--fade-ms` | 10 | Audio fade in/out per clip |
| `--vad-threshold` | 0.5 | Speech probability cut-off (0 to 1) |
| `--crf` | 20 | H.264 quality (lower is better) |
| `--scale` | source | Output height in pixels |
| `--dry-run` | off | Write the segment list only, no video |

`--dry-run` takes seconds even for long clips, so use it to tune before rendering. See [docs/TUNING.md](docs/TUNING.md).

Exit codes: 0 success (also when no sounds are found, with a warning), 1 failure (missing file, no audio track, ffmpeg error), 2 bad options.

## Notes

- Output keeps the input's resolution, so large inputs re-encode slowly. A 3-minute 1440x2560 clip took about 50 seconds on a 2-CPU machine. Use `--scale` for quick previews.
- Docker on a Mac runs without GPU; none is needed.
- Plan and design notes: [docs/PLAN.md](docs/PLAN.md).
