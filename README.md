# superfoley

Cut a video down to just its non-speech, non-silent sounds. Point it at someone doing mechanical work while narrating, and it keeps the clanks, clicks, hisses and scrapes and drops the talking and the quiet: an ASMR supercut.

How it works: find the speech (Silero VAD) and the silence (an RMS threshold), subtract both from the timeline, and cut what's left with ffmpeg. No sound classification.

## Usage

With Docker. Videos go in `data/in/`, results come out in `data/out/` (the container always uses `/in` and `/out`):

```sh
docker compose build
docker compose run --rm superfoley              # every video in data/in/
docker compose run --rm superfoley video.mp4    # just data/in/video.mp4
```

- With no filename, every video in `data/in/` is processed, one after another. Videos that already have output are skipped; add `--force` to redo them. One failing video doesn't stop the rest.
- When all that is done, the supercuts are also joined into one `data/out/supersupercut.mp4` (with `supersupercut.txt` listing what's in it). It's rebuilt only when the set of supercuts changes, needs at least two supercuts, and `--no-combine` skips it.
- With a filename (looked up in `data/in/`), that video is always processed, and no combined file is built.
- `--out` and the other options still work; `--dry-run` writes the segment list only.

Without Docker (needs Python 3.11+ and ffmpeg; run from the repo directory):

```sh
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/superfoley path/to/video.mp4 --out out
```

Outside Docker the input folder is whatever `SUPERFOLEY_IN` points at (unset by default, so give a file path), and `SUPERFOLEY_OUT` sets the default `--out`.

Per video it writes, under `<out>/<video-name>/`:

- `<video-name>.supercut.mp4`: the kept sounds, joined, original audio and resolution (so `video.mp4` gives `video.supercut.mp4`)
- `segments.json` and `segments.csv`: the kept time ranges
- `clips/0001.mp4`, ...: each kept sound as its own clip

## Options

| Flag | Default | What it does |
|---|---|---|
| `--out` | `./out` (`/out` in Docker) | Output directory |
| `--force` | off | Folder mode: redo videos that already have output |
| `--no-combine` | off | Folder mode: don't build `supersupercut.mp4` |
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

Exit codes: 0 success (also when no sounds are found, with a warning), 1 failure (missing file, no audio track, ffmpeg error, or any video failing in folder mode), 2 bad options.

A video counts as done when its `.supercut.mp4` exists (or a finished run found nothing to keep). The combined file takes its frame size, frame rate and audio format from the first supercut (alphabetically); others are scaled to fit with black bars. It is re-encoded, so it plays in browsers (index first, no timestamp gaps), unlike a plain copy-join of separately encoded files.

Folder mode only looks at the top level of the input folder, ignores hidden files, and refuses two videos with the same name but different extensions, since they'd share an output folder.

## Notes

- Output keeps the input's resolution, so large inputs re-encode slowly. A 3-minute 1440x2560 clip took about 50 seconds on a 2-CPU machine. Use `--scale` for quick previews.
- Docker on a Mac runs without GPU; none is needed.
- Plan and design notes: [docs/PLAN.md](docs/PLAN.md).
