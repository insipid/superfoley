# superfoley: Plan (v3)

Cut a video down to only its non-speech, non-silent sounds (an ASMR supercut of someone doing mechanical work), as a Dockerized CLI.

## Locked decisions

- **Name:** superfoley (repo `insipid/superfoley`, Python package `superfoley`, CLI `superfoley`, compose service `superfoley`).
- **Core method:** non-speech sound = everything - speech - silence. No sound classification or labelling.
- **Inputs:** one video at a time, a few minutes long. Speech and work sounds rarely overlap; very little background noise.
- **Outputs per video:** `supercut.mp4`, `segments.json`, `segments.csv`, individual clip files. No EDL / editor timeline.
- **Audio:** original audio only, no normalisation.
- **Invocation:** `docker compose run --rm superfoley /in/video.mp4 --out /out [flags]`
- **Target:** M4 Mac (arm64) only. Cloud runner is out of scope for now.
- **Stack:** Python 3.12 (Docker image) / 3.11+ locally, ffmpeg, `pysilero-vad` (Silero VAD, bundled model, no PyTorch; has aarch64 wheels), numpy, pytest.

## Reference sample

First real test clip (kept out of git, in `samples/`): vertical 1440x2560 AV1 video at 60 fps, Opus stereo 48 kHz, about 3 minutes, 152 MB. Consequences:

- Decoding AV1 must work in the container's ffmpeg.
- The output keeps the input's resolution by default (whatever file goes in is what comes out). Re-encoding to H.264 is slow at this size, so `--scale` (output height) and `--crf` are optional knobs, off by default.
- Audio is stereo 48 kHz; analysis downmixes to 16 kHz mono, but clips keep the original audio untouched.

## CLI flags (all defaults overridable)

`--silence-db -45` · `--speech-pad-ms 150` · `--min-gap-ms 200` · `--min-dur-ms 150` · `--pad-ms 100` · `--fade-ms 10` · `--vad-threshold 0.5` · `--crf 20` · `--scale` (optional output height; default: keep source resolution) · `--dry-run` (write segment list only, no rendering)

## Commit and PR conventions

- Plain commit messages only: **no** `Co-Authored-By`, session-link or "generated with" trailers, so contribution stats stay clean.
- Commits are authored as the repo owner (the GitHub account's email), never as Claude.
- One commit per task, on a `build` branch; final PR to `main`.
- No attribution lines in PR descriptions either.

## How the build runs

- A fresh subagent builds each task in order: implement, run its tests, commit.
- I review each diff and re-run the tests before moving on.
- A failing task gets at most two fix attempts. After that I stop and report instead of burning budget.
- Tests run natively in the cloud workspace (Python venv + ffmpeg). The cloud workspace probably can't run Docker, so the Dockerfile and compose file are verified on the Mac: `docker compose build` and one `docker compose run`, with the result reported back.
- At the end, a separate agent that hasn't seen the build follows only the README to run the tool.
- Budget: the plan's usage limits are the cap. Pay-as-you-go "extra usage" stays off.

## Tasks

| # | Task | Deliverable | Test / acceptance |
|---|------|-------------|-------------------|
| T0 | Scaffold | `pyproject.toml`, `src/superfoley/` layout, empty CLI, `Dockerfile`, `docker-compose.yml`, pytest config, `.gitignore` (incl. `samples/`, `out/`) | `pytest` runs (0 tests OK); `python -m superfoley --help` works |
| T1 | Test fixtures | Script that generates a synthetic video with **known ground truth**: speech, mechanical sounds (clicks, scrapes, noise bursts), silence, fixed frame rate; writes `truth.json` | Deterministic; truth ranges don't overlap |
| T2 | Audio extraction + silence mask | `audio.py`: ffmpeg to 16 kHz mono WAV; RMS dBFS frame mask | Unit tests on synthetic arrays; mask matches fixture silence within one frame |
| T3 | Speech detection | `vad.py`: `pysilero-vad` wrapper (512-sample chunks at 16 kHz) returning speech ranges | On fixture, speech ranges overlap truth by at least 90% |
| T4 | Interval math | `segments.py`: complement, dilate, merge gaps, drop short, pad, clamp | Property tests: sorted, non-overlapping, in range, flags respected; edge cases (empty, all speech) |
| T5 | CLI + segment output | Wire T2-T4, write `segments.json` / `.csv`, `--dry-run` | End-to-end on fixture: detected segments match truth within ±150 ms |
| T6 | Render | `render.py`: cut clips (re-encode H.264/AAC), concat with `-c copy`, audio fades | ffprobe: supercut duration = sum of segments (within one frame); has video and audio |
| T7 | Edge cases | AV1 / variable-frame-rate input normalised; no audio track; no sounds found; very short clips; odd filenames | One test per case; clear error messages |
| T8 | Docker verification | Build image, run the CLI in the container on the fixture | Run on the Mac and reported back; fix any arm64 or image issues |
| T9 | Real-clip tuning + docs | Run `--dry-run` on the reference sample, tune defaults, write the `--silence-db` tuning notes; fresh agent follows the README | Segment list sane when spot-checked; fresh agent succeeds from README alone |

## Risks

- **Synthetic speech may not trigger Silero.** If T3 fails on it, fall back to the real sample clip.
- **Silence threshold** varies per video; T9 documents tuning.
- **Continuous noise (air tools, compressors) is wanted**, so everything that isn't speech or silence is kept, as finely cut as possible. The silence threshold must sit just above the true silence floor, and the min-duration and gap defaults should be small; T9 tunes this on the real clip.
- **Variable frame rate / AV1** can desync audio and video; handled in T7.
- **Docker is only verified on the Mac** (see above), so T8 depends on a reply.

## Status

T0-T7 built and tested (244 tests). T9 docs done; T8 (Docker build on the Mac) pending. Real-clip render verified outside Docker (3 min 1440x2560 AV1: about 50 s, 15 clips, 14.6 s supercut).

## Out of scope (for now)

Sound labelling, EDL/FCPXML export, cloud runner, batch folder mode, web UI, a longer technical README.
