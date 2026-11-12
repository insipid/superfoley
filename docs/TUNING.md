# Tuning

Start with `--dry-run` and look at `segments.csv`. The pipeline only decides *what is speech* and *what is silence*; everything else is kept.

## The one knob that matters: `--silence-db`

Frames quieter than this are silence and are dropped. Too high and quiet real sounds vanish; too low and room tone is kept as "sound".

Find your floor: run with `--dry-run` at -55, -50, -45 and compare the kept time. If the kept total jumps a lot between two values, the room-tone floor sits between them; set the threshold just above it.

## Other knobs

- `--vad-threshold`: raise (0.6-0.7) if non-speech sounds are being treated as speech; lower if talking leaks into the kept sounds.
- `--speech-pad-ms`: how far to stay away from detected speech. Lower it to keep sounds that start right after a sentence; raise it if word tails leak in.
- `--min-gap-ms` / `--min-dur-ms`: lower both for finer detail (more, shorter segments); raise to smooth out flicker.
- `--pad-ms`: breathing room added around each kept sound after everything else.

## Reference clip (vertical 3-minute mechanic video, mostly narrated)

Speech was detected for about 79% of the timeline. Kept time at various settings:

| Options | Segments | Kept |
|---|---|---|
| defaults | 15 | 14.6 s |
| `--silence-db -50` | 15 | 16.7 s |
| `--silence-db -55` | 14 | 18.1 s |
| `--speech-pad-ms 50` | 15 | 15.6 s |
| `--speech-pad-ms 50 --min-gap-ms 80 --min-dur-ms 80 --silence-db -50` | 20 | 17.4 s |
| `--vad-threshold 0.7` | 14 | 15.3 s |

Check the result by listening. Defaults were not tuned by ear.

## Known limits

- Sounds that overlap speech are dropped (speech wins).
- A tonal sound right after speech can be read as speech by the detector for a few hundred milliseconds, which clips the start of that sound. Lowering `--speech-pad-ms` helps.
