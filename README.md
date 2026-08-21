<div align="center">

# whisper-diarize

**Local, speaker-aware transcription for Apple Silicon.**

Whisper large-v3-turbo for speech recognition, Sortformer v2.1 for stateful speaker diarization, word-level speaker attribution for files, and true incremental microphone/system-audio processing for live workloads.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

## Highlights

- **Local on Apple Silicon** — MLX-native inference; no hosted transcription API required.
- **Speaker-attributed output** — Whisper word timestamps are matched to Sortformer speaker turns.
- **Long-recording friendly** — offline diarization uses Sortformer v2.1's stateful streaming path instead of full-sequence attention over the entire recording.
- **True live PCM input** — microphone or loopback audio is incrementally fed into Whisper's streaming decoder and persistent Sortformer state.
- **Useful output formats** — JSON, YAML, Markdown, SRT, VTT and plain text for files; NDJSON events for live mode.
- **Inspectible pipeline** — ASR, diarization and attribution remain separate, replaceable stages rather than one opaque end-to-end result.

## Quick start

Requires macOS on Apple Silicon and Python 3.10+.

```bash
git clone https://github.com/elix1er/whisper-diarize-mlx.git
cd whisper-diarize-mlx

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

whisper-diarize audio.m4a -o md
```

Models are downloaded on first use.

## File transcription

```bash
# Structured JSON
whisper-diarize meeting.m4a -o json

# Speaker-readable Markdown
whisper-diarize meeting.m4a -o md

# Known cast size (Sortformer supports up to four speakers)
whisper-diarize interview.m4a --num-speakers 2 -o md

# Subtitles
whisper-diarize recording.wav -o srt
whisper-diarize recording.wav -o vtt
```

Pipe WAV or raw 16 kHz mono PCM through stdin:

```bash
cat audio.wav | whisper-diarize -o json
ffmpeg -i input.flac -f s16le -ar 16000 -ac 1 - | whisper-diarize -o json
```

### Python API

```python
from whisper_diarize import transcribe

result = transcribe("meeting.m4a", language="en")

for segment in result.segments:
    print(
        f"[{segment.start:7.2f}-{segment.end:7.2f}] "
        f"speaker_{segment.speaker}: {segment.text}"
    )

    for word in segment.words:
        print(word.word, word.start, word.end, word.probability, word.speaker)
```

## Live transcription

Live mode emits newline-delimited JSON events as audio arrives.

```bash
# Default microphone
whisper-diarize --live

# Stop automatically after 60 seconds
whisper-diarize --live --seconds 60

# BlackHole / other loopback input
whisper-diarize --live --source blackhole

# Find available input devices
whisper-diarize --list-devices
```

Example stream:

```json
{"type":"status","msg":"listening"}
{"type":"asr_partial","t":1.42,"text":"hello there","is_final":false}
{"type":"diar","start":0.0,"end":4.7,"speaker":0}
{"type":"final","duration_s":30.0,"text":"...","diarization":[...],"speakers":[0,1]}
```

The live path is incremental: new PCM is fed into Whisper's streaming decoder while Sortformer keeps persistent speaker state across diarization chunks. It is not merely token streaming over an already-loaded recording.

### System audio on macOS

A loopback input such as [BlackHole](https://github.com/ExistentialAudio/BlackHole) can expose system playback as an input device. After configuring a Multi-Output Device in Audio MIDI Setup:

```bash
whisper-diarize --live --source blackhole
```

macOS microphone/screen-audio permissions still apply to the terminal or Python process doing the capture.

## Architecture

```text
                         audio
                           |
                +----------+----------+
                |                     |
             Whisper              Sortformer
          speech + words        speaker activity
                |                     |
                +----------+----------+
                           |
                    attribution
                           |
                 speaker transcript
```

The package keeps four concerns deliberately small and separate:

```text
whisper_diarize/
├── offline.py      # file/array ASR + stateful diarization
├── streaming.py    # true incremental ASR + diarization
├── attribution.py  # word/segment ↔ speaker policy
├── types.py        # model-independent result objects
├── sources.py      # live capture + resampling
├── formatters.py   # json/yaml/md/srt/vtt/txt
└── cli.py          # terminal interface
```

`core.py` remains as a compatibility facade for earlier imports.

### Why Whisper + Sortformer instead of one diarizing ASR model?

`mlx-audio` also supports end-to-end speaker-aware models such as [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize), which generate transcript text, timestamps and speaker labels jointly.

That is attractive for simple offline transcription. This project intentionally keeps specialist ASR and diarization components separate because that is a better fit for its primary requirements:

| | Whisper + Sortformer | End-to-end diarizing ASR |
|---|---|---|
| Live PCM | **Incremental Whisper + persistent Sortformer state** | Depends on model/runtime; token streaming is not necessarily live audio streaming |
| Timing | **Whisper word timestamps + confidence** | Commonly segment-oriented generated timestamps |
| Debugging | **ASR, diarization and attribution inspectible separately** | Errors are coupled in one generated result |
| Model upgrades | **ASR and diarizer replaceable independently** | Transcription and speaker behavior are tied together |
| Offline simplicity | More moving parts | **Single-model pipeline** |

So this is not a claim that a cascade always wins accuracy. It wins here on **live-stream architecture, word-level output, observability and replaceable best-of-breed components**. End-to-end models remain useful benchmark targets and potential future offline backends.

## Models

| Role | Default | Notes |
|---|---|---|
| Offline ASR | `mlx-community/whisper-large-v3-turbo` | MLX Whisper port with word timestamps |
| Live ASR | `openai/whisper-large-v3-turbo` | Used through mlx-audio's Whisper streaming decoder |
| Diarization | `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` | Stateful Sortformer v2.1, up to four speakers |

Override model IDs with `--asr-model` and `--diar-model`.

The package currently pins `mlx-audio` to `>=0.4.6,<0.5` because live Whisper integration uses its streaming implementation directly. That boundary should be reviewed when moving to a newer mlx-audio minor/major API.

### Model licensing

The repository source code is MIT licensed. Model weights are downloaded separately and keep their own upstream licenses. In particular, NVIDIA Sortformer v2.1 is distributed under the NVIDIA Open Model License. Review each selected model's upstream model card/license for your deployment and redistribution requirements.

## Behavior and limitations

- Sortformer exposes up to four speaker channels; `--num-speakers` accepts `1..4` for offline files.
- When speaker count is not supplied, tiny low-activity channels are suppressed with a small absolute activity floor rather than a recording-length-relative threshold.
- Word attribution prefers temporal overlap and only uses nearest-speaker fallback across a bounded short gap. Distant diarization misses remain unknown instead of being force-labeled.
- Offline Sortformer processing preserves streaming speaker state across the recording, but file mode can still prepare full-file features/right context internally; memory is therefore not strictly constant with recording length.
- Live events currently expose ASR and diarization updates independently. Consumers that need a continuously revised speaker-attributed transcript should reconcile those events or use offline final transcription.

## Development

The fast tests exercise attribution and CLI contracts without downloading models:

```bash
python -m unittest discover -s tests -v
```

For real audio validation, run the installed CLI on representative recordings and live input on Apple Silicon. Model-backed benchmarks are intentionally not published in the README until they are reproducible against the current default model versions.

## License

[MIT](LICENSE) for this repository's source code. Downloaded model weights are licensed separately by their respective upstream projects.
