# whisper-diarize

MLX Whisper + MLX speaker diarization for Apple Silicon. Fast, accurate,
terminal-first. File, stdin-pipe, and live-microphone modes. JSON / YAML /
Markdown / SRT / VTT / NDJSON output.

Built on [mlx-audio](https://github.com/Blaizzy/mlx-audio) (v0.4.6) and
[mlx-whisper](https://github.com/ml-explore/mlx-examples).

## What it does

- **ASR**: MLX Whisper `large-v3-turbo` — fastest accurate Whisper on M-series.
- **Speaker diarization**: MLX Sortformer 4-spk (`mlx-community`) — native MLX,
  end-to-end "who spoke when".
- **Merge**: whisperX-style IntervalTree word↔speaker overlap join.
- **Outputs**: every field the segmentation produces — speaker labels, turn
  timestamps, word-level alignment, confidence.

## Install

A dedicated venv keeps MLX deps isolated (does not touch system Python):

```bash
cd /Users/priv8/.zcode/workspace/default
source .venv-whisper/bin/activate
pip install -e .          # installs the `whisper-diarize` console script
```

Or use the launcher script directly (it points at the venv):

```bash
./whisper-diarize --help
```

## Usage

### File mode
```bash
whisper-diarize audio.m4a -o json        # M4A decoded through ffmpeg
whisper-diarize conversation.m4a --num-speakers 2 -o md
whisper-diarize audio.wav -o json        # default
whisper-diarize audio.wav -o yaml
whisper-diarize audio.wav -o md          # speaker-attributed markdown
whisper-diarize audio.wav -o srt         # [SPEAKER_N] tagged
whisper-diarize audio.wav -o vtt
whisper-diarize audio.mp3 -o json | jq '.segments[].text'
```

### Pipe mode (Unix-friendly)
```bash
# WAV via stdin
cat audio.wav | whisper-diarize -o yaml

# Raw PCM s16le 16k mono (e.g. from ffmpeg/arecord)
ffmpeg -i input.flac -f s16le -ar 16000 -ac 1 - | whisper-diarize -o json

# Pipe into an LLM-friendly view
whisper-diarize meeting.wav -o md | glow -
```

### Live mode (streams NDJSON)
```bash
# Microphone (default)
whisper-diarize --live --seconds 60

# BlackHole system-audio loopback (captures call output)
whisper-diarize --live --source blackhole

# Pipe live events into jq for filtering
whisper-diarize --live --seconds 30 | jq 'select(.type=="diar")'

# Both mic + system: run two instances and merge by timestamp
```

Each NDJSON line is one event:
```json
{"type":"status","msg":"listening"}
{"type":"asr_partial","t":1.42,"text":"hello there","is_final":false}
{"type":"diar","start":0.0,"end":4.7,"speaker":0}
{"type":"final","duration_s":30.0,"text":"...","speakers":[0,1]}
```

### Python API
```python
from whisper_diarize import transcribe

result = transcribe("audio.wav", language="en")
print(result.text)
for seg in result.segments:
    print(f"[{seg.start:.2f}-{seg.end:.2f}] speaker_{seg.speaker}: {seg.text}")
    for w in seg.words:
        print(f"    {w.word} [{w.start:.2f}-{w.end:.2f}] p={w.probability:.2f} spk={w.speaker}")
```

Streaming API:
```python
from whisper_diarize import transcribe_stream
from whisper_diarize.sources import LiveAudioSource, EOS
import threading

stop = threading.Event()
src = LiveAudioSource(device=None, stop_event=stop)  # mic
src.start()
for ev in transcribe_stream(lambda: next(src.chunks(), EOS), language="en"):
    print(ev)   # dict events as above
src.stop()
```

## Device selection

```bash
whisper-diarize --list-devices
#   [1] MacBook Pro Microphone  in=1 sr=48000.0  <- mic
#   [4] BlackHole 2ch           in=2 sr=48000.0  <- system loopback

whisper-diarize --live --source 4            # by index
whisper-diarize --live --source blackhole    # by name substring
whisper-diarize --live --source "teams"      # any name match
```

## macOS microphone permission (live mode)

First live run from a new Python/Terminal binary triggers a macOS permission
dialog. If denied, macOS returns **all-zero buffers silently** (no error).
Symptoms: only `{"text":" Thank"}`-style hallucinations.

Fix once: **System Settings → Privacy & Security → Microphone → enable your
Terminal app** (or the venv Python binary). Re-run.

## System audio capture (BlackHole)

To capture what the Mac *plays* (the other side of an online call):

```bash
brew install --cask blackhole-2ch
sudo installer -pkg /opt/homebrew/Caskroom/blackhole-2ch/*/BlackHole2ch-*.pkg -target /
# reboot, then in Audio MIDI Setup create a Multi-Output Device
# containing both your speakers + BlackHole. Set it as the sound output.
whisper-diarize --live --source blackhole
```

## Models

| Mode | ASR model | Why |
|---|---|---|
| Batch | `mlx-community/whisper-large-v3-turbo` | weights-only MLX port, fast |
| Live  | `openai/whisper-large-v3-turbo` | bundles WhisperProcessor (streaming needs it); MLX still runs inference |
| Diar  | `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` | Stateful streaming diarization, ≤4 speakers |

Override anytime: `--asr-model <hf-id> --diar-model <hf-id>`.

Batch diarization uses Sortformer v2.1's native streaming state with five-second
inference chunks. Speaker identity is carried across the entire recording;
chunks are not diarized independently or stitched by guessed overlap. The
stateful attention path is bounded by the streaming cache, although file mode
still loads the recording and prepares full-file features/right context, so
total memory is not strictly constant with recording length. If the cast is
known, pass `--num-speakers N`. Otherwise, only tiny low-activity output
channels are suppressed automatically instead of using a recording-length-
scaled threshold that could remove a legitimate quiet participant.

> **Licensing note**: the NVIDIA Sortformer v2.1 source model is distributed
> under the NVIDIA Open Model License. That license permits commercial use
> subject to its terms. Review the upstream model card and license for your
> deployment and any redistributed weights; this project does not replace those
> terms with a non-commercial restriction.

## Performance (Apple M3 Max, 36 GB)

| | RTF | Notes |
|---|---|---|
| Streaming ASR (AlignAtt) | 0.45× | >2× faster than realtime |
| Streaming diarization | 0.05× | 42–566 ms per 5s chunk |
| Batch ASR + diar (18s clip) | ~0.6× | end-to-end faster than realtime |

## Layout

```
whisper_diarize/
  __init__.py     # public API
  core.py         # transcribe() + transcribe_stream() + merge logic
  sources.py      # file / stdin / mic / BlackHole audio sources
  formatters.py   # json / yaml / md / srt / vtt / txt
  cli.py          # whisper-diarize entry point
  __main__.py
whisper-diarize   # venv-launcher shell script
poc/              # proof-of-concept scripts (poc.py, realtime_*.py, self-tests)
pyproject.toml
```
