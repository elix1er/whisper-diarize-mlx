---
name: whisper-diarize-mlx
description: Transcribe, diarize, compare, or troubleshoot real audio locally on Apple Silicon with this repository. Use for whisper-diarize runtime work and transcript-quality evaluation; do not use for generic audio summaries that do not require this tool.
---

# Whisper Diarize MLX

Use this repository's current CLI and Python API as the authority. Verify the
checked-out revision, available command options, input media, and local runtime
instead of assuming an earlier run still represents the current code.

## Run file transcription

1. Inspect the source with `ffprobe` when codec, duration, channels, or sample
   rate could affect the result. Keep the original recording as the source of
   truth; do not silently reuse an old conversion or transcript.
2. Reuse a working project environment. When a fresh environment is needed,
   use a Python version supported by `pyproject.toml` and install this checkout
   editable so the invoked CLI and imported package resolve to the same
   revision.
3. Run `whisper-diarize` with an explicit language when it is known. Use the
   default large-v3-turbo model unless the user requests another model or the
   task is a model comparison. Supply `--num-speakers` only when the cast size is
   known; otherwise preserve automatic speaker selection.
4. Write generated transcripts and model outputs outside the repository unless
   the user explicitly asks to version a fixture. Quote input and output paths.
   Do not commit recordings, transcripts, derived JSON, or downloaded weights.
5. Keep stdout machine-readable when redirecting output. Avoid `--verbose` for
   JSON or other captured output unless progress and payload have first been
   separated.

Example:

```bash
whisper-diarize "meeting.m4a" --language de --num-speakers 2 -o md > "transcript.md"
```

The offline API intentionally defaults to segment timestamps and
`condition_on_previous_text=False`. Preserve those defaults for long-form work:
they prevent word-alignment seek drift and reduce the chance that one failed
window contaminates later windows. Opt into `word_timestamps=True` only when
word-level timing is actually required.

## Compare transcript quality

Hold the audio, language, diarization choice, and decoding settings constant;
change only the model under evaluation. Use `--no-diar` when the question is ASR
quality alone so speaker attribution does not confound the comparison.

Read both transcripts across the recording, including the beginning, middle,
and end. Compare meaning and evidence, not file size or segment count:

- missing or materially changed statements;
- invented text, repetition loops, or sudden language drift;
- names, domain terms, numbers, and sentence coherence;
- timestamp continuity, blank spans, and recovery after difficult audio.

Without a human reference transcript, call the result a qualitative comparison,
not a measured accuracy win. A larger model is not automatically better. If one
run collapses, first reproduce it with the same safe decoding defaults before
attributing the failure to model quality.

## Validate changes

For code or dependency changes, run the fast test suite first, then exercise the
affected path with representative real audio on Apple Silicon. Inspect the
actual rendered transcript or parsed JSON and, when diarization is involved,
speaker IDs and unknown assignments. Unit tests alone do not validate model
loading, media decoding, long-recording behavior, or transcript quality.

Report the exact model, language, relevant options, source duration, output
path, and what was directly verified. Keep transcript-derived summaries grounded
in what was said; do not turn uncertain ASR text into confident business facts.
