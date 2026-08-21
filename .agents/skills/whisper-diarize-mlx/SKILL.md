---
name: whisper-diarize-mlx
description: Turn local audio or video recordings into speaker-labelled transcripts on Apple Silicon using this repository. Use when asked to transcribe, diarize, extract, summarize, or compare recordings with whisper-diarize-mlx.
---

# Whisper Diarize MLX

Use this repository to take a recording through local transcription and speaker
diarization and deliver the requested transcript or transcript-derived result.
Treat the checked-out code, its README, and current CLI help as authoritative.

## Produce the result

1. Resolve the source recording and requested destination. Work from the
   original source unless the user explicitly asks to reuse an existing run.
   Inspect media metadata only when format, duration, channels, or decoding may
   matter.
2. Reuse a working project environment. When setup is needed, use a Python
   version supported by `pyproject.toml` and install the current checkout
   editable so the CLI and imported package resolve to the same revision.
3. Choose only the options the task needs:
   - Set `--language` when the language is known.
   - Use the default large-v3-turbo ASR model unless another model is requested.
   - Set `--num-speakers` only when the speaker count is known.
   - Select the output format that best fits the requested artifact: Markdown
     for reading, JSON for further processing, or SRT/VTT for subtitles.
4. Run the transcription and save the result at the requested location. Quote
   paths, especially filenames containing spaces or punctuation.
5. Read the generated output before delivery. Check that it contains coherent
   text across the recording, usable timestamps, and plausible speaker labels;
   look for repetition loops, blank spans, language drift, or obvious model
   collapse.
6. Deliver the artifact with the model, language, relevant options, and any
   material quality caveat. If the user asks for a summary or extraction, keep
   it grounded in the transcript and distinguish uncertain recognition from
   confidently spoken content.

Example:

```bash
whisper-diarize "meeting.m4a" --language de --num-speakers 2 -o md > "transcript.md"
```

Keep recordings, generated transcripts, derived JSON, and downloaded model
weights outside the repository unless the user explicitly requests a versioned
fixture. Avoid `--verbose` when redirecting structured output because progress
must not contaminate the payload.

## Preserve reliable long-form decoding

The offline API defaults to segment timestamps and
`condition_on_previous_text=False`. Preserve these defaults for normal runs:
they retain speaker-labelled segments while reducing timestamp drift and
cross-window repetition. Opt into `word_timestamps=True` only when word-level
timing is part of the requested result.

## Compare models when requested

Hold the audio, language, diarization choice, and decoding settings constant;
change only the ASR model. Use `--no-diar` when comparing recognition alone.
Read both transcripts across the beginning, middle, and end, then compare
meaning, omissions, invented or repeated text, names, domain terms, numbers,
sentence coherence, timestamp continuity, and recovery after difficult audio.

Without a human reference transcript, describe the result as a qualitative
comparison rather than measured accuracy. If one run collapses, reproduce it
with the same reliable decoding defaults before blaming the model itself.

## Validate repository changes when requested

For implementation or dependency changes, run the fast tests and then exercise
the affected path with representative real audio on Apple Silicon. Unit tests
alone do not verify model loading, media decoding, long-recording behavior,
speaker attribution, or transcript quality.
