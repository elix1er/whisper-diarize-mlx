#!/usr/bin/env python3
"""
Shared realtime core: MLX streaming ASR (AlignAtt) + streaming diarization
(Sortformer feed()), driven by an audio source that yields 16k mono float32
chunks. Used by both:
  - realtime_sim.py : chunk source = a file (offline validation)
  - realtime_mic.py : chunk source = the live microphone

Design:
  - ASR:   one StreamingDecoder instance, fed each new chunk's mel. It
           accumulates mel internally and emits partial/final text.
  - DIAR:  one Sortformer streaming state, fed accumulated audio every
           DIAR_CHUNK seconds via model.feed().

Emits NDJSON to stdout. This is the kernel the final CLI will wrap.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

SR = 16000

# Sentinel a chunk_source returns to signal end-of-stream. (A mic source never
# returns this; a file/batch source returns it when its audio is exhausted.)
EOS = object()


def emit(obj: dict, stream=sys.stdout):
    stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    stream.flush()


@dataclass
class RealtimeResult:
    text: str = ""
    diarization: List[dict] = field(default_factory=list)
    speakers: List[int] = field(default_factory=list)


def make_asr_decoder(asr_model, language: str = "en"):
    """Construct a stateful StreamingDecoder bound to a loaded Whisper model."""
    from mlx_audio.stt.models.whisper.streaming import StreamingConfig, StreamingDecoder
    cfg = StreamingConfig(frame_threshold=25)
    return StreamingDecoder(asr_model, cfg, language=language, task="transcribe")


def mel_of(chunk: np.ndarray, n_mels: int):
    """Log-mel spectrogram for a 16k mono chunk (matches Whisper frontend)."""
    from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram
    return log_mel_spectrogram(chunk, n_mels=n_mels)


def run_realtime(
    *,
    asr_model,
    diar_model,
    chunk_source: Callable[[float, "RealtimeState"], Optional[np.ndarray]],
    language: str = "en",
    asr_chunk_sec: float = 1.0,
    diar_chunk_sec: float = 5.0,
    max_seconds: float = 0.0,
    verbose: bool = False,
    emit_events: bool = True,
) -> RealtimeResult:
    """Drive the two streaming pipelines from `chunk_source`.

    chunk_source(timeout_s, state) -> np.ndarray (16k mono float32) or None.
    Returns a RealtimeResult with full text + diarization.
    """
    decoder = make_asr_decoder(asr_model, language=language)
    n_mels = asr_model.dims.n_mels

    diar_state = diar_model.init_streaming_state() if diar_model is not None else None
    acc_diar = np.zeros(0, dtype=np.float32)
    diar_off = 0.0

    diar_chunk_samples = int(SR * diar_chunk_sec)
    result = RealtimeResult()
    t0 = time.time()
    deadline = (t0 + max_seconds) if max_seconds > 0 else None
    chunk_no = 0

    if emit_events:
        emit({"type": "status", "msg": "listening"})

    while True:
        if deadline and time.time() > deadline:
            break
        chunk = chunk_source(timeout_s=0.2, state=None)
        if chunk is None:
            continue
        if chunk is EOS:
            break
        if len(chunk) == 0:
            continue
        elapsed = time.time() - t0
        chunk_no += 1

        # --- ASR: feed ONLY the new chunk's mel; decoder keeps state ---
        mel = mel_of(chunk, n_mels)
        try:
            res = decoder.decode_chunk(mel, is_last=False)
        except Exception as e:
            if emit_events:
                emit({"type": "error", "stage": "asr", "msg": str(e)})
            continue
        if res.text.strip():
            if emit_events:
                emit({"type": "asr_partial", "t": round(elapsed, 2), "text": res.text,
                      "is_final": bool(res.is_final)})
            if verbose:
                print(f"  [asr {elapsed:6.2f}s] {res.text!r}", file=sys.stderr)
            # accumulate text: StreamingDecoder emits only newly-committed tokens,
            # so appending gives the running transcript.
            result.text = (result.text + " " + res.text).strip()

        # --- DIAR: accumulate and feed every diar_chunk_sec ---
        if diar_model is not None:
            acc_diar = np.concatenate([acc_diar, chunk])
            while len(acc_diar) >= diar_chunk_samples:
                feed = acc_diar[:diar_chunk_samples]
                acc_diar = acc_diar[diar_chunk_samples:]
                try:
                    out, diar_state = diar_model.feed(feed, diar_state)
                except Exception as e:
                    if emit_events:
                        emit({"type": "error", "stage": "diar", "msg": str(e)})
                    break
                for s in out.segments:
                    rec = {"type": "diar",
                           "start": round(s.start + diar_off, 3),
                           "end": round(s.end + diar_off, 3),
                           "speaker": int(s.speaker)}
                    if emit_events:
                        emit(rec)
                    if verbose:
                        print(f"  [diar {elapsed:6.2f}s] spk{s.speaker} "
                              f"{s.start+diar_off:.2f}-{s.end+diar_off:.2f}",
                              file=sys.stderr)
                    result.diarization.append(
                        {"start": rec["start"], "end": rec["end"], "speaker": rec["speaker"]})
                diar_off += diar_chunk_sec

    # flush ASR
    try:
        if len(acc_diar) > 0 and diar_model is not None:
            pass  # nothing to flush for ASR; decoder state discarded
    except Exception:
        pass

    result.speakers = sorted({d["speaker"] for d in result.diarization})
    if emit_events:
        emit({"type": "final", "duration_s": round(time.time() - t0, 2),
              "text": result.text, "diarization": result.diarization,
              "speakers": result.speakers})
    return result
