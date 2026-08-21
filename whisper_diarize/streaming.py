"""True incremental microphone/system-audio transcription and diarization."""
from __future__ import annotations

import time
from typing import Callable, Generator, Optional

import numpy as np

from .offline import DEFAULT_DIAR, DEFAULT_DIAR_CHUNK_SEC
from .sources import EOS, SR

DEFAULT_ASR_STREAM = "openai/whisper-large-v3-turbo"


def transcribe_stream(
    chunk_source: Callable[[], Optional[np.ndarray]],
    *,
    asr_model: str = DEFAULT_ASR_STREAM,
    diar_model: str = DEFAULT_DIAR,
    language: str = "en",
    no_diar: bool = False,
    diar_chunk_sec: float = DEFAULT_DIAR_CHUNK_SEC,
    diar_threshold: float = 0.5,
    max_seconds: float = 0.0,
    verbose: bool = False,
) -> Generator[dict, None, None]:
    """Stream speaker-aware events from a source of 16 kHz mono float32 chunks."""
    from mlx_audio.stt import load as load_stt
    from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram
    from mlx_audio.stt.models.whisper.streaming import StreamingConfig, StreamingDecoder

    if verbose:
        print(f"[whisper_diarize] ASR(stream) {asr_model}", flush=True)
    asr = load_stt(asr_model)
    decoder = StreamingDecoder(
        asr,
        StreamingConfig(frame_threshold=25),
        language=language,
        task="transcribe",
    )
    n_mels = asr.dims.n_mels

    diar_model_obj = None
    diar_state = None
    if not no_diar:
        if verbose:
            print(f"[whisper_diarize] DIAR(stream) {diar_model}", flush=True)
        from mlx_audio.vad import load as load_vad

        diar_model_obj = load_vad(diar_model)
        diar_state = diar_model_obj.init_streaming_state()

    yield {"type": "status", "msg": "listening"}

    diar_buffer = np.zeros(0, dtype=np.float32)
    diar_chunk_samples = int(SR * diar_chunk_sec)
    full_text_parts: list[str] = []
    diar_segments: list[dict] = []
    started = time.time()
    deadline = started + max_seconds if max_seconds > 0 else None

    def emit_diar_chunk(feed: np.ndarray):
        nonlocal diar_state
        try:
            output, diar_state = diar_model_obj.feed(
                feed,
                diar_state,
                threshold=diar_threshold,
                min_duration=0.1,
                merge_gap=0.1,
            )
        except Exception as exc:
            return [{"type": "error", "stage": "diar", "msg": str(exc)}]

        events = []
        for segment in output.segments:
            # mlx-audio feed() already returns stream-global timestamps derived
            # from state.frames_processed; do not apply an extra chunk offset.
            rec = {
                "type": "diar",
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "speaker": int(segment.speaker),
            }
            diar_segments.append({k: v for k, v in rec.items() if k != "type"})
            events.append(rec)
        return events

    while True:
        if deadline and time.time() > deadline:
            break
        chunk = chunk_source()
        if chunk is EOS:
            break
        if chunk is None or len(chunk) == 0:
            continue

        elapsed = time.time() - started
        chunk = np.asarray(chunk, dtype=np.float32)

        # ASR and diarization intentionally fail independently: a transient ASR
        # failure must not remove audio from the speaker-state timeline.
        try:
            mel = log_mel_spectrogram(chunk, n_mels=n_mels)
            result = decoder.decode_chunk(mel, is_last=False)
            if result.text.strip():
                full_text_parts.append(result.text)
                yield {
                    "type": "asr_partial",
                    "t": round(elapsed, 2),
                    "text": result.text,
                    "is_final": bool(result.is_final),
                }
                if verbose:
                    print(f"  [asr {elapsed:6.2f}s] {result.text!r}", flush=True)
        except Exception as exc:
            yield {"type": "error", "stage": "asr", "msg": str(exc)}

        if diar_model_obj is not None:
            diar_buffer = np.concatenate([diar_buffer, chunk])
            while len(diar_buffer) >= diar_chunk_samples:
                feed = diar_buffer[:diar_chunk_samples]
                diar_buffer = diar_buffer[diar_chunk_samples:]
                for event in emit_diar_chunk(feed):
                    yield event

    # Flush any tokens that AlignAtt held back without duplicating audio. An
    # empty mel chunk leaves accumulated audio unchanged while is_last=True
    # asks the decoder to commit remaining tokens.
    try:
        import mlx.core as mx

        final_result = decoder.decode_chunk(mx.zeros((0, n_mels)), is_last=True)
        if final_result.text.strip():
            full_text_parts.append(final_result.text)
            yield {
                "type": "asr_partial",
                "t": round(time.time() - started, 2),
                "text": final_result.text,
                "is_final": True,
            }
    except Exception as exc:
        yield {"type": "error", "stage": "asr_flush", "msg": str(exc)}

    # Do not silently discard the final partial diarization window.
    if diar_model_obj is not None and len(diar_buffer) > 0:
        for event in emit_diar_chunk(diar_buffer):
            yield event

    speakers = sorted({segment["speaker"] for segment in diar_segments})
    yield {
        "type": "final",
        "duration_s": round(time.time() - started, 2),
        "text": " ".join(full_text_parts).strip(),
        "diarization": diar_segments,
        "speakers": speakers,
    }
