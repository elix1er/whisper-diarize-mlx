"""Offline file/array transcription using Whisper + stateful Sortformer diarization."""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np

from .attribution import merge_asr_segments, merge_turns, select_speakers
from .types import DiarizationResult

DEFAULT_ASR_FILE = "mlx-community/whisper-large-v3-turbo"
DEFAULT_DIAR = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
DEFAULT_DIAR_CHUNK_SEC = 5.0
DEFAULT_HALLUCINATION_SILENCE_SEC = 2.0


def transcribe(
    audio: Union[str, np.ndarray],
    *,
    asr_model: str = DEFAULT_ASR_FILE,
    diar_model: str = DEFAULT_DIAR,
    language: Optional[str] = None,
    no_diar: bool = False,
    word_timestamps: bool = True,
    condition_on_previous_text: bool = False,
    hallucination_silence_threshold: Optional[float] = DEFAULT_HALLUCINATION_SILENCE_SEC,
    initial_prompt: Optional[str] = None,
    diar_threshold: float = 0.5,
    diar_chunk_sec: float = DEFAULT_DIAR_CHUNK_SEC,
    num_speakers: Optional[int] = None,
    verbose: bool = False,
) -> DiarizationResult:
    """Transcribe an audio file or 16 kHz mono array and attribute words to speakers.

    Offline files default to a repetition-safe decode: each Whisper window is
    decoded without feeding the prior window's text back as a prompt, and the
    library's silence-aware hallucination guard is enabled. This avoids a bad
    window poisoning the remaining recording while retaining word timestamps
    for speaker attribution.
    """
    if verbose:
        print(f"[whisper_diarize] ASR {asr_model}", flush=True)

    import mlx_whisper

    started = time.time()
    asr = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=asr_model,
        word_timestamps=word_timestamps,
        language=language,
        condition_on_previous_text=condition_on_previous_text,
        hallucination_silence_threshold=hallucination_silence_threshold,
        initial_prompt=initial_prompt,
    )
    if verbose:
        print(f"[whisper_diarize] ASR done {time.time() - started:.2f}s", flush=True)

    turns: list[dict] = []
    if not no_diar:
        if verbose:
            print(f"[whisper_diarize] DIAR {diar_model}", flush=True)
        from mlx_audio.vad import load as load_vad

        started = time.time()
        model = load_vad(diar_model)
        raw_turns: list[dict] = []
        for output in model.generate_stream(
            audio,
            chunk_duration=diar_chunk_sec,
            threshold=diar_threshold,
            min_duration=0.1,
            merge_gap=0.1,
            verbose=verbose,
        ):
            raw_turns.extend(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": int(segment.speaker),
                }
                for segment in output.segments
            )
        turns = merge_turns(select_speakers(raw_turns, num_speakers))
        if verbose:
            print(
                f"[whisper_diarize] DIAR done {time.time() - started:.2f}s ({len(turns)} turns)",
                flush=True,
            )

    segments = merge_asr_segments(asr, turns)
    speakers = sorted({segment.speaker for segment in segments if segment.speaker >= 0})
    return DiarizationResult(
        text=asr.get("text", "").strip(),
        segments=segments,
        speakers=speakers,
        diarization_turns=turns,
        language=asr.get("language"),
        models={
            "asr": asr_model,
            "diarization": None if no_diar else diar_model,
            "decode": {
                "condition_on_previous_text": condition_on_previous_text,
                "hallucination_silence_threshold": hallucination_silence_threshold,
                "initial_prompt": initial_prompt,
            },
        },
    )
