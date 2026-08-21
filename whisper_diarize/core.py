"""Compatibility facade for the public transcription API.

New code may import from ``offline``, ``streaming``, ``attribution`` and ``types``
directly; existing ``whisper_diarize.core`` imports remain supported.
"""
from .attribution import (
    _assign_speaker,
    _merge_segments,
    _merge_turns,
    _select_speakers,
    DEFAULT_SPEAKER_GAP_SEC,
)
from .offline import DEFAULT_ASR_FILE, DEFAULT_DIAR, DEFAULT_DIAR_CHUNK_SEC, transcribe
from .streaming import DEFAULT_ASR_STREAM, transcribe_stream
from .types import DiarizationResult, Segment, Word

# Backward-compatible name used by earlier releases.
DEFAULT_ASR_BATCH = DEFAULT_ASR_FILE

__all__ = [
    "DEFAULT_ASR_FILE",
    "DEFAULT_ASR_BATCH",
    "DEFAULT_ASR_STREAM",
    "DEFAULT_DIAR",
    "DEFAULT_DIAR_CHUNK_SEC",
    "DEFAULT_SPEAKER_GAP_SEC",
    "DiarizationResult",
    "Segment",
    "Word",
    "transcribe",
    "transcribe_stream",
]
