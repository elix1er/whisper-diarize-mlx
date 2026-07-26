"""whisper_diarize: MLX Whisper + MLX speaker diarization for Apple Silicon.

Two modes:
  - Batch (file / stdin pipe): full transcription + diarization + merge.
  - Live (mic / BlackHole system audio): streaming NDJSON events.

Outputs: json | yaml | md | srt | vtt | ndjson (live).

Public API:
  from whisper_diarize import transcribe, transcribe_stream
"""
from .core import (
    transcribe,
    transcribe_stream,
    DiarizationResult,
    Segment,
    Word,
)

__version__ = "0.1.0"
__all__ = ["transcribe", "transcribe_stream", "DiarizationResult", "Segment", "Word"]
