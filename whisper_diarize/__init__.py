"""Local speaker-aware transcription for Apple Silicon."""
from .offline import transcribe
from .streaming import transcribe_stream
from .types import DiarizationResult, Segment, Word

__version__ = "0.2.0"
__all__ = ["transcribe", "transcribe_stream", "DiarizationResult", "Segment", "Word"]
