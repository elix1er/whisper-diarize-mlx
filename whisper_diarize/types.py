"""Public result types for speaker-aware transcription."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float = 0.0
    speaker: int = -1


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: int
    no_speech_prob: float = 0.0
    words: list[Word] = field(default_factory=list)


@dataclass
class DiarizationResult:
    text: str
    segments: list[Segment]
    speakers: list[int]
    diarization_turns: list[dict]
    language: Optional[str] = None
    models: dict = field(default_factory=dict)
