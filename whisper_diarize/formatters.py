"""Output formatters for offline transcription results."""
from __future__ import annotations

import json
from dataclasses import asdict

from .types import DiarizationResult


def _result_dict(result: DiarizationResult, audio: str = None) -> dict:
    return {
        **({"audio": audio} if audio else {}),
        "language": result.language,
        "models": result.models,
        "speakers": result.speakers,
        "num_speakers": len(result.speakers),
        "diarization_turns": result.diarization_turns,
        "text": result.text,
        "segments": [asdict(segment) for segment in result.segments],
    }


def to_json(result: DiarizationResult, audio: str = None) -> str:
    return json.dumps(_result_dict(result, audio), indent=2, ensure_ascii=False)


def to_yaml(result: DiarizationResult, audio: str = None) -> str:
    import yaml

    return yaml.safe_dump(_result_dict(result, audio), sort_keys=False, allow_unicode=True)


def to_markdown(result: DiarizationResult, audio: str = None) -> str:
    lines = []
    if audio:
        lines.append(f"# Transcript: `{audio}`\n")
    lines.append(f"*{len(result.speakers)} speakers | {len(result.segments)} segments*\n")
    current = None
    for segment in result.segments:
        if segment.speaker != current:
            current = segment.speaker
            tag = f"SPEAKER_{current}" if current >= 0 else "UNKNOWN"
            lines.append(f"\n## {tag}  ({segment.start:.2f}s)\n")
        lines.append(f"> {segment.text}")
    return "\n".join(lines) + "\n"


def _timestamp(seconds: float, decimal: str = ",") -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal}{milliseconds:03d}"


def to_srt(result: DiarizationResult, audio: str = None) -> str:
    lines = []
    for index, segment in enumerate(result.segments, 1):
        tag = f"SPEAKER_{segment.speaker}" if segment.speaker >= 0 else "?"
        lines.extend(
            [
                str(index),
                f"{_timestamp(segment.start)} --> {_timestamp(segment.end)}",
                f"[{tag}] {segment.text}",
                "",
            ]
        )
    return "\n".join(lines)


def to_vtt(result: DiarizationResult, audio: str = None) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        tag = f"SPEAKER_{segment.speaker}" if segment.speaker >= 0 else "?"
        lines.extend(
            [
                f"{_timestamp(segment.start, '.')} --> {_timestamp(segment.end, '.')}",
                f"<v {tag}>{segment.text}",
                "",
            ]
        )
    return "\n".join(lines)


def to_txt(result: DiarizationResult, audio: str = None) -> str:
    return result.text + "\n"


FORMATTERS = {
    "json": to_json,
    "yaml": to_yaml,
    "md": to_markdown,
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_txt,
}
