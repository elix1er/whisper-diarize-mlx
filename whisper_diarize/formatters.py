"""Output formatters for DiarizationResult: json, yaml, md, srt, vtt, txt."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import List

from .core import DiarizationResult, Segment


def _result_dict(r: DiarizationResult, audio: str = None) -> dict:
    return {
        **({"audio": audio} if audio else {}),
        "language": r.language,
        "models": r.models,
        "speakers": r.speakers,
        "num_speakers": len(r.speakers),
        "diarization_turns": r.diarization_turns,
        "text": r.text,
        "segments": [
            {**asdict(s), "words": [asdict(w) for w in s.words]}
            for s in r.segments
        ],
    }


def to_json(r: DiarizationResult, audio: str = None) -> str:
    return json.dumps(_result_dict(r, audio), indent=2, ensure_ascii=False)


def to_yaml(r: DiarizationResult, audio: str = None) -> str:
    try:
        import yaml
    except ImportError:
        return "# PyYAML not installed; pip install pyyaml\n"
    return yaml.safe_dump(_result_dict(r, audio), sort_keys=False, allow_unicode=True)


def to_markdown(r: DiarizationResult, audio: str = None) -> str:
    lines = []
    if audio:
        lines.append(f"# Transcript: `{audio}`\n")
    lines.append(f"*{len(r.speakers)} speakers | {len(r.segments)} segments*\n")
    cur = -1
    for seg in r.segments:
        if seg.speaker != cur:
            cur = seg.speaker
            tag = f"SPEAKER_{cur}" if cur >= 0 else "UNKNOWN"
            lines.append(f"\n## {tag}  ({seg.start:.2f}s)\n")
        lines.append(f"> {seg.text}")
    return "\n".join(lines) + "\n"


def _ts(seconds: float) -> str:
    """SRT/VTT timestamp: HH:MM:SS,mmm or HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(r: DiarizationResult, audio: str = None) -> str:
    lines = []
    for i, seg in enumerate(r.segments, 1):
        tag = f"SPEAKER_{seg.speaker}" if seg.speaker >= 0 else "?"
        lines.append(str(i))
        lines.append(f"{_ts(seg.start)} --> {_ts(seg.end)}")
        lines.append(f"[{tag}] {seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_vtt(r: DiarizationResult, audio: str = None) -> str:
    lines = ["WEBVTT", ""]
    for seg in r.segments:
        tag = f"SPEAKER_{seg.speaker}" if seg.speaker >= 0 else "?"
        s = _ts(seg.start).replace(",", ".")
        e = _ts(seg.end).replace(",", ".")
        lines.append(f"{s} --> {e}")
        lines.append(f"<v {tag}>{seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_txt(r: DiarizationResult, audio: str = None) -> str:
    return r.text + "\n"


FORMATTERS = {
    "json": to_json,
    "yaml": to_yaml,
    "md": to_markdown,
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_txt,
}
