"""Pure speaker-attribution helpers shared by offline transcription and tests."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .types import Segment, Word

DEFAULT_SPEAKER_GAP_SEC = 2.0


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    return inter if inter > 0 else 0.0


def _gap_to_turn(t0: float, t1: float, turn: dict) -> float:
    start = float(turn["start"])
    end = float(turn["end"])
    if t1 < start:
        return start - t1
    if end < t0:
        return t0 - end
    return 0.0


def assign_speaker(
    t0: float,
    t1: float,
    turns: list[dict],
    max_gap_sec: float = DEFAULT_SPEAKER_GAP_SEC,
) -> int:
    """Assign the dominant overlapping speaker, with a bounded nearest fallback."""
    best_spk, best_dur = -1, 0.0
    for turn in turns:
        duration = _overlap(t0, t1, float(turn["start"]), float(turn["end"]))
        if duration > best_dur:
            best_dur, best_spk = duration, int(turn["speaker"])
    if best_spk >= 0 or not turns:
        return best_spk

    nearest = min(turns, key=lambda turn: _gap_to_turn(t0, t1, turn))
    if _gap_to_turn(t0, t1, nearest) <= max_gap_sec:
        return int(nearest["speaker"])
    return -1


def merge_asr_segments(asr_result: dict, turns: list[dict]) -> list[Segment]:
    out: list[Segment] = []
    for i, seg in enumerate(asr_result.get("segments", [])):
        words = [
            Word(
                word=(word.get("word") or "").strip(),
                start=float(word.get("start", 0.0)),
                end=float(word.get("end", 0.0)),
                probability=float(word.get("probability", 0.0)),
            )
            for word in (seg.get("words") or [])
        ]
        for word in words:
            word.speaker = assign_speaker(word.start, word.end, turns)

        if words:
            counts: dict[int, int] = {}
            for word in words:
                if word.speaker >= 0:
                    counts[word.speaker] = counts.get(word.speaker, 0) + 1
            speaker = max(counts, key=counts.get) if counts else -1
        else:
            speaker = assign_speaker(float(seg["start"]), float(seg["end"]), turns)

        out.append(
            Segment(
                id=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=(seg.get("text") or "").strip(),
                speaker=speaker,
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                words=words,
            )
        )
    return out


def merge_turns(turns: list[dict], max_gap: float = 0.5) -> list[dict]:
    """Merge consecutive same-speaker turns separated by a short gap."""
    merged: list[dict] = []
    for turn in turns:
        if (
            merged
            and merged[-1]["speaker"] == turn["speaker"]
            and float(turn["start"]) - float(merged[-1]["end"]) < max_gap
        ):
            merged[-1]["end"] = float(turn["end"])
        else:
            merged.append(
                {
                    "start": float(turn["start"]),
                    "end": float(turn["end"]),
                    "speaker": int(turn["speaker"]),
                }
            )
    return merged


def select_speakers(
    turns: list[dict],
    num_speakers: Optional[int] = None,
    *,
    min_activity_sec: float = 5.0,
) -> list[dict]:
    """Suppress tiny Sortformer channels and remap retained speakers to dense IDs."""
    if not turns:
        return []

    durations: dict[int, float] = defaultdict(float)
    first_seen: dict[int, float] = {}
    for turn in turns:
        speaker = int(turn["speaker"])
        durations[speaker] += max(0.0, float(turn["end"]) - float(turn["start"]))
        first_seen[speaker] = min(
            first_seen.get(speaker, float("inf")), float(turn["start"])
        )

    ranked = sorted(durations, key=lambda speaker: (-durations[speaker], speaker))
    if num_speakers is not None:
        if num_speakers < 1 or num_speakers > 4:
            raise ValueError("num_speakers must be between 1 and 4")
        if len(ranked) < num_speakers:
            raise ValueError(
                f"requested {num_speakers} speakers, but only {len(ranked)} active channels were detected"
            )
        selected = ranked[:num_speakers]
    else:
        selected = [speaker for speaker in ranked if durations[speaker] >= min_activity_sec]
        if not selected:
            selected = ranked[:1]

    selected.sort(key=lambda speaker: (first_seen[speaker], speaker))
    remap = {speaker: dense_id for dense_id, speaker in enumerate(selected)}
    return [
        {
            "start": float(turn["start"]),
            "end": float(turn["end"]),
            "speaker": remap[int(turn["speaker"])],
        }
        for turn in turns
        if int(turn["speaker"]) in remap
    ]


# Compatibility aliases for callers/tests that imported private helpers from core.py.
_assign_speaker = assign_speaker
_merge_segments = merge_asr_segments
_merge_turns = merge_turns
_select_speakers = select_speakers
