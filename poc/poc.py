#!/usr/bin/env python3
"""
PoC: MLX Whisper (large-v3-turbo) + MLX Sortformer speaker diarization
       -> unified JSON / YAML / markdown with per-speaker, per-word segments.

This is the GATE. If transcription is accurate and both synthetic speakers are
correctly separated and labeled, we move to Stage 2 (native macOS HITL dialog)
and only then to Stage 3 (the full wrapper CLI).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List

# IntervalTree via the stdlib `bisect`-free fallback. We avoid the external
# `intervaltree` dep for the PoC; the algorithm below is the same logic as
# whisperx.assign_word_speakers (exact time-intersection per word -> dominant
# speaker), implemented with a flat list (n is tiny here).

# -----------------------------------------------------------------------------
# Config: models chosen for the 5GB-disk / M3 Max target (see plan).
#   - ASR: Whisper large-v3-turbo, mlx-community fp16 (~1.5GB), fastest accurate Whisper.
#   - DIAR: Sortformer 4-spk v1, mlx-community fp16 (~400MB), MLX-native.
#     (CC-BY-NC; fine for PoC. Commercial swap -> MIT pyannote-seg-MLX + WeSpeaker.)
# -----------------------------------------------------------------------------
DEFAULT_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_DIAR_MODEL = "mlx-community/diar_sortformer_4spk-v1-fp16"


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float
    speaker: int = -1  # filled in by merge step; -1 = unassigned


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: int
    words: List[Word] = field(default_factory=list)


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Exact time-intersection of [a_start,a_end] and [b_start,b_end]."""
    inter = min(a_end, b_end) - max(a_start, b_start)
    return inter if inter > 0 else 0.0


def assign_speaker(
    t_start: float,
    t_end: float,
    diar_segments: List[dict],
) -> int:
    """Pick the speaker whose turns overlap [t_start,t_end] the most.
    Direct port of the dominant-intersection logic in whisperX's
    assign_word_speakers (no midpoint heuristic).
    """
    best_speaker, best_dur = -1, 0.0
    for d in diar_segments:
        dur = overlap_duration(t_start, t_end, d["start"], d["end"])
        if dur > best_dur:
            best_dur, best_speaker = dur, d["speaker"]
    return best_speaker


def run_asr(audio_path: str, model: str, language: str | None) -> dict:
    import mlx_whisper

    print(f"[asr] loading {model} ...", flush=True)
    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        word_timestamps=True,
        language=language,  # None -> auto-detect
    )
    print(f"[asr] done in {time.time()-t0:.2f}s", flush=True)
    return result


def run_diar(audio_path: str, model_id: str) -> List[dict]:
    from mlx_audio.vad import load

    print(f"[diar] loading {model_id} ...", flush=True)
    t0 = time.time()
    model = load(model_id)
    out = model.generate(audio_path, verbose=True)
    print(f"[diar] done in {time.time()-t0:.2f}s", flush=True)

    segs = [
        {"start": s.start, "end": s.end, "speaker": int(s.speaker)}
        for s in out.segments
    ]
    # Merge consecutive same-speaker turns (cleaner output).
    merged: List[dict] = []
    for s in segs:
        if merged and merged[-1]["speaker"] == s["speaker"] and \
           s["start"] - merged[-1]["end"] < 0.5:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    return merged


def merge(asr: dict, diar: List[dict]) -> List[Segment]:
    """Join Whisper word timestamps with Sortformer speaker turns."""
    segments: List[Segment] = []
    for i, seg in enumerate(asr.get("segments", [])):
        words_raw = seg.get("words") or []
        words = [
            Word(
                word=w.get("word", "").strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                probability=float(w.get("probability", 0.0)),
            )
            for w in words_raw
        ]
        for w in words:
            w.speaker = assign_speaker(w.start, w.end, diar)
        # Segment speaker = dominant per-word speaker; fallback to seg-level.
        if words:
            counts: dict[int, int] = {}
            for w in words:
                if w.speaker >= 0:
                    counts[w.speaker] = counts.get(w.speaker, 0) + 1
            seg_speaker = max(counts, key=counts.get) if counts else -1
        else:
            seg_speaker = assign_speaker(seg["start"], seg["end"], diar)

        segments.append(
            Segment(
                id=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg.get("text", "").strip(),
                speaker=seg_speaker,
                words=words,
            )
        )
    return segments


def to_json(segments: List[Segment], diar, audio, asr_model, diar_model) -> str:
    payload = {
        "audio": audio,
        "models": {"asr": asr_model, "diarization": diar_model},
        "speakers": sorted({s.speaker for s in segments if s.speaker >= 0}),
        "num_speakers": len({s.speaker for s in segments if s.speaker >= 0}),
        "diarization_turns": diar,
        "segments": [
            {
                **asdict(seg),
                "words": [asdict(w) for w in seg.words],
            }
            for seg in segments
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def to_yaml(segments: List[Segment], diar, audio, asr_model, diar_model) -> str:
    try:
        import yaml  # PyYAML ships with mlx-audio deps
    except ImportError:
        return "# PyYAML not installed\n"
    payload = {
        "audio": audio,
        "models": {"asr": asr_model, "diarization": diar_model},
        "speakers": sorted({s.speaker for s in segments if s.speaker >= 0}),
        "num_speakers": len({s.speaker for s in segments if s.speaker >= 0}),
        "diarization_turns": diar,
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "words": [asdict(w) for w in seg.words],
            }
            for seg in segments
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def to_markdown(segments: List[Segment], diar, audio) -> str:
    lines = [f"# Transcript: `{audio}`", ""]
    cur = -1
    for seg in segments:
        if seg.speaker != cur:
            cur = seg.speaker
            lines.append(f"\n## Speaker {cur}  ({seg.start:.2f}s)\n")
        lines.append(f"> {seg.text}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    p.add_argument("--diar-model", default=DEFAULT_DIAR_MODEL)
    p.add_argument("--language", default=None, help="e.g. 'en'; omit for auto-detect")
    p.add_argument("--out", default="poc/result.json", help="JSON output path")
    p.add_argument("--yaml", default=None, help="optional YAML output path")
    args = p.parse_args()

    t_total = time.time()
    asr = run_asr(args.audio, args.asr_model, args.language)
    diar = run_diar(args.audio, args.diar_model)
    segments = merge(asr, diar)
    elapsed = time.time() - t_total

    # --- Write artifacts ---
    json_str = to_json(segments, diar, args.audio, args.asr_model, args.diar_model)
    with open(args.out, "w") as f:
        f.write(json_str)
    if args.yaml:
        with open(args.yaml, "w") as f:
            f.write(to_yaml(segments, diar, args.audio, args.asr_model, args.diar_model))

    # --- Human-readable verification block ---
    n_spk = len({s.speaker for s in segments if s.speaker >= 0})
    print("\n" + "=" * 70)
    print("POC RESULT")
    print("=" * 70)
    print(f"Audio            : {args.audio}")
    print(f"ASR model        : {args.asr_model}")
    print(f"Diarization model: {args.diar_model}")
    print(f"Speakers detected: {n_spk}")
    print(f"Diarization turns: {len(diar)}")
    print(f"Total wall time  : {elapsed:.2f}s")
    print("-" * 70)
    for seg in segments:
        tag = f"SPEAKER_{seg.speaker}" if seg.speaker >= 0 else "UNKNOWN"
        print(f"[{seg.start:6.2f}-{seg.end:6.2f}] {tag:>11} | {seg.text}")
    print("-" * 70)
    print(f"JSON written to  : {args.out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
