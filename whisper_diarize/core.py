"""Core engine: batch transcription + merge, and streaming transcription.

Public:
  transcribe(path_or_array, ...) -> DiarizationResult   (batch)
  transcribe_stream(source, ...) -> generator of events  (live)
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Generator, List, Optional, Union

import numpy as np

from .sources import SR, EOS

# Model defaults chosen for the M3 Max / 5GB-disk target.
DEFAULT_ASR_BATCH = "mlx-community/whisper-large-v3-turbo"      # weights-only port
DEFAULT_ASR_STREAM = "openai/whisper-large-v3-turbo"           # needs processor
DEFAULT_DIAR = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
DEFAULT_DIAR_CHUNK_SEC = 5.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

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
    words: List[Word] = field(default_factory=list)


@dataclass
class DiarizationResult:
    text: str
    segments: List[Segment]
    speakers: List[int]
    diarization_turns: List[dict]
    language: Optional[str] = None
    models: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Merge logic (port of whisperX assign_word_speakers)
# ---------------------------------------------------------------------------

def _overlap(a0, a1, b0, b1) -> float:
    inter = min(a1, b1) - max(a0, b0)
    return inter if inter > 0 else 0.0


def _assign_speaker(t0, t1, turns: List[dict]) -> int:
    best_spk, best_dur = -1, 0.0
    for t in turns:
        d = _overlap(t0, t1, t["start"], t["end"])
        if d > best_dur:
            best_dur, best_spk = d, t["speaker"]
    if best_spk >= 0 or not turns:
        return best_spk
    midpoint = (t0 + t1) / 2.0
    nearest = min(
        turns,
        key=lambda turn: abs(
            (float(turn["start"]) + float(turn["end"])) / 2.0 - midpoint
        ),
    )
    return int(nearest["speaker"])


def _merge_segments(asr_result: dict, turns: List[dict]) -> List[Segment]:
    out: List[Segment] = []
    for i, seg in enumerate(asr_result.get("segments", [])):
        words = [
            Word(
                word=(w.get("word") or "").strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                probability=float(w.get("probability", 0.0)),
            )
            for w in (seg.get("words") or [])
        ]
        for w in words:
            w.speaker = _assign_speaker(w.start, w.end, turns)
        if words:
            counts: dict = {}
            for w in words:
                if w.speaker >= 0:
                    counts[w.speaker] = counts.get(w.speaker, 0) + 1
            seg_spk = max(counts, key=counts.get) if counts else -1
        else:
            seg_spk = _assign_speaker(float(seg["start"]), float(seg["end"]), turns)
        out.append(Segment(
            id=i, start=float(seg["start"]), end=float(seg["end"]),
            text=(seg.get("text") or "").strip(), speaker=seg_spk,
            no_speech_prob=float(seg.get("no_speech_prob", 0.0)), words=words,
        ))
    return out


def _merge_turns(turns: List[dict], max_gap: float = 0.5) -> List[dict]:
    """Merge consecutive same-speaker turns within `max_gap`."""
    merged: List[dict] = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"] \
                and t["start"] - merged[-1]["end"] < max_gap:
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    return merged


def _select_speakers(
    turns: List[dict],
    num_speakers: Optional[int] = None,
    *,
    min_activity_sec: float = 5.0,
    min_activity_share: float = 0.01,
) -> List[dict]:
    """Drop spurious channels and remap retained speakers to dense IDs.

    Sortformer always exposes four output channels. A few isolated activations
    on an otherwise unused channel must not be reported as extra people. When
    the speaker count is known, ``num_speakers`` is authoritative. Otherwise,
    channels need both a small absolute and relative amount of speech.
    """
    if not turns:
        return []
    durations: dict[int, float] = defaultdict(float)
    first_seen: dict[int, float] = {}
    for turn in turns:
        speaker = int(turn["speaker"])
        durations[speaker] += max(0.0, float(turn["end"]) - float(turn["start"]))
        first_seen[speaker] = min(first_seen.get(speaker, float("inf")), float(turn["start"]))

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
        total = sum(durations.values())
        floor = max(min_activity_sec, total * min_activity_share)
        selected = [speaker for speaker in ranked if durations[speaker] >= floor]
        if not selected:
            selected = ranked[:1]

    selected.sort(key=lambda speaker: (first_seen[speaker], speaker))
    remap = {speaker: dense_id for dense_id, speaker in enumerate(selected)}
    return [
        {"start": float(turn["start"]), "end": float(turn["end"]),
         "speaker": remap[int(turn["speaker"])]}
        for turn in turns
        if int(turn["speaker"]) in remap
    ]


# ---------------------------------------------------------------------------
# Batch: transcribe()
# ---------------------------------------------------------------------------

def transcribe(
    audio: Union[str, np.ndarray],
    *,
    asr_model: str = DEFAULT_ASR_BATCH,
    diar_model: str = DEFAULT_DIAR,
    language: Optional[str] = None,
    no_diar: bool = False,
    word_timestamps: bool = True,
    diar_threshold: float = 0.5,
    diar_chunk_sec: float = DEFAULT_DIAR_CHUNK_SEC,
    num_speakers: Optional[int] = None,
    verbose: bool = False,
) -> DiarizationResult:
    """Batch transcribe + diarize an audio file or 16k mono array.

    Returns a DiarizationResult with merged per-speaker, per-word segments.
    """
    if verbose:
        print(f"[whisper_diarize] ASR {asr_model}", flush=True)
    import mlx_whisper
    t0 = time.time()
    asr = mlx_whisper.transcribe(
        audio, path_or_hf_repo=asr_model, word_timestamps=word_timestamps,
        language=language,
    )
    if verbose:
        print(f"[whisper_diarize] ASR done {time.time()-t0:.2f}s", flush=True)

    turns: List[dict] = []
    if not no_diar:
        if verbose:
            print(f"[whisper_diarize] DIAR {diar_model}", flush=True)
        from mlx_audio.vad import load as load_vad
        d0 = time.time()
        m = load_vad(diar_model)
        raw_turns: List[dict] = []
        for out in m.generate_stream(
            audio,
            chunk_duration=diar_chunk_sec,
            threshold=diar_threshold,
            min_duration=0.1,
            merge_gap=0.1,
            verbose=verbose,
        ):
            raw_turns.extend(
                {"start": s.start, "end": s.end, "speaker": int(s.speaker)}
                for s in out.segments
            )
        turns = _merge_turns(_select_speakers(raw_turns, num_speakers))
        if verbose:
            print(f"[whisper_diarize] DIAR done {time.time()-d0:.2f}s "
                  f"({len(turns)} turns)", flush=True)

    segments = _merge_segments(asr, turns)
    speakers = sorted({s.speaker for s in segments if s.speaker >= 0})
    return DiarizationResult(
        text=asr.get("text", "").strip(),
        segments=segments, speakers=speakers, diarization_turns=turns,
        language=asr.get("language"),
        models={"asr": asr_model, "diarization": None if no_diar else diar_model},
    )


# ---------------------------------------------------------------------------
# Streaming: transcribe_stream()
# ---------------------------------------------------------------------------

def transcribe_stream(
    chunk_source: Callable[[], Optional[np.ndarray]],
    *,
    asr_model: str = DEFAULT_ASR_STREAM,
    diar_model: str = DEFAULT_DIAR,
    language: str = "en",
    no_diar: bool = False,
    asr_chunk_sec: float = 1.0,
    diar_chunk_sec: float = 5.0,
    max_seconds: float = 0.0,
    verbose: bool = False,
) -> Generator[dict, None, None]:
    """Stream ASR (+ diarization) from a chunk source -> NDJSON dicts.

    chunk_source() -> np.ndarray (16k mono) | None (no data yet) | EOS (done).
    Yields event dicts:
      {"type":"status","msg":...}
      {"type":"asr_partial","t":..,"text":..,"is_final":bool}
      {"type":"diar","start":..,"end":..,"speaker":int}
      {"type":"error","stage":..,"msg":..}
      {"type":"final","text":..,"diarization":[...],"speakers":[...],"duration_s":..}
    """
    from mlx_audio.stt import load as load_stt
    if verbose:
        print(f"[whisper_diarize] ASR(stream) {asr_model}", flush=True)
    asr = load_stt(asr_model)
    from mlx_audio.stt.models.whisper.streaming import StreamingConfig, StreamingDecoder
    decoder = StreamingDecoder(asr, StreamingConfig(frame_threshold=25),
                               language=language, task="transcribe")
    n_mels = asr.dims.n_mels

    diar_m = None
    diar_state = None
    if not no_diar:
        if verbose:
            print(f"[whisper_diarize] DIAR(stream) {diar_model}", flush=True)
        from mlx_audio.vad import load as load_vad
        diar_m = load_vad(diar_model)
        diar_state = diar_m.init_streaming_state()

    from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram

    yield {"type": "status", "msg": "listening"}

    acc_diar = np.zeros(0, dtype=np.float32)
    diar_off = 0.0
    diar_n = int(SR * diar_chunk_sec)
    full_text_parts: List[str] = []
    diar_segs: List[dict] = []
    t0 = time.time()
    deadline = (t0 + max_seconds) if max_seconds > 0 else None

    while True:
        if deadline and time.time() > deadline:
            break
        chunk = chunk_source()
        if chunk is EOS:
            break
        if chunk is None or len(chunk) == 0:
            continue
        elapsed = time.time() - t0

        # --- ASR: feed only the new chunk's mel; decoder keeps state ---
        try:
            mel = log_mel_spectrogram(np.array(chunk), n_mels=n_mels)
            res = decoder.decode_chunk(mel, is_last=False)
        except Exception as e:
            yield {"type": "error", "stage": "asr", "msg": str(e)}
            continue
        if res.text.strip():
            full_text_parts.append(res.text)
            yield {"type": "asr_partial", "t": round(elapsed, 2),
                   "text": res.text, "is_final": bool(res.is_final)}
            if verbose:
                print(f"  [asr {elapsed:6.2f}s] {res.text!r}", flush=True)

        # --- DIAR: accumulate and feed every diar_chunk_sec ---
        if diar_m is not None:
            acc_diar = np.concatenate([acc_diar, chunk])
            while len(acc_diar) >= diar_n:
                feed = acc_diar[:diar_n]
                acc_diar = acc_diar[diar_n:]
                try:
                    out, diar_state = diar_m.feed(feed, diar_state)
                except Exception as e:
                    yield {"type": "error", "stage": "diar", "msg": str(e)}
                    break
                for s in out.segments:
                    rec = {"type": "diar",
                           "start": round(s.start + diar_off, 3),
                           "end": round(s.end + diar_off, 3),
                           "speaker": int(s.speaker)}
                    yield rec
                    if verbose:
                        print(f"  [diar {elapsed:6.2f}s] spk{s.speaker} "
                              f"{s.start+diar_off:.2f}-{s.end+diar_off:.2f}", flush=True)
                    diar_segs.append({k: v for k, v in rec.items() if k != "type"})
                diar_off += diar_chunk_sec

    speakers = sorted({d["speaker"] for d in diar_segs})
    yield {"type": "final", "duration_s": round(time.time() - t0, 2),
           "text": " ".join(full_text_parts).strip(),
           "diarization": diar_segs, "speakers": speakers}
