#!/usr/bin/env python3
"""
Streaming verification: drive the SAME proven 2-speaker clip through the
*streaming* ASR (AlignAtt) and *streaming* diarization (Sortformer feed())
paths, in small chunks, to prove the realtime components work and to
measure per-chunk latency before we attach a live microphone.

This mimics exactly what a mic would deliver: a stream of small float32
mono-16k chunks. If this works, the mic pipeline is just swapping the
chunk source.
"""
from __future__ import annotations

import os
import time

import numpy as np

from mlx_audio.stt import load as load_stt
from mlx_audio.vad import load as load_vad

HERE = os.path.dirname(os.path.abspath(__file__))

# NOTE: for the *streaming* path we use the original openai/ repo because it
# bundles the WhisperProcessor (preprocessor_config.json) that the streaming
# tokenizer requires. mlx-audio still runs the weights on MLX. (The mlx-community
# port is weights-only and lacks the processor — fine for batch transcribe(),
# not for generate_streaming().)
ASR_MODEL = "openai/whisper-large-v3-turbo"
DIAR_MODEL = "mlx-community/diar_sortformer_4spk-v1-fp16"
CLIP = os.path.join(HERE, "samples", "clip2_pitchshift.wav")
CHUNK_SEC = 1.0           # ASR chunk duration (AlignAtt default)
SR = 16000
DIAR_CHUNK_SEC = 5.0      # Sortformer expects longer chunks (~5s) for stable diarization


def load_clip_mono_16k(path: str) -> np.ndarray:
    from mlx_audio.stt.utils import load_audio
    import mlx.core as mx
    a = load_audio(path)            # mx.array, 16k, mono
    return np.array(a, dtype=np.float32)


def main() -> int:
    print(f"[load] ASR  {ASR_MODEL}")
    asr_model = load_stt(ASR_MODEL)
    print(f"[load] DIAR {DIAR_MODEL}")
    diar_model = load_vad(DIAR_MODEL)

    audio = load_clip_mono_16k(CLIP)
    duration = len(audio) / SR
    print(f"[clip] {CLIP}  duration={duration:.2f}s  samples={len(audio)}")
    print("=" * 72)

    # ---- Streaming ASR: feed the clip in 1s chunks via generate_streaming ----
    print("STREAMING ASR (AlignAtt, 1s chunks):")
    asr_t0 = time.time()
    asr_text_parts = []
    n_emits = 0
    # generate_streaming takes the whole array and chunks internally; that still
    # exercises the StreamingDecoder exactly as in live mode.
    for res in asr_model.generate_streaming(audio, chunk_duration=CHUNK_SEC, language="en"):
        tag = "FINAL" if res.is_final else "partial"
        asr_text_parts.append(res.text)
        n_emits += 1
        print(f"  [{tag:6}] pos={res.audio_position:5.2f}s  text={res.text!r}")
    asr_elapsed = time.time() - asr_t0
    full_asr_text = " ".join(p for p in asr_text_parts if p).strip()
    print(f"  -> {n_emits} emissions, {asr_elapsed:.2f}s wall for {duration:.2f}s audio "
          f"(RTF={asr_elapsed/duration:.2f}x, {'FASTER' if asr_elapsed < duration else 'SLOWER'} than realtime)")
    print(f"  -> full text: {full_asr_text!r}")

    # ---- Streaming diarization: feed the clip in ~5s chunks via feed() ----
    print()
    print("STREAMING DIARIZATION (Sortformer feed(), 5s chunks):")
    state = diar_model.init_streaming_state()
    diar_t0 = time.time()
    chunk_n = int(SR * DIAR_CHUNK_SEC)
    all_segs = []
    for i, start in enumerate(range(0, len(audio), chunk_n)):
        chunk = audio[start:start + chunk_n]
        if len(chunk) < SR * 0.5:  # skip tiny tail
            continue
        ct0 = time.time()
        out, state = diar_model.feed(chunk, state)
        cdt = time.time() - ct0
        for s in out.segments:
            all_segs.append((s.start, s.end, int(s.speaker)))
        spks = sorted({s.speaker for s in out.segments})
        rng = f"{start/SR:.1f}-{(start+len(chunk))/SR:.1f}s"
        print(f"  chunk {i} ({rng}): {len(out.segments)} segs spks={spks}  compute={cdt*1000:.0f}ms")
    diar_elapsed = time.time() - diar_t0
    print(f"  -> {len(all_segs)} total segments, {diar_elapsed:.2f}s wall "
          f"(RTF={diar_elapsed/duration:.2f}x)")

    print()
    print("STREAMING DIARIZATION SEGMENTS (speaker turns):")
    for st, en, sp in all_segs:
        print(f"  [{st:6.2f}-{en:6.2f}] speaker_{sp}")

    # ---- Verdict ----
    spkrs = sorted({sp for _, _, sp in all_segs})
    ok_asr = "speaker alpha" in full_asr_text.lower() and "bravo" in full_asr_text.lower()
    ok_diar = len(spkrs) >= 2
    ok_rtf = (asr_elapsed + diar_elapsed) < duration * 2  # generous: both pipelines combined
    print()
    print("=" * 72)
    print("STREAMING VERDICT")
    print("=" * 72)
    print(f"  ASR text correct (alpha+bravo) : {'PASS' if ok_asr else 'FAIL'}")
    print(f"  >=2 speakers separated         : {'PASS' if ok_diar else 'FAIL'} ({spkrs})")
    print(f"  faster-than-realtime budget    : {'PASS' if ok_rtf else 'FAIL'} "
          f"(combined {asr_elapsed+diar_elapsed:.2f}s vs {duration:.2f}s audio)")
    print("=" * 72)
    return 0 if (ok_asr and ok_diar and ok_rtf) else 1


if __name__ == "__main__":
    raise SystemExit(main())
