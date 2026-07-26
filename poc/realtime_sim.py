#!/usr/bin/env python3
"""Offline validation of the realtime core: feed a file's audio in 1s chunks
through run_realtime(). This is the same code path the mic will use."""
import sys, os, time, numpy as np
from mlx_audio.stt import load as load_stt
from mlx_audio.stt.utils import load_audio
from mlx_audio.vad import load as load_vad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_core import SR, run_realtime, EOS

ASR="openai/whisper-large-v3-turbo"; DIAR="mlx-community/diar_sortformer_4spk-v1-fp16"
CLIP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"samples","clip2_pitchshift.wav")

audio=np.array(load_audio(CLIP),dtype=np.float32)
dur=len(audio)/SR
print(f"[sim] {CLIP} {dur:.2f}s",file=sys.stderr)
print("[load] ASR",file=sys.stderr); asr_model=load_stt(ASR)
print("[load] DIAR",file=sys.stderr); diar_model=load_vad(DIAR)

# chunk source: pre-slice the clip into 1s chunks, deliver in real time
chunk_samples=int(SR*1.0)
chunks=[audio[i:i+chunk_samples] for i in range(0,len(audio),chunk_samples)]
idx={"i":0}
def source(timeout_s, state):
    if idx["i"]>=len(chunks): return EOS
    c=chunks[idx["i"]]; idx["i"]+=1
    time.sleep(1.0)  # deliver at 1s cadence = realtime simulation
    return c

t0=time.time()
res=run_realtime(asr_model=asr_model, diar_model=diar_model,
                 chunk_source=source, language="en",
                 asr_chunk_sec=1.0, diar_chunk_sec=5.0, verbose=True)
wall=time.time()-t0
full=res.text.lower()
ok_asr="alpha" in full and "bravo" in full
ok_diar=len(res.speakers)>=2
print(f"\n[sim] wall={wall:.2f}s (audio {dur:.2f}s) ASR_ok={ok_asr} DIAR_ok={ok_diar} "
      f"spks={res.speakers}",file=sys.stderr)
print(f"[sim] text: {res.text!r}",file=sys.stderr)
sys.exit(0 if (ok_asr and ok_diar) else 1)
