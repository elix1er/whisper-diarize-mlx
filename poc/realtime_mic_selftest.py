#!/usr/bin/env python3
"""
Self-test for realtime_mic.py: monkeypatch the MicSource to feed the proven
2-speaker clip (as 48k samples that then get resampled) instead of the real
microphone. This exercises the EXACT production code path (resample, ring
buffer, core) with known audio and a known expected result, so we can confirm
the live mic harness is correct without needing a human speaker.

If this passes, realtime_mic.py is correct; only the physical mic + macOS
permission remain to be validated by a real human run.
"""
import sys, os, time, threading, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import realtime_mic as rm
from realtime_core import SR, EOS
from mlx_audio.stt import load as load_stt
from mlx_audio.stt.utils import load_audio
from mlx_audio.vad import load as load_vad

CLIP = os.path.join(HERE, "samples", "clip2_pitchshift.wav")

# Load clip at 16k, upsample to 48k to mimic raw mic input
audio_16k = np.array(load_audio(CLIP), dtype=np.float32)
# simple 1:3 upsample (zero-stuff + the same lowpass the resampler reverses)
audio_48k = np.zeros(len(audio_16k)*3, dtype=np.float32)
audio_48k[::3] = audio_16k
# mild lowpass to avoid harsh images (good enough for a path test)
from scipy.signal import firwin, lfilter
b = firwin(15, 7.5e3/(rm.SR_IN/2))
audio_48k = lfilter(b, [1.0], audio_48k).astype(np.float32)

print(f"[selftest] clip {CLIP} -> {len(audio_16k)/SR:.2f}s @16k, {len(audio_48k)/rm.SR_IN:.2f}s @48k",
      file=sys.stderr)

class FakeMic(rm.MicSource):
    """Pretends to be the mic by dripping the clip's 48k samples in 1s blocks."""
    def __init__(self):
        super().__init__()
        self._idx = 0
        self._block = int(rm.SR_IN * 1.0)
        self._data = audio_48k
    def start(self):
        pass  # no real stream
    def stop(self):
        pass
    def drain_16k(self, min_seconds=1.0):
        if self._idx >= len(self._data):
            return None
        block = self._data[self._idx:self._idx+self._block]
        self._idx += self._block
        time.sleep(1.0)  # realtime cadence
        return rm.resample_48k_to_16k(block)

# monkeypatch
rm.MicSource = FakeMic

print("[load] ASR", file=sys.stderr); asr_model = load_stt(rm.DEFAULT_ASR)
print("[load] DIAR", file=sys.stderr); diar_model = load_vad(rm.DEFAULT_DIAR)

mic = FakeMic()
stop_evt = threading.Event()

def source(timeout_s, state):
    if stop_evt.is_set():
        return EOS
    chunk = mic.drain_16k(min_seconds=1.0)
    if chunk is None:
        return EOS  # clip exhausted
    return chunk

from realtime_core import run_realtime
t0 = time.time()
res = run_realtime(asr_model=asr_model, diar_model=diar_model,
                   chunk_source=source, language="en",
                   asr_chunk_sec=1.0, diar_chunk_sec=5.0, verbose=False)
wall = time.time()-t0
full = res.text.lower()
ok_asr = "alpha" in full and "bravo" in full
ok_diar = len(res.speakers) >= 2
print(f"\n[selftest] wall={wall:.2f}s ASR_ok={ok_asr} DIAR_ok={ok_diar} spks={res.speakers}",
      file=sys.stderr)
print(f"[selftest] text: {res.text[:160]!r}", file=sys.stderr)
sys.exit(0 if (ok_asr and ok_diar) else 1)
