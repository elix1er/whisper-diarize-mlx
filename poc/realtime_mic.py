#!/usr/bin/env python3
"""
Live microphone: MLX streaming ASR + streaming diarization -> NDJSON.

Reuses the validated realtime_core.run_realtime(). Audio path:
  sounddevice (48k mono) -> ring buffer -> resample 16k -> core.

Output: NDJSON to stdout (one JSON object per event). Ctrl-C to stop.

NOTE on macOS microphone permission:
  On first run, macOS shows a system dialog asking to grant the Terminal/
  Python binary microphone access. If denied, sounddevice.open raises
  PortAudioError. Grant once; subsequent runs are silent.

NOTE on capturing macOS system audio (the other side of a call):
  Two options for capturing what the Mac is *playing* (not the mic):
    A) ScreenCaptureKit (macOS 13.3+, built in). Can capture any app's audio
       output natively — no kernel extension, no virtual device. Requires
       a Swift/ObjC helper or the `pyav`+SCStream route. Best long-term.
    B) BlackHole (virtual loopback). brew install --cask blackhole, then set
       a multi-output device in Audio MIDI Setup. Simplest today.
  This file implements the MIC path (sufficient for online calls per your
  note). System-audio capture is wired behind the same core by swapping the
  chunk_source.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

from realtime_core import SR, run_realtime

SR_IN = 48000
DEFAULT_ASR = "openai/whisper-large-v3-turbo"
DEFAULT_DIAR = "mlx-community/diar_sortformer_4spk-v1-fp16"


def resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    from scipy.signal import firwin, lfilter
    b = firwin(15, 7.5e3 / (SR_IN / 2))
    return lfilter(b, [1.0], x)[2::3].astype(np.float32)


class MicSource:
    """sounddevice -> 16k mono float32 chunks. Thread-safe ring buffer."""
    def __init__(self, in_device=None):
        self._buf: deque[float] = deque(maxlen=SR_IN * 30)
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._in_device = in_device
        self._overflow_logged = False

    def _cb(self, indata, frames, time_info, status):
        if status and status.input_overflow and not self._overflow_logged:
            sys.stderr.write("[audio] input overflow (CPU slow?)\n")
            self._overflow_logged = True
        with self._lock:
            self._buf.extend(indata[:, 0].tolist())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SR_IN, channels=1, dtype='float32',
            blocksize=int(SR_IN * 1.0), callback=self._cb,
            device=self._in_device,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop(); self._stream.close()

    def drain_16k(self, min_seconds: float = 1.0):
        min_samples = int(min_seconds * SR_IN)
        with self._lock:
            if len(self._buf) < min_samples:
                return None
            data = np.array(self._buf, dtype=np.float32)
            self._buf.clear()
        return resample_48k_to_16k(data)


def main() -> int:
    p = argparse.ArgumentParser(description="Realtime MLX mic ASR + diarization -> NDJSON")
    p.add_argument("--asr-model", default=DEFAULT_ASR)
    p.add_argument("--diar-model", default=DEFAULT_DIAR)
    p.add_argument("--language", default="en")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="auto-stop after N seconds (0 = run until Ctrl-C)")
    p.add_argument("--no-diar", action="store_true")
    p.add_argument("--list-devices", action="store_true",
                   help="list input devices and exit")
    p.add_argument("--device", type=int, default=None, help="input device index")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']}  in={d['max_input_channels']} sr={d['default_samplerate']}")
        return 0

    sys.stderr.write(f"[load] ASR  {args.asr_model}\n")
    from mlx_audio.stt import load as load_stt
    asr_model = load_stt(args.asr_model)

    diar_model = None
    if not args.no_diar:
        sys.stderr.write(f"[load] DIAR {args.diar_model}\n")
        from mlx_audio.vad import load as load_vad
        diar_model = load_vad(args.diar_model)

    mic = MicSource(in_device=args.device)
    mic.start()
    sys.stderr.write("[mic] streaming — speak now (Ctrl-C to stop)\n")

    stop_evt = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_evt.set())

    from realtime_core import EOS

    def source(timeout_s, state):
        if stop_evt.is_set():
            return EOS
        chunk = mic.drain_16k(min_seconds=1.0)
        if chunk is None:
            time.sleep(0.05)
            return None
        return chunk

    try:
        run_realtime(
            asr_model=asr_model, diar_model=diar_model,
            chunk_source=source, language=args.language,
            asr_chunk_sec=1.0, diar_chunk_sec=5.0,
            max_seconds=args.seconds, verbose=args.verbose,
        )
    finally:
        mic.stop()
        sys.stderr.write("[mic] stopped\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
