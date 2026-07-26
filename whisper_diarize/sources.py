"""Audio source abstraction: file, stdin, microphone, BlackHole system loopback.

Each source yields 16 kHz mono float32 numpy chunks (or EOS when finite).
A mic/BlackHole source yields forever until stopped via a signal/event.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from typing import Generator, Optional, Union

import numpy as np

SR = 16000  # all sources deliver at the model sample rate


class EOS:
    """End-of-stream sentinel (singleton)."""
    _inst = None
    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst
    def __repr__(self): return "EOS"
    def __bool__(self): return False


EOS = EOS()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# File / pipe sources (finite) -- deliver the whole clip in chunks
# ---------------------------------------------------------------------------

def file_source(path: str, chunk_sec: float = 1.0) -> Generator[np.ndarray, None, None]:
    """Yield a file's audio as 16k mono float32 chunks of `chunk_sec` seconds."""
    from mlx_audio.stt.utils import load_audio
    audio = np.array(load_audio(path), dtype=np.float32)
    n = int(SR * chunk_sec)
    for i in range(0, len(audio), n):
        yield audio[i:i + n]


def stdin_source(chunk_sec: float = 1.0) -> Generator[np.ndarray, None, None]:
    """Yield audio read from stdin as 16k mono float32 chunks.

    Accepts raw PCM s16le mono 16k (the simplest pipe format) or WAV.
    Detection: peek first 4 bytes for 'RIFF' -> WAV via miniaudio; else raw s16le.
    """
    raw = sys.stdin.buffer.read()
    if raw[:4] == b"RIFF":
        # write to a temp file and use the file loader (handles all formats)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            yield from file_source(tmp, chunk_sec)
        finally:
            os.unlink(tmp)
    else:
        # raw s16le mono 16k
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        n = int(SR * chunk_sec)
        for i in range(0, len(samples), n):
            yield samples[i:i + n]


# ---------------------------------------------------------------------------
# Live sources (mic / BlackHole) -- infinite until stopped
# ---------------------------------------------------------------------------

def _resample_to_16k(data: np.ndarray, in_rate: int) -> np.ndarray:
    """Resample float32 mono audio from `in_rate` to 16k."""
    if in_rate == SR:
        return data.astype(np.float32)
    # ratio-based: use scipy for arbitrary rates
    from scipy.signal import resample_poly, firwin, lfilter
    from math import gcd
    g = gcd(in_rate, SR)
    up, down = SR // g, in_rate // g
    # anti-alias lowpass at min(in_rate, SR)/2
    b = firwin(15, 0.8 * min(up, down) / max(up, down))
    filt = lfilter(b, [1.0], data)
    return resample_poly(filt, up, down).astype(np.float32)


class LiveAudioSource:
    """Threaded sounddevice capture -> 16k mono chunks. Backs mic + BlackHole.

    `device` selects the input:
      - None  : system default input (built-in mic)
      - int   : sounddevice device index (e.g. the BlackHole index)
      - str   : case-insensitive substring match against device names
                ("blackhole", "teams", etc.) -- resolved at start()
    """
    def __init__(self, device: Optional[Union[int, str]] = None,
                 in_rate: int = 48000, chunk_sec: float = 1.0,
                 stop_event: Optional[threading.Event] = None):
        self.device_spec = device
        self.in_rate = in_rate
        self.chunk_sec = chunk_sec
        self.stop_event = stop_event or threading.Event()
        self._buf: deque = deque(maxlen=in_rate * 30)
        self._lock = threading.Lock()
        import sounddevice as sd
        self._sd = sd
        self._stream = None
        self._resolved_index: Optional[int] = None

    def _resolve_device(self) -> Optional[int]:
        if self.device_spec is None:
            return None
        if isinstance(self.device_spec, int):
            return self.device_spec
        # str: match by name
        needle = self.device_spec.lower()
        for i, d in enumerate(self._sd.query_devices()):
            if d["max_input_channels"] > 0 and needle in d["name"].lower():
                # update in_rate to match the device default for fidelity
                self.in_rate = int(d["default_samplerate"])
                return i
        raise ValueError(f"no input device matching {self.device_spec!r}")

    def _callback(self, indata, frames, time_info, status):
        if status:
            sys.stderr.write(f"[audio] {status}\n")
        with self._lock:
            self._buf.extend(indata[:, 0].tolist())

    def start(self):
        self._resolved_index = self._resolve_device()
        self._stream = self._sd.InputStream(
            samplerate=self.in_rate, channels=1, dtype="float32",
            blocksize=int(self.in_rate * self.chunk_sec),
            callback=self._callback, device=self._resolved_index,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass

    def chunks(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding 16k mono float32 chunks until stop_event is set."""
        min_samples = int(self.in_rate * self.chunk_sec)
        while not self.stop_event.is_set():
            with self._lock:
                if len(self._buf) < min_samples:
                    data = None
                else:
                    data = np.array(self._buf, dtype=np.float32)
                    self._buf.clear()
            if data is None:
                time.sleep(0.05)
                continue
            yield _resample_to_16k(data, self.in_rate)


def list_input_devices() -> list:
    """Return a list of (index, name, channels, default_rate) for input devices."""
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], d["max_input_channels"], d["default_samplerate"]))
    return out
