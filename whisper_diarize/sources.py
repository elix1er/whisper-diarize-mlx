"""Live audio capture helpers for microphone and loopback devices."""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Generator, Optional, Union

import numpy as np

SR = 16000


class _EOS:
    def __repr__(self) -> str:
        return "EOS"

    def __bool__(self) -> bool:
        return False


EOS = _EOS()


def _resample_to_16k(data: np.ndarray, in_rate: int) -> np.ndarray:
    """Resample float32 mono audio to the model sample rate."""
    if in_rate == SR:
        return np.asarray(data, dtype=np.float32)

    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(in_rate, SR)
    up, down = SR // divisor, in_rate // divisor
    return resample_poly(data, up, down).astype(np.float32)


class LiveAudioSource:
    """Threaded sounddevice capture yielding 16 kHz mono float32 chunks."""

    def __init__(
        self,
        device: Optional[Union[int, str]] = None,
        in_rate: int = 48000,
        chunk_sec: float = 1.0,
        stop_event: Optional[threading.Event] = None,
    ):
        self.device_spec = device
        self.in_rate = in_rate
        self.chunk_sec = chunk_sec
        self.stop_event = stop_event or threading.Event()
        self._buf: deque[float] = deque(maxlen=in_rate * 30)
        self._lock = threading.Lock()
        import sounddevice as sd

        self._sd = sd
        self._stream = None
        self._resolved_index: Optional[int] = None

    def _resolve_device(self) -> Optional[int]:
        if self.device_spec is None:
            return None
        if isinstance(self.device_spec, int):
            info = self._sd.query_devices(self.device_spec)
            self.in_rate = int(info["default_samplerate"])
            return self.device_spec

        needle = self.device_spec.lower()
        for index, device in enumerate(self._sd.query_devices()):
            if device["max_input_channels"] > 0 and needle in device["name"].lower():
                self.in_rate = int(device["default_samplerate"])
                return index
        raise ValueError(f"no input device matching {self.device_spec!r}")

    def _callback(self, indata, frames, time_info, status):
        if status:
            sys.stderr.write(f"[audio] {status}\n")
        with self._lock:
            self._buf.extend(indata[:, 0].tolist())

    def start(self) -> None:
        self._resolved_index = self._resolve_device()
        self._buf = deque(maxlen=self.in_rate * 30)
        self._stream = self._sd.InputStream(
            samplerate=self.in_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.in_rate * self.chunk_sec),
            callback=self._callback,
            device=self._resolved_index,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def chunks(self) -> Generator[np.ndarray, None, None]:
        min_samples = int(self.in_rate * self.chunk_sec)
        while not self.stop_event.is_set():
            with self._lock:
                if len(self._buf) < min_samples:
                    data = None
                else:
                    data = np.asarray(self._buf, dtype=np.float32)
                    self._buf.clear()
            if data is None:
                time.sleep(0.05)
                continue
            yield _resample_to_16k(data, self.in_rate)


def list_input_devices() -> list[tuple[int, str, int, float]]:
    import sounddevice as sd

    return [
        (index, device["name"], device["max_input_channels"], device["default_samplerate"])
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]
