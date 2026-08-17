"""whisper-diarize CLI.

Modes:
  FILE   : whisper-diarize audio.wav -o json
  PIPE   : cat audio.wav | whisper-diarize -o yaml
  LIVE   : whisper-diarize --live [--source mic|blackhole|<idx>] [--seconds N]
           (streams NDJSON events to stdout)
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from .core import (
    DEFAULT_ASR_BATCH, DEFAULT_ASR_STREAM, DEFAULT_DIAR, DEFAULT_DIAR_CHUNK_SEC,
    transcribe, transcribe_stream,
)
from .formatters import FORMATTERS
from .sources import LiveAudioSource, EOS, file_source, list_input_devices, stdin_source


def _emit(obj: dict):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_lines(text: str):
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whisper-diarize",
        description="MLX Whisper + MLX speaker diarization (Apple Silicon). "
                    "File / pipe / live mic modes.",
    )
    p.add_argument("audio", nargs="?", default=None,
                   help="audio file path (omit to read from stdin, or use --live)")
    p.add_argument("-o", "--output", default="json",
                   choices=list(FORMATTERS) + ["ndjson"],
                   help="output format (default: json; ndjson only valid with --live)")
    p.add_argument("--live", action="store_true",
                   help="live capture mode (streams NDJSON events)")
    p.add_argument("--source", default="mic",
                   help="live source: 'mic' | 'blackhole' | device index | name substring")
    p.add_argument("--list-devices", action="store_true",
                   help="list input devices and exit")

    p.add_argument("--asr-model", default=None,
                   help=f"ASR model (default: {DEFAULT_ASR_BATCH} batch / {DEFAULT_ASR_STREAM} live)")
    p.add_argument("--diar-model", default=DEFAULT_DIAR)
    p.add_argument("--language", default=None, help="language code; omit to auto-detect")
    p.add_argument("--no-diar", action="store_true", help="disable speaker diarization")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="live: auto-stop after N seconds (0 = until Ctrl-C)")
    p.add_argument("--diar-threshold", type=float, default=0.5)
    p.add_argument("--diar-chunk-seconds", type=float, default=DEFAULT_DIAR_CHUNK_SEC,
                   help="batch diarization chunk size (default: 5; state is preserved across chunks)")
    p.add_argument("--num-speakers", type=int, choices=range(1, 5), default=None,
                   help="known speaker count, 1-4; omit to suppress insignificant channels automatically")
    p.add_argument("--verbose", action="store_true")
    return p


def _run_live(args: argparse.Namespace) -> int:
    if args.output != "ndjson" and args.output != "json":
        print("error: --live emits NDJSON; output forced to ndjson", file=sys.stderr)
    # resolve source device
    src_map = {"mic": None, "blackhole": "blackhole"}
    device = src_map.get(args.source, args.source)
    try:
        device_idx = int(args.source)
        device = device_idx
    except (ValueError, TypeError):
        pass

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    src = LiveAudioSource(device=device, stop_event=stop)
    src.start()
    sys.stderr.write(f"[whisper-diarize] live source={args.source!r} "
                     f"device={src._resolved_index} — streaming NDJSON\n")
    try:
        for ev in transcribe_stream(
            chunk_source=lambda: next(src.chunks(), EOS),
            asr_model=args.asr_model or DEFAULT_ASR_STREAM,
            diar_model=args.diar_model, language=args.language or "en",
            no_diar=args.no_diar, max_seconds=args.seconds, verbose=args.verbose,
        ):
            _emit(ev)
    finally:
        src.stop()
        sys.stderr.write("[whisper-diarize] stopped\n")
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    if args.audio is None and sys.stdin.isatty():
        print("error: provide an audio file, pipe via stdin, or use --live",
              file=sys.stderr)
        return 2
    # gather audio
    if args.audio is not None:
        audio_path = args.audio
    else:
        # read stdin to a temp file (supports wav/mp3/etc, not just raw pcm)
        import os, tempfile
        raw = sys.stdin.buffer.read()
        suffix = ".wav" if raw[:4] == b"RIFF" else ".raw"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw)
            audio_path = f.name
        # if raw pcm, wrap into wav via ffmpeg
        if suffix == ".raw":
            import subprocess
            wav_tmp = audio_path + ".wav"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
                 "-i", audio_path, wav_tmp],
                check=True, capture_output=True,
            )
            os.unlink(audio_path)
            audio_path = wav_tmp
        # cleanup on exit
        import atexit
        atexit.register(lambda: os.path.exists(audio_path) and os.unlink(audio_path))

    res = transcribe(
        audio_path, asr_model=args.asr_model or DEFAULT_ASR_BATCH,
        diar_model=args.diar_model, language=args.language,
        no_diar=args.no_diar, diar_threshold=args.diar_threshold,
        diar_chunk_sec=args.diar_chunk_seconds, num_speakers=args.num_speakers,
        verbose=args.verbose,
    )
    fn = FORMATTERS[args.output]
    _emit_lines(fn(res, audio=audio_path if args.audio else None))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print("Input devices:")
        for idx, name, ch, rate in list_input_devices():
            mark = ""
            if "blackhole" in name.lower():
                mark = "  <- system loopback"
            elif "microphone" in name.lower() or "mic" in name.lower():
                mark = "  <- mic"
            print(f"  [{idx}] {name}  in={ch} sr={rate}{mark}")
        return 0

    if args.live:
        return _run_live(args)
    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
