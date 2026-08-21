"""Command-line interface for whisper-diarize."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from .formatters import FORMATTERS
from .offline import (
    DEFAULT_ASR_FILE,
    DEFAULT_DIAR,
    DEFAULT_DIAR_CHUNK_SEC,
    DEFAULT_HALLUCINATION_SILENCE_SEC,
    transcribe,
)
from .sources import EOS, LiveAudioSource, list_input_devices
from .streaming import DEFAULT_ASR_STREAM, transcribe_stream


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_lines(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-diarize",
        description="Local MLX Whisper transcription + Sortformer speaker diarization for Apple Silicon.",
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="audio file path (omit for stdin, or use --live)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="json",
        choices=list(FORMATTERS) + ["ndjson"],
        help="offline format; live mode always emits NDJSON events",
    )
    parser.add_argument("--live", action="store_true", help="capture live audio and emit NDJSON events")
    parser.add_argument(
        "--source",
        default="mic",
        help="live source: mic | blackhole | device index | device-name substring",
    )
    parser.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    parser.add_argument(
        "--asr-model",
        default=None,
        help=f"ASR model (default: {DEFAULT_ASR_FILE} offline / {DEFAULT_ASR_STREAM} live)",
    )
    parser.add_argument("--diar-model", default=DEFAULT_DIAR)
    parser.add_argument(
        "--language",
        default=None,
        help="language code; offline auto-detects when omitted, live defaults to en",
    )
    parser.add_argument("--no-diar", action="store_true", help="disable speaker diarization")
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="feed earlier Whisper text into later windows; may improve continuity but can amplify loops",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=_nonnegative_float,
        default=DEFAULT_HALLUCINATION_SILENCE_SEC,
        metavar="SECONDS",
        help=(
            "skip likely hallucinations surrounded by this much silence "
            f"(default: {DEFAULT_HALLUCINATION_SILENCE_SEC:g}; 0 disables)"
        ),
    )
    parser.add_argument(
        "--initial-prompt",
        help="optional domain terms or names to prime offline Whisper decoding",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="live: stop after N seconds (0 = until Ctrl-C)",
    )
    parser.add_argument("--diar-threshold", type=float, default=0.5)
    parser.add_argument(
        "--diar-chunk-seconds",
        type=float,
        default=DEFAULT_DIAR_CHUNK_SEC,
        help="Sortformer chunk size in seconds (default: 5; state persists across chunks)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        choices=range(1, 5),
        default=None,
        help="offline known speaker count, 1-4",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_live_device(source: str):
    if source == "mic":
        return None
    if source == "blackhole":
        return "blackhole"
    try:
        return int(source)
    except (TypeError, ValueError):
        return source


def _run_live(args: argparse.Namespace) -> int:
    if args.audio is not None:
        print("error: positional audio cannot be combined with --live", file=sys.stderr)
        return 2
    if args.num_speakers is not None:
        print("error: --num-speakers is currently an offline-only option", file=sys.stderr)
        return 2

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    source = LiveAudioSource(device=_resolve_live_device(args.source), stop_event=stop)
    source.start()
    sys.stderr.write(
        f"[whisper-diarize] live source={args.source!r} device={source._resolved_index} — NDJSON\n"
    )
    try:
        chunks = source.chunks()
        for event in transcribe_stream(
            chunk_source=lambda: next(chunks, EOS),
            asr_model=args.asr_model or DEFAULT_ASR_STREAM,
            diar_model=args.diar_model,
            language=args.language or "en",
            no_diar=args.no_diar,
            diar_chunk_sec=args.diar_chunk_seconds,
            diar_threshold=args.diar_threshold,
            max_seconds=args.seconds,
            verbose=args.verbose,
        ):
            _emit(event)
    finally:
        source.stop()
        sys.stderr.write("[whisper-diarize] stopped\n")
    return 0


def _stdin_to_temp_audio() -> str:
    import atexit
    import os
    import subprocess
    import tempfile

    raw = sys.stdin.buffer.read()
    suffix = ".wav" if raw[:4] == b"RIFF" else ".raw"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(raw)
        path = handle.name

    if suffix == ".raw":
        wav_path = path + ".wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-i",
                path,
                wav_path,
            ],
            check=True,
            capture_output=True,
        )
        os.unlink(path)
        path = wav_path

    atexit.register(lambda: os.path.exists(path) and os.unlink(path))
    return path


def _run_offline(args: argparse.Namespace) -> int:
    if args.output == "ndjson":
        print("error: ndjson is only available with --live", file=sys.stderr)
        return 2
    if args.seconds:
        print("error: --seconds is only available with --live", file=sys.stderr)
        return 2
    if args.audio is None and sys.stdin.isatty():
        print("error: provide an audio file, pipe stdin, or use --live", file=sys.stderr)
        return 2

    audio_path = args.audio if args.audio is not None else _stdin_to_temp_audio()
    result = transcribe(
        audio_path,
        asr_model=args.asr_model or DEFAULT_ASR_FILE,
        diar_model=args.diar_model,
        language=args.language,
        no_diar=args.no_diar,
        condition_on_previous_text=args.condition_on_previous_text,
        hallucination_silence_threshold=args.hallucination_silence_threshold or None,
        initial_prompt=args.initial_prompt,
        diar_threshold=args.diar_threshold,
        diar_chunk_sec=args.diar_chunk_seconds,
        num_speakers=args.num_speakers,
        verbose=args.verbose,
    )
    formatter = FORMATTERS[args.output]
    _emit_lines(formatter(result, audio=audio_path if args.audio else None))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print("Input devices:")
        for index, name, channels, rate in list_input_devices():
            marker = ""
            if "blackhole" in name.lower():
                marker = "  <- system loopback"
            elif "microphone" in name.lower() or "mic" in name.lower():
                marker = "  <- mic"
            print(f"  [{index}] {name}  in={channels} sr={rate}{marker}")
        return 0

    return _run_live(args) if args.live else _run_offline(args)


if __name__ == "__main__":
    raise SystemExit(main())
