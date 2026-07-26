#!/usr/bin/env python3
"""
Reproducibly generate the multi-speaker test audio used by the PoC scripts.

Why this exists: the bundled poc/samples/clip2_pitchshift.wav is a *generated*
synthetic clip (two macOS `say` voices, one pitch-shifted), so anyone can
regenerate it instead of trusting a binary blob. This also documents exactly
how the diarization test signal is constructed.

Prereqs: macOS `say` (system) and `ffmpeg` on PATH.

Output: poc/samples/clip2_pitchshift.wav  (16k mono wav, ~18s, 2 speakers)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Two near-identical phrases spoken by the SAME voice (Alex), then one is
# pitch/formant-shifted so the two "speakers" are spectrally distinct enough
# for a diarizer trained on human voice variation to separate them.
LINES = [
    ("A", "Speaker Alpha reporting. The system is online and stable."),
    ("B", "Speaker Bravo responding. All sensors read nominal."),
]


def run(cmd: list[str], **kw):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    if shutil_which("say") is None:
        print("error: macOS `say` not found. Run on macOS.", file=sys.stderr)
        return 1
    if shutil_which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH. `brew install ffmpeg`.", file=sys.stderr)
        return 1

    tmp = HERE / "_tmp_gen"
    tmp.mkdir(exist_ok=True)
    try:
        # 1) Synthesize plain Alex for both lines
        raw = {}
        for tag, line in LINES:
            p = tmp / f"raw_{tag}.aiff"
            run(["say", "-v", "Alex", "-o", str(p), line])
            raw[tag] = p

        # 2) Pitch/formant-shift each into a distinct "speaker"
        #    A: shifted up, B: shifted down (atempo keeps duration sane)
        shifted = {}
        for tag, ratio, tempo in [("A", "1.2", "0.833"), ("B", "0.75", "1.33")]:
            p = tmp / f"raw_{tag}_shift.aiff"
            run([
                "ffmpeg", "-y", "-i", str(raw[tag]),
                "-af", f"asetrate=16000*{ratio},aresample=16000,atempo={tempo}",
                str(p),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shifted[tag] = p

        # 3) Interleave A B A B and concat into the final 16k mono wav
        parts = [shifted["A"], shifted["B"], shifted["A"], shifted["B"]]
        concat_txt = tmp / "concat.txt"
        concat_txt.write_text("".join(f"file '{p.name}'\n" for p in parts))

        out = OUT_DIR / "clip2_pitchshift.wav"
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\nGenerated: {out}  ({out.stat().st_size} bytes)")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def shutil_which(cmd: str):
    from shutil import which
    return which(cmd)


if __name__ == "__main__":
    sys.exit(main())
