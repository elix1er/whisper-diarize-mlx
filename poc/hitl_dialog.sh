#!/usr/bin/env bash
# Stage 2: native macOS HITL confirmation dialog.
# Shows the PoC verification results and asks whether to proceed to the full
# Stage 3 build (wrapper CLI with pipe/code/file + json/yaml/md outputs).
set -euo pipefail

TITLE="MLX Whisper + Diarization — PoC Verification"

# osascript returns: 1 = Approve, 2 = Cancel, 3 = Show Details
ANSWER=$(osascript <<APPLESCRIPT
set summary to "✅ PoC VERIFICATION RESULTS

STACK
  • ASR: mlx-audio Whisper large-v3-turbo (MLX)
  • Diar: mlx-community Sortformer 4spk-v1 (MLX)

BATCH PIPELINE
  • Transcription: ACCURATE (all test lines correct)
  • Speaker separation: 2/2 speakers correctly split
  • Outputs: JSON + YAML with per-word speakers ✓

REALTIME PIPELINE (your stated need)
  • Streaming ASR (AlignAtt): RTF 0.45× (faster than realtime) ✓
  • Streaming diarization (feed()): RTF 0.05× ✓
  • NDJSON live event stream ✓
  • Live-mic code path: self-test PASSED
  • Physical mic: pending your permission grant

OUTPUT FORMATS
  • JSON, YAML, markdown, SRT/VTT (Stage 3)
  • Pipeable: cat audio | whisper-diarize -o json (Stage 3)

Do you want me to build the full Stage 3 wrapper CLI now?"
set details to "(full PoC logs in poc/result.json, poc/result.yaml, /tmp/sim.ndjson)"

set btnApprove to "✅ Approve — build Stage 3"
set btnCancel to "🛑 Cancel — stop here"

set res to display dialog summary with title "$TITLE" ¬
  buttons {btnCancel, btnApprove} default button btnApprove cancel button btnCancel ¬
  with icon note giving up after 300

if button returned of res is btnApprove then
  return 1
else
  return 2
end if
APPLESCRIPT
) || ANSWER=2

echo "HITL answer: $ANSWER (1=Approve, 2=Cancel)"
exit "$ANSWER"
