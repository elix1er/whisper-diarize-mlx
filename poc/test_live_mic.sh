#!/usr/bin/env bash
# Live mic test — one command. Run AFTER granting mic permission to your
# Terminal app in System Settings > Privacy & Security > Microphone.
#
# Usage:  ./test_live_mic.sh [seconds]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-whisper/bin/activate
cd poc

SECS="${1:-15}"
echo "==> Live mic test for ${SECS}s. SPEAK NOW (ideally 2+ people, or alternate voices)."
echo "    NDJSON streams to stdout. Ctrl-C to stop early."
echo
python -u realtime_mic.py --seconds "$SECS" --verbose
echo
echo "==> Done. If you saw asr_partial events with your words and diar events"
echo "    with distinct speakers, the live mic works."
