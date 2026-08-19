import unittest

from whisper_diarize.attribution import assign_speaker, select_speakers
from whisper_diarize.cli import _run_live, _run_offline, build_parser


class AttributionTests(unittest.TestCase):
    def test_silence_gap_uses_nearest_confident_speaker(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": 0},
            {"start": 8.0, "end": 10.0, "speaker": 1},
        ]
        self.assertEqual(assign_speaker(2.5, 3.0, turns), 0)
        self.assertEqual(assign_speaker(7.0, 7.5, turns), 1)

    def test_distant_gap_remains_unassigned(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": 0},
            {"start": 100.0, "end": 102.0, "speaker": 1},
        ]
        self.assertEqual(assign_speaker(50.0, 51.0, turns), -1)

    def test_auto_keeps_legitimate_low_talk_speaker(self):
        turns = [
            {"start": 0.0, "end": 1800.0, "speaker": 0},
            {"start": 0.0, "end": 1000.0, "speaker": 1},
            {"start": 600.0, "end": 620.0, "speaker": 2},
            {"start": 1200.0, "end": 1203.0, "speaker": 3},
        ]
        selected = select_speakers(turns)
        self.assertEqual({turn["speaker"] for turn in selected}, {0, 1, 2})

    def test_known_count_is_authoritative(self):
        turns = [
            {"start": 0.0, "end": 30.0, "speaker": 3},
            {"start": 1.0, "end": 2.0, "speaker": 1},
            {"start": 3.0, "end": 3.5, "speaker": 2},
        ]
        selected = select_speakers(turns, num_speakers=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual([turn["speaker"] for turn in selected], [0, 1])


class CliContractTests(unittest.TestCase):
    def test_offline_ndjson_is_rejected_before_model_loading(self):
        args = build_parser().parse_args(["audio.wav", "-o", "ndjson"])
        self.assertEqual(_run_offline(args), 2)

    def test_live_known_speaker_count_is_rejected_before_capture(self):
        args = build_parser().parse_args(["--live", "--num-speakers", "2"])
        self.assertEqual(_run_live(args), 2)


if __name__ == "__main__":
    unittest.main()
