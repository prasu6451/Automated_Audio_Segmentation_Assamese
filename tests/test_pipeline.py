"""Local tests for segmentation, scoring, and the Flask API."""

from __future__ import annotations

import io
import unittest
import wave
from pathlib import Path

import numpy as np

from segmentation.audio_io import read_wav, write_wav
from segmentation.evaluate import evaluate_boundaries
from segmentation.pipeline import segment_audio

ROOT = Path(__file__).resolve().parents[1]


def _tone(frequency: float, seconds: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = 0.45 * np.sin(2 * np.pi * frequency * t)
    signal += 0.2 * np.sin(2 * np.pi * 2 * frequency * t)
    signal += 0.05 * np.sin(2 * np.pi * 3 * frequency * t)
    signal += 0.008 * rng.normal(size=len(t))
    return signal.astype(np.float32)


def synthetic_sentences(sample_rate: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(7)
    gap = np.zeros(int(sample_rate * 0.62), dtype=np.float32)
    return np.concatenate(
        [
            _tone(150, 1.05, sample_rate, rng),
            gap,
            _tone(190, 0.95, sample_rate, rng),
            gap,
            _tone(230, 1.1, sample_rate, rng),
        ]
    )


class PipelineTests(unittest.TestCase):
    def test_three_sentences_are_recovered(self):
        audio = synthetic_sentences()
        _, payload = segment_audio(
            audio,
            16000,
            denoise=True,
            min_pause=0.45,
            use_features=True,
            target="sentence",
        )
        self.assertEqual(len(payload["segments"]), 3)
        self.assertGreater(payload["stats"]["speech_ratio"], 0.45)
        self.assertLess(payload["segments"][0]["start"], 0.2)
        self.assertGreater(payload["segments"][-1]["end"], payload["duration"] - 0.2)
        self.assertEqual(len(payload["features"]["ste"]), len(payload["features"]["f0"]))
        self.assertTrue(any(value is not None for value in payload["features"]["f0"]))
        self.assertGreater(len(payload["spectrogram"]["data_b64"]), 100)

    def test_pause_only_matches_feature_mode_on_clear_gaps(self):
        audio = synthetic_sentences()
        _, with_features = segment_audio(audio, 16000, use_features=True, min_pause=0.45)
        _, pauses_only = segment_audio(audio, 16000, use_features=False, min_pause=0.45)
        self.assertEqual(len(with_features["segments"]), len(pauses_only["segments"]))

    def test_r_value_is_perfect_when_boundaries_match(self):
        scores = evaluate_boundaries([1.2, 3.4], [1.22, 3.35], tolerance=0.25)
        self.assertEqual(scores["hits"], 2)
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertGreater(scores["r_value"], 0.95)

    def test_missed_boundary_lowers_recall(self):
        scores = evaluate_boundaries([1.2], [1.2, 3.4], tolerance=0.2)
        self.assertEqual(scores["hits"], 1)
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 0.5)
        self.assertLess(scores["r_value"], 0.95)

    def test_wav_roundtrip_16_and_24_bit(self):
        rng = np.random.default_rng(1)
        audio = (0.2 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)).astype(np.float32)
        audio += 0.01 * rng.normal(size=len(audio)).astype(np.float32)
        buffer = io.BytesIO()
        write_wav(buffer, audio, 16000)
        buffer.seek(0)
        loaded, sample_rate, info = read_wav(buffer)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(info["channels"], 1)
        self.assertGreater(np.corrcoef(audio[: len(loaded)], loaded[: len(audio)])[0, 1], 0.99)

        path = ROOT / "tests" / "_tmp24.wav"
        try:
            _write_pcm24(path, audio, 16000)
            loaded24, sample_rate, info = read_wav(path)
            self.assertEqual(info["sampwidth"], 3)
            self.assertEqual(sample_rate, 16000)
            self.assertGreater(np.corrcoef(audio, loaded24[: len(audio)])[0, 1], 0.99)
        finally:
            path.unlink(missing_ok=True)

    def test_real_dataset_file_runs(self):
        sample = next((ROOT / "Female_Audio").glob("*.WAV"), None)
        if sample is None:
            self.skipTest("dataset not present")
        audio, sample_rate, info = read_wav(sample)
        self.assertEqual(info["channels"], 2)
        self.assertEqual(info["sampwidth"], 3)
        self.assertEqual(sample_rate, 96000)
        _, payload = segment_audio(audio, sample_rate, min_pause=0.5, use_features=True)
        self.assertGreaterEqual(len(payload["segments"]), 1)
        self.assertAlmostEqual(payload["segments"][-1]["end"], payload["duration"], delta=0.15)
        self.assertGreater(payload["stats"]["speech_ratio"], 0.2)


class ApiTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_dataset_lists_both_speakers(self):
        response = self.client.get("/api/dataset")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreater(len(body["female"]), 50)
        self.assertGreater(len(body["male"]), 50)
        self.assertTrue(body["female"][0]["name"].lower().endswith(".wav"))

    def test_path_traversal_is_rejected(self):
        response = self.client.post(
            "/api/segment",
            data={"group": "female", "filename": "../project.png", "reel": "1"},
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_segment_and_export(self):
        audio = synthetic_sentences()
        buffer = io.BytesIO()
        write_wav(buffer, audio, 16000)
        response = self.client.post(
            "/api/segment",
            data={
                "file": (io.BytesIO(buffer.getvalue()), "sentences.wav"),
                "min_pause": "0.45",
                "use_features": "1",
                "target": "sentence",
                "vad_method": "energy_spectral",
                "denoise": "1",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_json()
        self.assertEqual(len(body["segments"]), 3)
        audio_response = self.client.get(f"/api/audio/{body['session_id']}")
        self.assertEqual(audio_response.status_code, 200)
        self.assertIn("audio/wav", audio_response.mimetype)
        export = self.client.post(
            f"/api/export/{body['session_id']}",
            json={"segments": body["segments"]},
        )
        self.assertEqual(export.status_code, 200)
        self.assertIn("zip", export.mimetype)


def _write_pcm24(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    values = np.clip(np.round(audio * 8388607.0), -8388607, 8388607).astype(np.int32)
    unsigned = values.astype(np.uint32) & 0xFFFFFF
    raw = np.empty(len(values) * 3, dtype=np.uint8)
    raw[0::3] = unsigned & 0xFF
    raw[1::3] = (unsigned >> 8) & 0xFF
    raw[2::3] = (unsigned >> 16) & 0xFF
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(3)
        handle.setframerate(sample_rate)
        handle.writeframes(raw.tobytes())


if __name__ == "__main__":
    unittest.main()
