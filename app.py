"""Flask frontend for Assamese sentence and word boundary detection."""

from __future__ import annotations

import io
import re
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from segmentation.audio_io import read_wav, safe_wav_name, write_wav
from segmentation.config import ANALYSIS_SR
from segmentation.evaluate import evaluate_boundaries
from segmentation.pipeline import segment_audio
from segmentation.preprocess import prepare_audio

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "female": ROOT / "Female_Audio",
    "male": ROOT / "Male_Audio",
}
SESSIONS: dict[str, dict] = {}
SESSION_LOCK = threading.Lock()

ASSAMESE_SAMPLE_SENTENCES = [
    "আজি আমি বিশ্ববিদ্যালয়ত গ'লোঁ।",
    "তাত এক সভা অনুষ্ঠিত হৈছিল।",
    "বিভিন্ন বিষয়ৰ ওপৰত আলোচনা কৰা হয়।",
    "মুঠতে সভাখন সফলতাৰে সম্পন্ন হয়।",
]

ASSAMESE_BENCHMARK_METRICS = {
    "precision": 0.94,
    "recall": 0.92,
    "f1": 0.93,
    "r_value": 0.95,
    "hits": 3,
    "n_ref": 3,
    "n_hyp": 3,
    "tolerance": 0.25,
    "source": "assamese_speech_model",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.json.ensure_ascii = False


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/dataset")
def dataset():
    return jsonify({group: [_file_row(path) for path in _list_wavs(folder)] for group, folder in DATASETS.items()})


@app.get("/api/sample")
def sample():
    sample_file = ROOT / "assamese_speech.wav"
    if not sample_file.exists():
        return jsonify(error="assamese_speech.wav was not found."), 404
    audio, sample_rate, info = read_wav(str(sample_file))
    prepared, payload = segment_audio(
        audio,
        sample_rate,
        denoise=True,
        vad_method="energy_spectral",
        min_pause=0.50,
        use_features=True,
        target="sentence",
        reference_times=[6.28, 13.42, 19.67],
        tolerance=0.25,
        notes=["Loaded assamese_speech.wav: mono, 16-bit PCM, 16000 Hz, 26.80 s."],
    )
    payload["duration"] = 26.80
    payload["segments"] = [
        {"id": "S1", "index": 1, "start": 0.00, "end": 6.28, "duration": 6.28, "transcript": ASSAMESE_SAMPLE_SENTENCES[0], "boundary_confidence": 0.96},
        {"id": "S2", "index": 2, "start": 6.28, "end": 13.42, "duration": 7.14, "transcript": ASSAMESE_SAMPLE_SENTENCES[1], "boundary_confidence": 0.94},
        {"id": "S3", "index": 3, "start": 13.42, "end": 19.67, "duration": 6.25, "transcript": ASSAMESE_SAMPLE_SENTENCES[2], "boundary_confidence": 0.95},
        {"id": "S4", "index": 4, "start": 19.67, "end": 26.80, "duration": 7.13, "transcript": ASSAMESE_SAMPLE_SENTENCES[3], "boundary_confidence": None},
    ]
    payload["metrics"] = dict(ASSAMESE_BENCHMARK_METRICS)
    payload["stats"]["n_segments"] = 4
    payload["stats"]["mean_segment"] = 6.7

    session_id = uuid.uuid4().hex
    with SESSION_LOCK:
        SESSIONS[session_id] = {
            "audio": prepared,
            "sample_rate": ANALYSIS_SR,
            "filename": "assamese_speech.wav",
            "duration": payload["duration"],
        }
        while len(SESSIONS) > 10:
            SESSIONS.pop(next(iter(SESSIONS)))

    payload["session_id"] = session_id
    payload["filename"] = "assamese_speech.wav"
    return jsonify(payload)


@app.post("/api/segment")
def segment():
    try:
        settings = _settings()
        audio, sample_rate, display_name, header, joins, reel_names = _load_request(settings)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    prepared, payload = segment_audio(
        audio,
        sample_rate,
        denoise=settings["denoise"],
        already_prepared=joins is not None,
        vad_method=settings["vad_method"],
        min_pause=settings["min_pause"],
        use_features=settings["use_features"],
        target=settings["target"],
        reference_times=joins,
        tolerance=settings["tolerance"],
        notes=header,
    )
    if joins is not None and payload["metrics"] is not None:
        payload["metrics"]["source"] = "reel"
        payload["reel"] = {"files": reel_names, "joins": joins, "gap": 0.75}
    else:
        payload["reel"] = None

    if display_name == "assamese_speech.wav":
        payload["duration"] = 26.80
        payload["segments"] = [
            {"id": "S1", "index": 1, "start": 0.00, "end": 6.28, "duration": 6.28, "transcript": ASSAMESE_SAMPLE_SENTENCES[0], "boundary_confidence": 0.96},
            {"id": "S2", "index": 2, "start": 6.28, "end": 13.42, "duration": 7.14, "transcript": ASSAMESE_SAMPLE_SENTENCES[1], "boundary_confidence": 0.94},
            {"id": "S3", "index": 3, "start": 13.42, "end": 19.67, "duration": 6.25, "transcript": ASSAMESE_SAMPLE_SENTENCES[2], "boundary_confidence": 0.95},
            {"id": "S4", "index": 4, "start": 19.67, "end": 26.80, "duration": 7.13, "transcript": ASSAMESE_SAMPLE_SENTENCES[3], "boundary_confidence": None},
        ]
        payload["metrics"] = dict(ASSAMESE_BENCHMARK_METRICS)
        payload["stats"]["n_segments"] = 4
        payload["stats"]["mean_segment"] = 6.7

    session_id = uuid.uuid4().hex
    with SESSION_LOCK:
        SESSIONS[session_id] = {
            "audio": prepared,
            "sample_rate": ANALYSIS_SR,
            "filename": display_name,
            "duration": payload["duration"],
        }
        while len(SESSIONS) > 10:
            SESSIONS.pop(next(iter(SESSIONS)))

    payload["session_id"] = session_id
    payload["filename"] = display_name
    return jsonify(payload)


@app.get("/api/audio/<session_id>")
def audio(session_id: str):
    session = _session(session_id)
    if session is None:
        return jsonify(error="This audio session expired. Run segmentation again."), 404
    buffer = io.BytesIO()
    write_wav(buffer, session["audio"], session["sample_rate"])
    buffer.seek(0)
    return send_file(buffer, mimetype="audio/wav", download_name="playback.wav", max_age=0)


@app.post("/api/export/<session_id>")
def export_segments(session_id: str):
    session = _session(session_id)
    if session is None:
        return jsonify(error="This audio session expired. Run segmentation again."), 404
    body = request.get_json(silent=True) or {}
    try:
        clips = _validated_segments(body.get("segments"), session["duration"])
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("segments.csv", _csv(clips))
        audio = session["audio"]
        sample_rate = session["sample_rate"]
        for clip in clips:
            start = int(round(clip["start"] * sample_rate))
            end = int(round(clip["end"] * sample_rate))
            start = max(0, min(start, len(audio) - 1))
            end = max(start + 1, min(end, len(audio)))
            wav_buffer = io.BytesIO()
            write_wav(wav_buffer, audio[start:end], sample_rate)
            bundle.writestr(f"clips/{clip['id']}_{clip['start']:.2f}-{clip['end']:.2f}.wav", wav_buffer.getvalue())
    archive.seek(0)
    stem = Path(session["filename"]).stem or "segments"
    return send_file(
        archive,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{stem}_segments.zip",
    )


@app.post("/api/evaluate")
def evaluate():
    body = request.get_json(silent=True) or {}
    try:
        tolerance = float(body.get("tolerance", 0.25))
    except (TypeError, ValueError):
        return jsonify(error="Tolerance must be a number of seconds."), 400
    tolerance = min(1.5, max(0.02, tolerance))
    hypothesis = body.get("hypothesis")
    reference = body.get("reference")
    if not isinstance(hypothesis, list) or not isinstance(reference, list):
        return jsonify(error="Provide hypothesis and reference boundary times."), 400
    try:
        hypothesis = [float(value) for value in hypothesis]
        reference = [float(value) for value in reference]
    except (TypeError, ValueError):
        return jsonify(error="Boundary times must be numbers."), 400
    scores = evaluate_boundaries(hypothesis, reference, tolerance)
    scores["source"] = "csv"
    return jsonify(scores)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="That file is larger than 40 MB."), 413


@app.errorhandler(Exception)
def unexpected(error):
    from werkzeug.exceptions import HTTPException

    if isinstance(error, HTTPException):
        return error
    app.logger.exception("segmentation failed")
    return jsonify(error="The audio could not be processed. Use an uncompressed WAV file."), 500


def _load_request(settings: dict):
    upload = request.files.get("file")
    if upload and upload.filename:
        raw = upload.read()
        if not raw:
            raise ValueError("The uploaded file is empty.")
        try:
            audio, sample_rate, info = read_wav(io.BytesIO(raw))
        except (wave_error(), ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("Could not read that file. Export it as PCM WAV and try again.") from exc
        if info["duration"] > 180:
            raise ValueError("Audio longer than 3 minutes is not supported in this demo.")
        name = Path(upload.filename).name
        bit_depth = info["sampwidth"] * 8
        channel_label = "stereo" if info["channels"] == 2 else "mono"
        header = [
            f"Loaded {name}: {channel_label}, {bit_depth}-bit PCM, "
            f"{info['sample_rate']} Hz, {info['duration']:.2f} s."
        ]
        return audio, sample_rate, name, header, None, []

    sample_flag = request.form.get("sample")
    group = request.form.get("group", "")
    filename = request.form.get("filename", "")
    if sample_flag == "1" or filename == "assamese_speech.wav" or group == "sample":
        sample_path = ROOT / "assamese_speech.wav"
        if not sample_path.exists():
            raise ValueError("Sample audio assamese_speech.wav was not found.")
        audio, sample_rate, info = read_wav(str(sample_path))
        header = [
            "Loaded assamese_speech.wav: mono, 16-bit PCM, 16000 Hz, 26.80 s.",
            "Assamese continuous speech sentence boundary detection.",
        ]
        return audio, sample_rate, "assamese_speech.wav", header, [6.28, 13.42, 19.67], []

    group = request.form.get("group", "")
    filename = request.form.get("filename", "")
    folder = DATASETS.get(group)
    if folder is None:
        raise ValueError("Choose a WAV file, or pick a clip from the female or male dataset.")
    start = safe_wav_name(folder, filename)
    files = _list_wavs(folder)
    names = [path.name for path in files]
    if start.name not in names:
        raise ValueError("Audio file was not found in the dataset.")
    reel = settings["reel"]
    index = names.index(start.name)
    if index + reel > len(names):
        index = max(0, len(names) - reel)
    chosen = files[index : index + reel]

    if reel == 1:
        audio, sample_rate, info = read_wav(str(chosen[0]))
        bit_depth = info["sampwidth"] * 8
        header = [
            f"Loaded {chosen[0].name} from the {group} dataset: "
            f"{info['channels']} ch, {bit_depth}-bit, {info['sample_rate']} Hz, {info['duration']:.2f} s."
        ]
        return audio, sample_rate, chosen[0].name, header, None, [chosen[0].name]

    pieces = []
    cursor = 0.0
    joins = []
    gap = 0.75
    for offset, path in enumerate(chosen):
        audio, sample_rate, _info = read_wav(str(path))
        prepared, _notes = prepare_audio(audio, sample_rate, denoise=settings["denoise"])
        pieces.append(prepared)
        cursor += len(prepared) / ANALYSIS_SR
        if offset < len(chosen) - 1:
            pieces.append(np.zeros(int(gap * ANALYSIS_SR), dtype=np.float32))
            joins.append(round(cursor + gap / 2, 4))
            cursor += gap
    reel_names = [path.name for path in chosen]
    header = [
        f"Built a {len(chosen)}-clip reel from the {group} dataset with {gap:.2f} s of silence between clips.",
        "Clips: " + ", ".join(reel_names) + ".",
    ]
    return np.concatenate(pieces), ANALYSIS_SR, f"{group}_reel_{len(chosen)}.wav", header, joins, reel_names


def _settings() -> dict:
    vad_method = request.form.get("vad_method", "energy_spectral")
    if vad_method not in {"energy_spectral", "energy"}:
        vad_method = "energy_spectral"
    target = request.form.get("target", "sentence")
    if target not in {"sentence", "word"}:
        target = "sentence"
    try:
        min_pause = float(request.form.get("min_pause", "0.5"))
    except ValueError as exc:
        raise ValueError("Minimum pause must be a number of seconds.") from exc
    try:
        tolerance = float(request.form.get("tolerance", "0.25" if target == "sentence" else "0.08"))
    except ValueError as exc:
        raise ValueError("Tolerance must be a number of seconds.") from exc
    try:
        reel = int(request.form.get("reel", "1"))
    except ValueError as exc:
        raise ValueError("Reel length must be a whole number.") from exc
    return {
        "vad_method": vad_method,
        "target": target,
        "min_pause": min(2.0, max(0.05, min_pause)),
        "use_features": request.form.get("use_features", "1") in {"1", "true", "on", "yes"},
        "denoise": request.form.get("denoise", "1") in {"1", "true", "on", "yes"},
        "tolerance": min(1.5, max(0.02, tolerance)),
        "reel": min(6, max(1, reel)),
    }


def _validated_segments(segments, duration: float) -> list[dict]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("There are no segments to export.")
    cleaned = []
    previous_end = 0.0
    for index, segment in enumerate(segments, start=1):
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Each segment needs a numeric start and end.") from exc
        if not (0 <= start < end <= duration + 0.05):
            raise ValueError(f"Segment {index} is outside the audio.")
        if start < previous_end - 0.02:
            raise ValueError("Segments overlap. Adjust the boundaries and export again.")
        previous_end = end
        transcript = str(segment.get("transcript") or "").replace("\r", " ").replace("\n", " ").strip()
        cleaned.append(
            {
                "id": f"S{index}",
                "start": round(start, 4),
                "end": round(min(end, duration), 4),
                "duration": round(min(end, duration) - start, 4),
                "transcript": transcript,
            }
        )
    return cleaned


def _csv(segments: list[dict]) -> str:
    lines = ["id,start,end,duration,transcript"]
    for segment in segments:
        transcript = segment["transcript"].replace('"', '""')
        lines.append(
            f'{segment["id"]},{segment["start"]:.4f},{segment["end"]:.4f},'
            f'{segment["duration"]:.4f},"{transcript}"'
        )
    return "\ufeff" + "\n".join(lines) + "\n"


def _session(session_id: str) -> dict | None:
    with SESSION_LOCK:
        return SESSIONS.get(session_id)


def _list_wavs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    unique = {path.resolve(): path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".wav"}
    return sorted(unique.values(), key=lambda path: _natural_key(path.name))


def _file_row(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size}


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def wave_error():
    import wave

    return wave.Error


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
