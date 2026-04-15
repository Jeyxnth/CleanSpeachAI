import os
import uuid
import shutil
import logging
import re
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from processing.audio_cleaner import clean_audio
from processing.transcriber import transcribe
from processing.word_filter import (
    get_filler_words, get_flagged_words, add_flagged_word
)

UPLOAD_DIR = "backend/temp/uploads"
ORIGINAL_DIR = "backend/temp/original"
CLEANED_DIR = "backend/temp/cleaned"
LEGACY_OUTPUT_DIR = "backend/temp/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(LEGACY_OUTPUT_DIR, exist_ok=True)

app = FastAPI()
logger = logging.getLogger("clean_speech.pipeline")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIO_EXTENSIONS = {"mp3", "wav", "ogg", "m4a", "webm"}
FILLERS = [
    "uh", "um", "uh-huh", "hmm", "like", "you know",
    "i mean", "sort of", "kind of", "basically", "literally",
    "actually", "so", "right", "okay so", "well",
]


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "normal").strip().lower()
    if normalized not in {"normal", "clear"}:
        raise HTTPException(status_code=400, detail="mode must be either 'normal' or 'clear'.")
    return normalized


def _select_model(language: str) -> str:
    return "medium.en" if language.lower().startswith("en") else "medium"


def _load_as_wav(input_path: str, output_path: str, target_sr: int = 16000) -> None:
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    if audio.size == 0:
        raise ValueError("Uploaded audio has no samples")
    sf.write(output_path, audio, sr, format="WAV", subtype="PCM_16")


def _compute_audio_metrics(original_path: str, cleaned_path: str) -> tuple[float, float]:
    original, _ = sf.read(original_path)
    cleaned, _ = sf.read(cleaned_path)

    if isinstance(original, np.ndarray) and original.ndim == 2:
        original = original.mean(axis=1)
    if isinstance(cleaned, np.ndarray) and cleaned.ndim == 2:
        cleaned = cleaned.mean(axis=1)

    original = np.asarray(original, dtype=np.float32)
    cleaned = np.asarray(cleaned, dtype=np.float32)

    n = min(len(original), len(cleaned))
    if n == 0:
        return 0.0, 0.0

    original = original[:n]
    cleaned = cleaned[:n]

    rms_original = float(np.sqrt(np.mean(original**2)))
    rms_cleaned = float(np.sqrt(np.mean(cleaned**2)))
    rms_diff = abs(rms_original - rms_cleaned)

    noise = original - cleaned
    signal_power = float(np.mean(cleaned**2))
    noise_power = float(np.mean(noise**2))
    if noise_power == 0:
        snr_db = 0.0
    else:
        snr_db = round(10 * np.log10(signal_power / noise_power), 2)

    return round(rms_diff, 6), snr_db


def _safe_audio_filename(filename: str) -> str:
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid audio filename")
    return safe_name


def remove_fillers(text: str) -> dict:
    original_words = text.split()
    cleaned = text

    for filler in FILLERS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r" +", " ", cleaned).strip()

    cleaned_words = cleaned.split() if cleaned else []
    fillers_removed = max(0, len(original_words) - len(cleaned_words))

    return {
        "cleaned_text": cleaned,
        "fillers_removed": fillers_removed,
        "original_word_count": len(original_words),
        "cleaned_word_count": len(cleaned_words),
    }


def calculate_clarity_score(noise_pct: float, fillers_removed: int) -> float:
    base = 60.0
    noise_bonus = min(noise_pct * 0.3, 30.0)
    filler_bonus = min(float(fillers_removed) * 2.0, 10.0)
    return round(base + noise_bonus + filler_bonus, 1)

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    mode: str = Form("normal"),
    apply_noise_reduction: bool = Form(False),
    language: str = Form("en"),
):
    ext = file.filename.split(".")[-1].lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    selected_mode = _normalize_mode(mode)
    use_noise_reduction = bool(apply_noise_reduction)
    use_chunked_vad = False

    print(f"[API RECEIVED] apply_noise_reduction={use_noise_reduction}")

    job_id = str(uuid.uuid4())
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
    original_filename = f"{job_id}.wav"
    cleaned_filename = f"{job_id}.wav"
    original_path = os.path.join(ORIGINAL_DIR, original_filename)
    cleaned_path = os.path.join(CLEANED_DIR, cleaned_filename)

    # Save upload
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        _load_as_wav(upload_path, original_path, target_sr=16000)
    except Exception as e:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")

    clean_result = {
        "noise_reduction_percent": 0,
        "success": False,
        "error": None,
    }

    if use_noise_reduction:
        clean_result = clean_audio(original_path, cleaned_path)
        if not clean_result.get("success", False):
            shutil.copyfile(original_path, cleaned_path)
    else:
        shutil.copyfile(original_path, cleaned_path)

    noise_reduction_percent = float(clean_result.get("noise_reduction_percent", 0) or 0)
    noise_reduction_applied = bool(use_noise_reduction and clean_result.get("success", False))
    noise_reduction_failed = bool(use_noise_reduction and not noise_reduction_applied)
    noise_reduction_error = clean_result.get("error")

    rms_diff, snr_db = _compute_audio_metrics(original_path, cleaned_path)
    if use_noise_reduction and rms_diff < 0.001:
        noise_reduction_applied = False
        noise_reduction_failed = True
        noise_reduction_percent = 0.0
        if not noise_reduction_error:
            noise_reduction_error = "Cleaned audio nearly identical to original"
        logger.error("noise_reduction.failed job_id=%s reason=low_rms_diff rms_diff=%.6f", job_id, rms_diff)

    transcription_input = original_path
    transcription_model = _select_model(language)

    logger.info(
        "transcription.start job_id=%s mode=%s model=%s noise_reduction=%s vad_chunking=%s language=%s filename=%s transcription_input=%s",
        job_id,
        selected_mode,
        transcription_model,
        use_noise_reduction,
        use_chunked_vad,
        language,
        file.filename,
        transcription_input,
    )

    # Transcribe
    try:
        raw_transcript = transcribe(
            transcription_input,
            model_name=transcription_model,
            language=language,
        )
    except Exception as e:
        logger.exception(
            "transcription.failed job_id=%s mode=%s model=%s noise_reduction=%s vad_chunking=%s language=%s",
            job_id,
            selected_mode,
            transcription_model,
            use_noise_reduction,
            use_chunked_vad,
            language,
        )
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    filler_result = {
        "cleaned_text": raw_transcript,
        "fillers_removed": 0,
        "original_word_count": len(raw_transcript.split()),
        "cleaned_word_count": len(raw_transcript.split()),
    }

    if selected_mode == "clear":
        filler_result = remove_fillers(raw_transcript)

    cleaned_transcript = filler_result["cleaned_text"]
    filler_count = int(filler_result["fillers_removed"])
    flagged_count = 0
    removed_words = []
    clarity_score = calculate_clarity_score(noise_reduction_percent, filler_count)

    # Clean up upload
    if os.path.exists(upload_path):
        os.remove(upload_path)

    logger.info(
        "transcription.complete job_id=%s mode=%s model=%s noise_reduction_requested=%s noise_reduction_applied=%s vad_chunking=%s rms_diff=%.6f snr_db=%.2f words_raw=%s words_cleaned=%s",
        job_id,
        selected_mode,
        transcription_model,
        use_noise_reduction,
        noise_reduction_applied,
        use_chunked_vad,
        rms_diff,
        snr_db,
        len((raw_transcript or "").split()),
        len((cleaned_transcript or "").split()),
    )

    original_audio_url = f"/audio/original/{original_filename}"
    cleaned_audio_url = f"/audio/cleaned/{cleaned_filename}"
    audio_download_url = cleaned_audio_url

    return {
        "job_id": job_id,
        "transcript": filler_result["cleaned_text"],
        "original_transcript": raw_transcript,
        "mode": selected_mode,
        "noise_applied": use_noise_reduction,
        "noise_reduction_percent": noise_reduction_percent,
        "fillers_removed": filler_count,
        "original_word_count": int(filler_result["original_word_count"]),
        "cleaned_word_count": int(filler_result["cleaned_word_count"]),
        "clarity_score": clarity_score,
        "noise_reduction_success": noise_reduction_applied,
        "noise_reduction_error": noise_reduction_error,
        "noise_reduction_applied": noise_reduction_applied,
        "noise_reduction_failed": noise_reduction_failed,
        "rms_diff": rms_diff,
        "snr_db": snr_db,
        "chunked_vad_applied": False,
        "chunk_count": 1,
        "large_model_requested": False,
        "model_fallback_applied": False,
        "model_used": transcription_model,
        "raw_transcript": raw_transcript,
        "cleaned_transcript": cleaned_transcript,
        "removed_words": removed_words,
        "filler_count": filler_count,
        "flagged_count": flagged_count,
        "audio_download_url": audio_download_url,
        "original_audio_url": original_audio_url,
        "cleaned_audio_url": cleaned_audio_url,
    }

@app.get("/download/{job_id}")
def download_audio(job_id: str):
    cleaned_wav_path = os.path.join(CLEANED_DIR, f"{job_id}.wav")
    legacy_wav_path = os.path.join(LEGACY_OUTPUT_DIR, f"{job_id}.wav")
    legacy_mp3_path = os.path.join(LEGACY_OUTPUT_DIR, f"{job_id}.mp3")

    if os.path.exists(cleaned_wav_path):
        return FileResponse(cleaned_wav_path, media_type="audio/wav", filename=f"cleaned_{job_id}.wav")

    if os.path.exists(legacy_wav_path):
        return FileResponse(legacy_wav_path, media_type="audio/wav", filename=f"cleaned_{job_id}.wav")

    if os.path.exists(legacy_mp3_path):
        return FileResponse(legacy_mp3_path, media_type="audio/mpeg", filename=f"cleaned_{job_id}.mp3")

    raise HTTPException(status_code=404, detail="File not found.")


@app.get("/audio/original/{filename}")
def get_original_audio(filename: str):
    safe_name = _safe_audio_filename(filename)
    original_path = os.path.join(ORIGINAL_DIR, safe_name)
    if not os.path.exists(original_path):
        raise HTTPException(status_code=404, detail="Original audio not found.")
    return FileResponse(original_path, media_type="audio/wav", filename=safe_name)


@app.get("/audio/cleaned/{filename}")
def get_cleaned_audio(filename: str):
    safe_name = _safe_audio_filename(filename)
    cleaned_path = os.path.join(CLEANED_DIR, safe_name)
    if not os.path.exists(cleaned_path):
        raise HTTPException(status_code=404, detail="Cleaned audio not found.")
    return FileResponse(cleaned_path, media_type="audio/wav", filename=safe_name)

@app.get("/words/filler")
def list_filler_words():
    return {"filler_words": get_filler_words()}

@app.get("/words/flagged")
def list_flagged_words():
    return {"flagged_words": get_flagged_words()}

@app.post("/words/flagged/add")
def add_flagged(word: str):
    if not word or not word.strip():
        raise HTTPException(status_code=400, detail="Word must not be empty.")
    add_flagged_word(word.strip())
    return {"flagged_words": get_flagged_words()}
