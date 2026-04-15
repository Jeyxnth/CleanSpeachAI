import os
import shutil
import tempfile
from functools import lru_cache

import whisper


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is required for transcription but was not found. Install ffmpeg or add imageio-ffmpeg."
        ) from exc

    packaged_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    shim_dir = os.path.join(tempfile.gettempdir(), "clean_speech_ffmpeg")
    os.makedirs(shim_dir, exist_ok=True)
    shim_path = os.path.join(shim_dir, "ffmpeg.exe")

    if not os.path.exists(shim_path):
        shutil.copyfile(packaged_ffmpeg, shim_path)

    os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is unavailable after imageio-ffmpeg bootstrap.")


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    return whisper.load_model(model_name)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _default_model_for_language(language: str) -> str:
    return "medium.en" if language.lower().startswith("en") else "medium"


def transcribe(audio_path: str, model_name: str | None = None, language: str = "en", initial_prompt: str | None = None) -> str:
    ensure_ffmpeg_available()
    effective_model = model_name or _default_model_for_language(language)
    model = _load_model(effective_model)
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language=language,
        temperature=0.0,
        fp16=False,
        best_of=5,
        beam_size=5,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.2,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        initial_prompt=initial_prompt or None,
    )
    return _normalize_text(result.get("text", ""))
