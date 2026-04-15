import argparse
import os
import re
import tempfile

from processing.audio_cleaner import clean_audio
from processing.transcriber import transcribe
from processing.word_filter import filter_words


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def _default_model_for_language(language: str) -> str:
    return "medium.en" if language.lower().startswith("en") else "medium"


def run_accuracy_test(audio_path: str, reference: str, mode: str, apply_noise_reduction: bool, language: str) -> None:
    model_used = _default_model_for_language(language)
    vad_chunking_used = False

    transcription_input = audio_path
    cleaned_temp_path = None

    try:
        if apply_noise_reduction:
            fd, cleaned_temp_path = tempfile.mkstemp(suffix=".wav", prefix="accuracy_clean_")
            os.close(fd)
            clean_result = clean_audio(audio_path, cleaned_temp_path)
            cleaned_ok = bool(clean_result.get("success", False))
            if cleaned_ok:
                transcription_input = cleaned_temp_path
            else:
                print("Noise reduction failed in test path; using original audio for transcription.")

        raw_transcript = transcribe(
            transcription_input,
            model_name=model_used,
            language=language,
        )

        if mode == "clear":
            cleaned_transcript = filter_words(raw_transcript)["cleaned_transcript"]
        else:
            cleaned_transcript = raw_transcript

        # WER should be measured on raw transcript fidelity.
        wer = word_error_rate(reference, raw_transcript)

        print("=== Accuracy Test Result ===")
        print(f"model: {model_used}")
        print(f"mode: {mode}")
        print(f"noise_reduction: {apply_noise_reduction}")
        print(f"vad_chunking_used: {vad_chunking_used}")
        print(f"reference: {reference}")
        print(f"raw_transcript: {raw_transcript}")
        print(f"cleaned_transcript: {cleaned_transcript}")
        print(f"WER: {wer:.4f} ({wer * 100:.2f}%)")
        print(f"Accuracy: {(1 - wer) * 100:.2f}%")
    finally:
        if cleaned_temp_path and os.path.exists(cleaned_temp_path):
            os.remove(cleaned_temp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple transcription accuracy check and print WER.",
    )
    parser.add_argument("--audio", required=True, help="Path to test audio file")
    parser.add_argument("--reference", required=True, help="Known reference sentence/text for the audio")
    parser.add_argument("--mode", choices=["normal", "clear"], default="normal", help="Processing mode")
    parser.add_argument("--noise", action="store_true", help="Enable non-destructive noise reduction")
    parser.add_argument("--language", default="en", help="Whisper language code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    run_accuracy_test(
        audio_path=audio_path,
        reference=args.reference,
        mode=args.mode,
        apply_noise_reduction=args.noise,
        language=args.language,
    )


if __name__ == "__main__":
    main()
