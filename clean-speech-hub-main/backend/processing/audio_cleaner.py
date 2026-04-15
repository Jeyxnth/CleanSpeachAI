import noisereduce as nr
import soundfile as sf
import numpy as np


def clean_audio(input_path: str, output_path: str) -> dict:
    result = {
        "noise_reduction_percent": 0,
        "success": False,
        "error": None,
    }

    try:
        data, rate = sf.read(input_path)

        if data.ndim == 2:
            data = data.mean(axis=1)

        data = np.asarray(data, dtype=np.float32)

        rms_before = float(np.sqrt(np.mean(data**2)))

        noise_len = min(int(0.5 * rate), len(data) // 4)
        if noise_len <= 0:
            raise ValueError("Noise profile length is zero")
        noise_clip = data[:noise_len]

        cleaned = nr.reduce_noise(
            y=data,
            sr=rate,
            y_noise=noise_clip,
            prop_decrease=0.90,
            stationary=False,
            n_fft=1024,
            hop_length=256,
        )

        cleaned = np.asarray(cleaned, dtype=np.float32)
        rms_after = float(np.sqrt(np.mean(cleaned**2)))

        if rms_before > 0:
            noise_removed = max(0.0, rms_before - rms_after)
            percent = round((noise_removed / rms_before) * 100, 1)
        else:
            percent = 0

        peak = float(np.max(np.abs(cleaned)))
        if peak > 0:
            cleaned = cleaned / peak * 0.95

        sf.write(output_path, cleaned, rate, subtype="PCM_16")

        result["noise_reduction_percent"] = percent
        result["success"] = True

        print(f"[NOISE REDUCTION] Before RMS: {rms_before:.4f}")
        print(f"[NOISE REDUCTION] After RMS:  {rms_after:.4f}")
        print(f"[NOISE REDUCTION] Removed:    {percent}%")
    except Exception as e:
        result["error"] = str(e)
        print(f"[NOISE REDUCTION FAILED]: {e}")
        import shutil
        shutil.copy2(input_path, output_path)

    return result
