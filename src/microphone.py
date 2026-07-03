import audioop
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import speech_recognition as sr

logger = logging.getLogger(__name__)

# Device-name substrings that are NOT real physical microphones.
_EXCLUDED_KEYWORDS = [
    "mapper",
    "stereo mix",
    "output",
    "speaker",
    "headphone",
    "loopback",
]


def list_microphones() -> list[dict]:
    """
    List all available microphone devices.

    Returns a list of dictionaries, each with 'index' (int) and 'name' (str).
    Returns an empty list if PyAudio is unavailable or no devices are found.
    """
    try:
        names = sr.Microphone.list_microphone_names()
        return [{"index": i, "name": name} for i, name in enumerate(names)]
    except (AttributeError, OSError) as exc:
        logger.warning("Could not list microphones: %s", exc)
        return []


def pick_best_microphone() -> Optional[dict]:
    """
    Automatically select the best physical input microphone.

    Selection logic (in order):
      1. First input-only device whose name contains "microphone", "mic",
         "array", "headset", or "audio".
      2. First device that is not excluded by ``_EXCLUDED_KEYWORDS``.
      3. First device overall.
      4. ``None`` if no devices are found.

    Returns a dict with ``index`` and ``name``, or ``None``.
    """
    devices = list_microphones()
    if not devices:
        logger.warning("No microphone devices found.")
        return None

    # Score each device: prefer long descriptive names (not truncated)
    # and exact matches over short/truncated names.
    def _score(dev: dict) -> int:
        name_lower = dev["name"].lower()
        excluded = any(kw in name_lower for kw in _EXCLUDED_KEYWORDS)
        if excluded:
            return -1
        score = 0
        if "microphone" in name_lower or "mic" in name_lower:
            score += 10
        if "array" in name_lower:
            score += 5
        if "headset" in name_lower:
            score += 3
        # Longer, non-truncated names indicate a richer device descriptor.
        score += min(len(dev["name"]), 60)
        return score

    scored = [(d, _score(d)) for d in devices]
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored and scored[0][1] > 0:
        best = scored[0][0]
        logger.info("Selected microphone [%d] %s (score=%d)", best["index"], best["name"], scored[0][1])
        return best

    # Fallback: first device that is not an excluded type.
    for d in devices:
        name_lower = d["name"].lower()
        excluded = any(kw in name_lower for kw in _EXCLUDED_KEYWORDS)
        if not excluded:
            logger.info("Fallback microphone [%d] %s", d["index"], d["name"])
            return d

    # Last resort: first device.
    logger.info("Using first available device [%d] %s", devices[0]["index"], devices[0]["name"])
    return devices[0]


def get_microphone(device_index: Optional[int] = None) -> Optional[sr.Microphone]:
    """
    Obtain a Microphone instance for the given device index (or best available).

    When ``device_index`` is ``None``, ``pick_best_microphone()`` is used
    instead of the system default, because the system default on Windows is
    often the virtual "Microsoft Sound Mapper" which can silently fail.

    Returns None if the device cannot be opened.
    """
    index = device_index
    if index is None:
        best = pick_best_microphone()
        if best is None:
            return None
        index = best["index"]
    else:
        available = sr.Microphone.list_microphone_names()
        if index < 0 or index >= len(available):
            logger.error("Device index %d out of range (0-%d).", index, len(available) - 1)
            return None

    try:
        mic = sr.Microphone(device_index=index, sample_rate=16000)
        logger.info("Opened microphone [%d] %s @ 16 kHz", index, sr.Microphone.list_microphone_names()[index])
        return mic
    except (OSError, AttributeError) as exc:
        logger.error("Microphone unavailable [%d]: %s", index, exc)
        return None


def calibrate_and_log(
    recognizer: sr.Recognizer,
    source: sr.Microphone,
    duration: float = 1.0,
) -> None:
    """
    Calibrate the recognizer's energy threshold against ambient noise
    and log the resulting value.

    A longer calibration duration (1-2 s) helps Windows audio drivers
    stabilise and yields a more reliable threshold.
    """
    logger.info("Calibrating for ambient noise (%.1f s)...", duration)
    recognizer.adjust_for_ambient_noise(source, duration=duration)
    logger.info(
        "Energy threshold after calibration: %d  (dynamic=%s)",
        recognizer.energy_threshold,
        recognizer.dynamic_energy_threshold,
    )


_SILENCE_DEVICE_INDEXES: list[int] = []


def capture_audio(
    duration: float = 8.0,
    phrase_time_limit: Optional[float] = None,
    device_index: Optional[int] = None,
    energy_threshold: int = 300,
) -> Optional[sr.AudioData]:
    """
    Record audio from the microphone and return an AudioData object.
    Uses ``pyaudio.paInt32`` directly because the Intel Smart Sound
    microphone array produces corrupted audio under ``paInt16``.
    """
    del phrase_time_limit

    global _SILENCE_DEVICE_INDEXES

    idx = device_index
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        attempts += 1

        if idx is None and attempts == 1:
            best = pick_best_microphone()
            if best is None:
                return None
            idx = best["index"]

        if idx in _SILENCE_DEVICE_INDEXES:
            logger.info("Skipping previously-silent device [%d], trying next ...", idx)
            idx = _next_device(idx)
            continue

        if idx is None:
            logger.error("No device index available.")
            return None

        mic_name = sr.Microphone.list_microphone_names()[idx] if 0 <= idx < len(sr.Microphone.list_microphone_names()) else f"device_{idx}"
        logger.info("")
        logger.info("Attempt %d/%d - microphone [%d] %s", attempts, max_attempts, idx, mic_name)
        logger.info("Recording %.1f s of audio ...", duration)

        audio = _capture_pa32(idx, duration)
        if audio is None:
            idx = _next_device(idx)
            continue

        rms = _rms(audio)
        dur = len(audio.frame_data) / audio.sample_width / audio.sample_rate
        logger.info("Captured %d bytes @ %d Hz (%.1f s)  RMS=%.1f",
                     len(audio.frame_data), audio.sample_rate, dur, rms)

        if rms == 0.0:
            logger.warning("Audio RMS is 0.0 — device [%d] returned pure silence.", idx)
            save_audio(audio, "output/debug_last_capture.wav")
            _SILENCE_DEVICE_INDEXES.append(idx)
            logger.warning(
                "Troubleshooting (Windows 11):\n"
                "  1. Settings > Privacy & security > Microphone\n"
                "     -> Enable 'Let apps access your microphone'\n"
                "  2. Ensure Python / your terminal app is allowed.\n"
                "  3. Settings > System > Sound > Input: select correct mic.\n"
                "  Debug WAV saved to output/debug_last_capture.wav"
            )
            idx = _next_device(idx)
            continue

        logger.info("Audio captured (RMS=%.1f).", rms)
        return audio

    logger.error("All %d microphone attempts failed.", max_attempts)
    return None


def _capture_pa32(device_index: int, duration: float) -> Optional[sr.AudioData]:
    """
    Capture audio using ``pyaudio.paInt32`` format.
    Converts int32 samples to int16 for compatibility with the rest of the pipeline.
    """
    sample_rate = 16000
    chunk = 1024
    warmup_sec = 0.5

    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk,
        )
    except Exception as exc:
        logger.error("Failed to open PyAudio stream [%d]: %s", device_index, exc)
        return None

    try:
        warmup_frames = int(sample_rate / chunk * warmup_sec)
        for _ in range(warmup_frames):
            stream.read(chunk, exception_on_overflow=False)

        frames = b''
        needed = int(sample_rate / chunk * duration)
        for _ in range(needed):
            data = stream.read(chunk, exception_on_overflow=False)
            frames += data
    except Exception as exc:
        logger.error("Error during capture [%d]: %s", device_index, exc)
        stream.close()
        p.terminate()
        return None

    stream.close()
    p.terminate()

    float32_samples = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
    int16_samples = (float32_samples * 32767).astype(np.int16)
    frame_data = int16_samples.tobytes()

    return sr.AudioData(frame_data, sample_rate, 2)


def _next_device(current_idx: int) -> Optional[int]:
    """Return the next best device index, or None to stop."""
    devices = list_microphones()
    if not devices:
        return None

    # Prefer input-type devices; skip already-known silence indexes.
    for d in devices:
        if d["index"] != current_idx and d["index"] not in _SILENCE_DEVICE_INDEXES:
            name_lower = d["name"].lower()
            if "microphone" in name_lower or "mic" in name_lower or "input" in name_lower or "array" in name_lower:
                if not any(kw in name_lower for kw in _EXCLUDED_KEYWORDS):
                    return d["index"]

    # Fallback: any non-excluded, non-silent device.
    for d in devices:
        if d["index"] != current_idx and d["index"] not in _SILENCE_DEVICE_INDEXES:
            name_lower = d["name"].lower()
            if not any(kw in name_lower for kw in _EXCLUDED_KEYWORDS):
                return d["index"]

    return None


def _rms(audio_data: sr.AudioData) -> float:
    """Compute the root-mean-square energy of an AudioData clip."""
    count = len(audio_data.frame_data) // audio_data.sample_width
    if count == 0:
        return 0.0
    return audioop.rms(audio_data.frame_data, audio_data.sample_width)


def _save_debug_audio(
    recognizer: sr.Recognizer,
    mic: sr.Microphone,
    filepath: str | Path,
) -> None:
    """
    Open the microphone again and record a short ambient sample for
    debugging purposes.  This helps determine whether the microphone is
    producing any audio signal at all.
    """
    try:
        with mic as source:
            logger.info("Capturing 2 s diagnostic audio to %s ...", filepath)
            debug_audio = recognizer.record(source, duration=2.0)
            save_audio(debug_audio, filepath)
            logger.info(
                "Diagnostic audio saved (%d bytes).  "
                "Listen to %s to verify your microphone is working.",
                len(debug_audio.frame_data),
                filepath,
            )
    except Exception as exc:
        logger.error("Failed to capture diagnostic audio: %s", exc)


def save_audio(audio_data: sr.AudioData, filepath: str | Path) -> bool:
    """
    Persist an AudioData object as a WAV file on disk.

    Creates parent directories automatically.  Returns True on success,
    False on any I/O failure.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = audio_data.get_wav_data()
        path.write_bytes(wav_bytes)
        logger.info("Audio saved to %s", path)
        return True
    except (IOError, OSError) as exc:
        logger.error("Failed to save audio: %s", exc)
        return False


def record_and_save(
    filepath: str | Path = "output/recording.wav",
    duration: float = 8.0,
    device_index: Optional[int] = None,
) -> bool:
    """
    High-level helper: capture microphone audio and write it to a WAV file.

    Returns True only when both capture and save succeed.
    """
    audio = capture_audio(duration=duration, device_index=device_index)
    if audio is None:
        return False
    return save_audio(audio, filepath)
