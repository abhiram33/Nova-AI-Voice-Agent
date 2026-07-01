import audioop
import logging
import time
from pathlib import Path
from typing import Optional

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

    Unlike ``listen()`` (which uses energy-gated voice activity detection
    that is unreliable on certain Windows drivers), this function uses
    ``record()`` to capture a fixed-duration clip.  Whisper handles
    silence and noise internally, so VAD at the recording layer is
    unnecessary.

    The function:

    1. Picks a real physical microphone (not "Sound Mapper").
    2. Records *duration* seconds of audio.
    3. Computes the RMS energy of the recording.  If it is below
       *energy_threshold* the audio is treated as silence and ``None``
       is returned.  A diagnostic WAV is still saved to
       ``output/debug_ambient.wav`` for debugging.

    Parameters
    ----------
    duration : float
        Fixed recording length in seconds (default 8).
    phrase_time_limit : float or None
        **Ignored** (present only for API compatibility).
    device_index : int or None
        Index of the microphone to use.  ``None`` auto-selects the best
        physical microphone via ``pick_best_microphone()``.
    energy_threshold : int
        Minimum RMS energy below which the recording is considered
        silence and discarded.  Default 100 (adjust if your mic is
        very quiet or very sensitive).

    Returns
    -------
    AudioData or None
        ``None`` indicates the recorded audio was below the energy
        threshold (likely silence).  A debug WAV is saved to
        ``output/debug_ambient.wav`` for analysis.
    """
    del phrase_time_limit  # unused — record() captures a fixed window

    global _SILENCE_DEVICE_INDEXES

    idx = device_index
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        attempts += 1

        # Resolve the device index to try on this attempt.
        if idx is None and attempts == 1:
            best = pick_best_microphone()
            if best is None:
                return None
            idx = best["index"]

        # Skip indexes already known to produce silence.
        if idx in _SILENCE_DEVICE_INDEXES:
            logger.info(
                "Skipping previously-silent device [%d], trying next ...", idx
            )
            idx = _next_device(idx)
            continue

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = energy_threshold
        recognizer.dynamic_energy_threshold = False

        mic = get_microphone(idx)
        if mic is None:
            idx = _next_device(idx)
            continue

        mic_name = sr.Microphone.list_microphone_names()[mic.device_index]
        logger.info("")
        logger.info("Attempt %d/%d - microphone [%d] %s",
                     attempts, max_attempts, mic.device_index, mic_name)
        logger.info("Recording %.1f s of audio ...", duration)

        try:
            with mic as source:
                # Warm-up: Intel Smart Sound mic array returns silence on the
                # first read.  Discard a short initial recording.
                logger.debug("Warming up microphone ...")
                recognizer.record(source, duration=0.3)
                logger.info("Recording for %.1f seconds ...", duration)
                audio = recognizer.record(source, duration=duration)
        except (OSError, AttributeError) as exc:
            logger.error("Microphone error during capture: %s", exc)
            idx = _next_device(idx)
            continue

        # --- Compute RMS energy (diagnostic only — no rejection) -----------
        rms = _rms(audio)
        dur = len(audio.frame_data) / audio.sample_width / audio.sample_rate
        logger.info(
            "Captured %d bytes @ %d Hz (%.1f s)  RMS=%.1f",
            len(audio.frame_data), audio.sample_rate, dur, rms,
        )

        # --- Zero-signal detection -----------------------------------------
        if rms == 0.0:
            logger.warning(
                "Audio RMS is 0.0 — device [%d] returned pure silence. "
                "This indicates a Windows privacy/permissions issue.",
                idx,
            )
            save_audio(audio, "output/debug_last_capture.wav")
            _SILENCE_DEVICE_INDEXES.append(idx)
            logger.warning(
                "Troubleshooting (Windows 11):\n"
                "  1. Settings > Privacy & security > Microphone\n"
                "     -> Enable 'Let apps access your microphone'\n"
                "  2. Scroll to 'Let desktop apps access your microphone'\n"
                "     -> Ensure Python / your terminal app is allowed.\n"
                "  3. Settings > System > Sound > Input\n"
                "     -> Select the correct mic and check it's not muted.\n"
                "  A debug WAV was saved to output/debug_last_capture.wav —\n"
                "  if it's silent audio, the OS is blocking the signal."
            )
            idx = _next_device(idx)
            continue

        # --- Audio captured successfully ----------------------------------
        logger.info("Audio captured (RMS=%.1f).", rms)
        return audio

    logger.error("All %d microphone attempts failed.", max_attempts)
    return None


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
