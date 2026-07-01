import logging
import threading
from typing import Optional

import pyttsx3
from pyttsx3 import Engine

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-speech engine backed by pyttsx3 (system TTS).

    Speaks text immediately through the default audio output device.
    All public methods are thread-safe.

    Usage
    -----
    >>> tts = TextToSpeech()
    >>> tts.speak("Hello, I am Nova.")
    """

    def __init__(self, voice_id: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        voice_id : str or None
            Identifier of the voice to use (e.g. ``"HKEY_LOCAL_MACHINE\\SOFTWARE\\..."
            on Windows).  ``None`` selects the system default voice.

        Raises
        ------
        RuntimeError
            If the system TTS engine cannot be initialised (e.g. no drivers
            available, missing SAPI5 on Windows, or no espeak on Linux).
        """
        try:
            self._engine: Engine = pyttsx3.init()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise TTS engine: {exc}") from exc

        self._lock = threading.Lock()

        voices = self._engine.getProperty("voices")
        if not voices:
            logger.warning("No TTS voices found — speech will be silent.")

        self._engine.setProperty("rate", 180)
        self._engine.setProperty("volume", 1.0)

        if voice_id is not None:
            self.set_voice(voice_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> bool:
        """
        Speak the given text immediately (blocks until finished).

        Parameters
        ----------
        text : str
            Text to be spoken aloud.  Empty or whitespace-only text is
            silently ignored.

        Returns
        -------
        bool
            ``True`` if the text was queued for speech, ``False`` if the
            text was empty or an error occurred.
        """
        cleaned = text.strip()
        if not cleaned:
            logger.info("Empty text — nothing to speak.")
            return False

        logger.debug("Speaking: %.80s%s", cleaned, "…" if len(cleaned) > 80 else "")

        with self._lock:
            try:
                self._engine.say(cleaned)
                self._engine.runAndWait()
            except RuntimeError as exc:
                logger.error("TTS engine error during speech: %s", exc)
                return False
            except Exception as exc:
                logger.error("Unexpected TTS error: %s", exc)
                return False

        return True

    def speak_async(self, text: str) -> bool:
        """
        Speak text in a background thread (non-blocking).

        Returns ``True`` if the text was queued, ``False`` if the text was
        empty or the engine is unavailable.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        thread = threading.Thread(target=self.speak, args=(cleaned,), daemon=True)
        thread.start()
        return True

    def set_voice(self, voice_id: str) -> bool:
        """
        Switch to a specific voice by identifier.

        Use ``list_voices()`` to discover available identifiers.
        Returns ``True`` on success, ``False`` if the voice ID is unknown.
        """
        with self._lock:
            try:
                self._engine.setProperty("voice", voice_id)
                logger.info("Switched to voice: %s", voice_id)
                return True
            except Exception as exc:
                logger.error("Failed to set voice: %s", exc)
                return False

    def set_rate(self, words_per_minute: int) -> None:
        """Set the speaking rate in words per minute (default 180)."""
        with self._lock:
            self._engine.setProperty("rate", max(50, min(words_per_minute, 400)))

    def set_volume(self, volume: float) -> None:
        """Set the volume (0.0 = silent, 1.0 = full)."""
        with self._lock:
            self._engine.setProperty("volume", max(0.0, min(volume, 1.0)))

    def list_voices(self) -> list[dict]:
        """
        Enumerate all available TTS voices.

        Returns a list of dicts with keys ``id``, ``name``, ``languages``,
        and ``gender``.
        """
        with self._lock:
            voices = self._engine.getProperty("voices")
        return [
            {
                "id": v.id,
                "name": v.name,
                "languages": v.languages,
                "gender": v.gender,
            }
            for v in voices
        ]

    def stop(self) -> None:
        """Stop any current speech immediately."""
        with self._lock:
            try:
                self._engine.stop()
            except Exception:
                pass
