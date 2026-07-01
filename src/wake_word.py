import logging
import threading
import time
from typing import Callable, Optional

import speech_recognition as sr

from src.microphone import pick_best_microphone, _rms
from src.stt import SpeechToText

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Continuously listens for a wake word in a background thread.

    Uses ``speech_recognition``'s VAD-based ``listen()`` to capture short
    audio clips when speech is detected, then sends them to Groq Whisper
    for transcription.  When the wake word appears in the transcript the
    ``on_wake`` callback is invoked.

    Microphone access is cooperative: the detector opens the mic only
    inside each listen cycle, so the mic is free for the pipeline when
    the detector is paused or between cycles.
    """

    def __init__(
        self,
        wake_word: str = "hey nova",
        api_key: Optional[str] = None,
        on_wake: Optional[Callable[[], None]] = None,
        chunk_duration: float = 2.0,
        listen_timeout: float = 0.5,
        energy_threshold: int = 300,
    ) -> None:
        """
        Parameters
        ----------
        wake_word : str
            Phrase to listen for (case-insensitive, default ``"hey nova"``).
        api_key : str or None
            Groq API key (falls back to ``GROQ_API_KEY`` env var).
        on_wake : callable or None
            Called from the background thread when the wake word is heard.
        chunk_duration : float
            Max seconds to record once speech starts (default 2).
        listen_timeout : float
            Seconds to wait for speech before giving up (default 0.5).
        energy_threshold : int
            Minimum RMS to consider the audio as non-silence (default 300).
        """
        self.wake_word = wake_word.lower().strip()
        self.on_wake = on_wake
        self.chunk_duration = chunk_duration
        self.listen_timeout = listen_timeout
        self.energy_threshold = energy_threshold

        self._stt = SpeechToText(model="whisper-large-v3-turbo", api_key=api_key)
        self._running = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused by default
        self._thread: Optional[threading.Thread] = None
        self._mic_index: Optional[int] = None
        self._last_transcribe_time: float = 0.0
        self._cooldown = 3.0  # seconds between transcriptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background wake-word detection thread."""
        if self._running:
            return

        best = pick_best_microphone()
        if best is None:
            logger.error("No microphone available — cannot start wake word detection.")
            return
        self._mic_index = best["index"]
        logger.info("Wake word detector using microphone [%d] %s", self._mic_index, best["name"])

        self._running = True
        self._pause_event.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Wake word detection started — listening for '%s'", self.wake_word)

    def stop(self) -> None:
        """Stop the background thread."""
        self._running = False
        self._pause_event.set()  # unblock if paused
        logger.info("Wake word detection stopped.")

    def pause(self) -> None:
        """Pause detection so the pipeline can use the microphone."""
        self._pause_event.clear()
        logger.debug("Wake word detector paused.")

    def resume(self) -> None:
        """Resume detection after the pipeline is done."""
        self._pause_event.set()
        logger.debug("Wake word detector resumed.")

    @property
    def is_paused(self) -> bool:
        """``True`` while wake word detection is paused."""
        return not self._pause_event.is_set()

    @property
    def is_running(self) -> bool:
        """``True`` while the background thread is alive."""
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop — runs in a daemon thread."""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.energy_threshold
        recognizer.dynamic_energy_threshold = False
        recognizer.pause_threshold = 0.5

        # Calibrate once at startup (discards first-silence warm-up).
        try:
            with sr.Microphone(device_index=self._mic_index, sample_rate=16000) as source:
                recognizer.record(source, duration=0.3)  # warm-up
                logger.info("Calibrating wake word detector for ambient noise ...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                logger.info(
                    "Wake word energy threshold: %d",
                    recognizer.energy_threshold,
                )
        except (OSError, AttributeError) as exc:
            logger.error("Wake word mic calibration failed: %s", exc)
            self._running = False
            return

        while self._running:
            # Block while paused (mic is free for the pipeline).
            self._pause_event.wait()

            try:
                with sr.Microphone(device_index=self._mic_index, sample_rate=16000) as source:
                    audio = recognizer.listen(
                        source,
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.chunk_duration,
                    )
            except sr.WaitTimeoutError:
                continue  # no speech in this window
            except (OSError, AttributeError) as exc:
                logger.error("Wake word listen error: %s", exc)
                time.sleep(0.5)
                continue

            # Quick energy gate — skip obvious silence without an API call.
            rms = _rms(audio)
            if rms < self.energy_threshold:
                continue

            # Rate limit — don't hit Groq Whisper API more than once per cooldown.
            now = time.monotonic()
            if now - self._last_transcribe_time < self._cooldown:
                continue
            self._last_transcribe_time = now

            # Transcribe via Groq Whisper.
            try:
                text = self._stt.transcribe(audio)
            except Exception as exc:
                logger.error("Wake word transcription error: %s", exc, exc_info=True)
                continue

            if not text:
                continue

            logger.debug("Wake word detector heard: %s", text.strip())

            if self.wake_word in text.lower().strip():
                logger.info(
                    "Wake word '%s' detected in: %s", self.wake_word, text.strip()
                )
                if self.on_wake:
                    self.on_wake()
