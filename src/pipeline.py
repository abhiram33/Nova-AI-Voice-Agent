import logging
from dataclasses import dataclass, field
from typing import Optional

from src.groq_client import GroqClient
from src.microphone import capture_audio, save_audio
from src.stt import SpeechToText
from src.tools import TOOL_DEFINITIONS, execute_tool
from src.tts import TextToSpeech

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Bounded conversation history for the voice pipeline.

    Stores alternating ``user`` / ``assistant`` messages and automatically
    discards the oldest entries when the limit is exceeded.
    """

    def __init__(self, max_messages: int = 10) -> None:
        self._max = max_messages
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append a message and trim to ``max_messages``."""
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._max:
            self._messages.pop(0)

    def get_history(self) -> list[dict]:
        """Return a copy of the stored message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all stored messages."""
        self._messages.clear()

    @property
    def count(self) -> int:
        return len(self._messages)

    def __len__(self) -> int:
        return len(self._messages)


@dataclass
class TurnResult:
    """
    Result of a single voice conversation turn.

    Attributes
    ----------
    success : bool
        ``True`` when the full pipeline completed (transcript → LLM → TTS).
    transcript : str
        What the user said (empty if no speech was detected).
    response : str
        What the AI replied (empty if the LLM call failed or returned nothing).
    audio_path : str or None
        Path to the saved WAV file of the user's speech, if available.
    error : str or None
        Human-readable error description when ``success`` is ``False``.
    """
    success: bool = False
    transcript: str = ""
    response: str = ""
    audio_path: Optional[str] = None
    error: Optional[str] = None


DEFAULT_SYSTEM_PROMPT = (
    "You are Nova, a helpful voice AI assistant. "
    "Keep responses concise and conversational since they will be spoken aloud. "
    "Avoid markdown, bullet points, or code blocks. "
    "You have access to tools: web_search (search the web), "
    "calculate (perform math), and get_weather (check weather). "
    "When using a tool, answer from the result directly and concisely. "
    "Never call the same tool more than once per user request."
)


class VoicePipeline:
    """
    Orchestrates a complete voice conversation turn:

        Microphone → Speech-to-Text → Groq LLM → Text-to-Speech

    Usage
    -----
    >>> pipeline = VoicePipeline()
    >>> result = pipeline.run_once()
    >>> print(result.transcript, result.response)
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        stt_model: str = "whisper-large-v3-turbo",
        tts_voice_id: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        recording_duration: float = 10.0,
        phrase_time_limit: Optional[float] = 8.0,
        device_index: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        groq_api_key : str or None
            Groq API key (falls back to ``GROQ_API_KEY`` env var).
            Used for both LLM and Whisper transcription.
        stt_model : str
            Groq Whisper model (``whisper-large-v3`` or
            ``whisper-large-v3-turbo``, default).
        tts_voice_id : str or None
            TTS voice identifier (``None`` = system default).
        system_prompt : str
            System-level instruction for the LLM.
        recording_duration : float
            Max seconds to wait for speech to **start** (default 10).
        phrase_time_limit : float or None
            Max seconds for a single spoken phrase (default 8).
            ``None`` uses *recording_duration*.
        device_index : int or None
            Microphone device index (``None`` = auto-detect best mic).
        """
        self.system_prompt = system_prompt
        self.recording_duration = recording_duration
        self.phrase_time_limit = phrase_time_limit
        self.device_index = device_index
        self.memory = ConversationMemory(max_messages=10)

        logger.info("Initialising VoicePipeline components...")
        self._stt = SpeechToText(
            model=stt_model,
            api_key=groq_api_key,
        )
        self._groq = GroqClient(api_key=groq_api_key)
        self._tts = TextToSpeech(voice_id=tts_voice_id)
        logger.info("VoicePipeline ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self) -> TurnResult:
        """
        Execute one full conversation turn.

        1. Record audio from the microphone.
        2. Transcribe speech to text.
        3. Send transcript to Groq.
        4. Speak the LLM response aloud.

        Returns a ``TurnResult`` describing what happened.
        """
        # --- Step 1: Capture audio -------------------------------------------
        logger.info("Step 1/4 — Capturing audio...")
        audio = capture_audio(
            duration=self.recording_duration,
            phrase_time_limit=self.phrase_time_limit,
            device_index=self.device_index,
        )
        if audio is None:
            return TurnResult(
                error="No speech detected or microphone unavailable.",
            )

        audio_path = f"output/turn_{id(audio)}.wav"
        save_audio(audio, audio_path)
        save_audio(audio, "output/last_recording.wav")

        # --- Step 2: Speech-to-text ------------------------------------------
        logger.info("Step 2/4 — Transcribing...")
        try:
            transcript = self._stt.transcribe(audio)
        except Exception as exc:
            logger.error("Transcription failed: %s", exc, exc_info=True)
            return TurnResult(
                audio_path=audio_path,
                error=f"Transcription error: {exc}",
            )
        logger.info("Transcript: %s", transcript)
        if not transcript:
            logger.warning("Transcription returned empty text.")
            return TurnResult(
                audio_path=audio_path,
                error="Speech was detected but could not be transcribed.",
            )

        # --- Step 3: Groq LLM (with tool calling) ---------------------------
        logger.info("Step 3/4 — Sending to Groq...")
        try:
            response = self._groq.generate_with_tools(
                prompt=transcript,
                tools=TOOL_DEFINITIONS,
                tool_executor=execute_tool,
                system_prompt=self.system_prompt,
                history=self.memory.get_history(),
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as exc:
            msg = f"Groq API error: {exc}"
            logger.error(msg)
            return TurnResult(transcript=transcript, error=msg)

        if not response:
            logger.warning("Groq returned an empty response.")
            return TurnResult(transcript=transcript, error="LLM returned empty response.")

        logger.info("Response: %s", response)

        # --- Persist to memory ------------------------------------------------
        self.memory.add("user", transcript)
        self.memory.add("assistant", response)

        # --- Step 4: Text-to-speech ------------------------------------------
        logger.info("Step 4/4 — Speaking response...")
        tts_ok = self._tts.speak(response)
        if not tts_ok:
            logger.warning("TTS reported failure — text response still available.")

        return TurnResult(
            success=True,
            transcript=transcript,
            response=response,
            audio_path=audio_path,
            error=None,
        )

    def stop_tts(self) -> None:
        """Stop any currently playing TTS audio."""
        self._tts.stop()

    def clear_memory(self) -> None:
        """Erase all stored conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def run_interactive(self, turns: int = 0) -> None:
        """
        Run conversation turns in a loop.

        Parameters
        ----------
        turns : int
            Number of turns.  ``0`` (default) runs indefinitely until
            the user says a stop phrase or interrupts with Ctrl+C.
        """
        stop_phrases = {"exit", "quit", "goodbye", "stop", "that's all"}
        count = 0

        logger.info("Starting interactive session (say 'exit' to stop)...")
        while True:
            if turns > 0 and count >= turns:
                break

            result = self.run_once()
            count += 1

            if result.success:
                logger.info(
                    "Turn %d complete — transcript=%r response=%r",
                    count, result.transcript, result.response,
                )
                if result.transcript.strip().lower() in stop_phrases:
                    logger.info("Stop phrase detected — ending session.")
                    self._tts.speak("Goodbye!")
                    break
            else:
                logger.warning("Turn %d failed: %s", count, result.error)
                if "speech" in (result.error or "").lower():
                    self._tts.speak("I didn't catch that. Could you repeat?")

            if turns == 0:
                logger.info("Listening for next turn...")
