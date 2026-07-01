import logging
import os
from pathlib import Path
from typing import Optional

import speech_recognition as sr
from dotenv import load_dotenv
from groq import Groq
from groq import APIError, AuthenticationError, RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

WHISPER_MODELS = ("whisper-large-v3", "whisper-large-v3-turbo")
DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"


class SpeechToText:
    """
    Speech-to-text engine backed by Groq's hosted Whisper API.

    Usage
    -----
    >>> stt = SpeechToText()
    >>> text = stt.transcribe(audio_data)
    >>> text = stt.transcribe_file("recording.wav")
    """

    def __init__(
        self,
        model: str = DEFAULT_WHISPER_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        model : str
            Whisper model to use. One of ``whisper-large-v3``,
            ``whisper-large-v3-turbo`` (default).
        api_key : str or None
            Groq API key. Falls back to ``GROQ_API_KEY`` env var.
        """
        if model not in WHISPER_MODELS:
            raise ValueError(
                f"Unknown Whisper model {model!r}. "
                f"Choose from {WHISPER_MODELS}"
            )

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Provide it via the constructor, set the environment variable, "
                "or add it to a .env file."
            )
        self._client = Groq(api_key=key)
        self._model = model
        logger.info("SpeechToText using Groq Whisper model %s", model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_data: Optional[sr.AudioData],
        language: str = "en",
    ) -> str:
        """
        Transcribe microphone ``AudioData`` to text via Groq Whisper API.

        Parameters
        ----------
        audio_data : AudioData or None
            Audio captured from a microphone.  ``None`` or empty audio
            produces an empty string.
        language : str
            ISO-639-1 language code (default ``"en"``).

        Returns
        -------
        str
            Transcript text (empty string if no speech was detected).
        """
        if audio_data is None:
            logger.info("No audio data provided — returning empty transcript.")
            return ""

        if len(audio_data.frame_data) == 0:
            logger.info("Empty audio frame data — returning empty transcript.")
            return ""

        wav_bytes = audio_data.get_wav_data()
        file = ("audio.wav", wav_bytes, "audio/wav")

        logger.info(
            "Sending %d bytes to Groq Whisper (%s)...",
            len(wav_bytes), self._model,
        )

        try:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=file,
                language=language,
                response_format="text",
            )
        except AuthenticationError as exc:
            logger.error("Groq Whisper auth failed — check your API key: %s", exc)
            raise
        except RateLimitError as exc:
            logger.error("Groq Whisper rate limit exceeded: %s", exc)
            raise
        except APIError as exc:
            logger.error("Groq Whisper API error — status=%s body=%s",
                         exc.status_code, exc.body if hasattr(exc, 'body') else 'N/A')
            raise
        except Exception as exc:
            logger.error("Groq Whisper unexpected error: %s", exc, exc_info=True)
            raise

        transcript = (response or "").strip()
        if not transcript:
            logger.info("Groq Whisper returned empty transcript (no speech detected).")
        else:
            logger.info("Transcription: %s", transcript)
        return transcript

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str = "en",
    ) -> str:
        """
        Transcribe a WAV file directly from disk via Groq Whisper API.

        Parameters
        ----------
        audio_path : str or Path
            Path to a WAV file on disk.
        language : str
            ISO-639-1 language code (default ``"en"``).

        Returns
        -------
        str
            Transcript text (empty string if no speech was detected).
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        file_size = path.stat().st_size
        logger.info(
            "Sending %s (%d bytes) to Groq Whisper (%s)...",
            path.name, file_size, self._model,
        )

        with open(path, "rb") as f:
            try:
                response = self._client.audio.transcriptions.create(
                    model=self._model,
                    file=f,
                    language=language,
                    response_format="text",
                )
            except AuthenticationError as exc:
                logger.error("Groq Whisper auth failed — check your API key: %s", exc)
                raise
            except RateLimitError as exc:
                logger.error("Groq Whisper rate limit exceeded: %s", exc)
                raise
            except APIError as exc:
                logger.error("Groq Whisper API error — status=%s body=%s",
                             exc.status_code, exc.body if hasattr(exc, 'body') else 'N/A')
                raise
            except Exception as exc:
                logger.error("Groq Whisper unexpected error: %s", exc, exc_info=True)
                raise

        transcript = (response or "").strip()
        if not transcript:
            logger.info("Groq Whisper returned empty transcript (no speech detected).")
        return transcript
