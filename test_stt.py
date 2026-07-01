import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import speech_recognition as sr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.stt import SpeechToText


def _make_audio(duration_sec: float = 2.0, sample_rate: int = 16000) -> sr.AudioData:
    """Create an ``sr.AudioData`` object containing silence."""
    num_frames = int(duration_sec * sample_rate)
    frame_data = b"\x00\x00" * num_frames
    return sr.AudioData(frame_data, sample_rate, sample_width=2)


def _make_noisy_audio(duration_sec: float = 2.0, sample_rate: int = 16000) -> sr.AudioData:
    """Create an ``sr.AudioData`` object with non-zero audio content."""
    import numpy as np
    samples = (np.random.randn(int(duration_sec * sample_rate)) * 5000).astype(np.int16).tobytes()
    return sr.AudioData(samples, sample_rate, sample_width=2)


def test_invalid_model():
    """An unknown model name should raise ValueError."""
    print("\n--- Invalid model ---")
    try:
        SpeechToText(model="nonexistent")
        print("  FAIL: no exception raised")
        return False
    except ValueError as exc:
        print(f"  OK: ValueError raised — {exc}")
        return True


def test_missing_api_key():
    """Construction without key and without env var should raise ValueError."""
    print("\n--- Missing API key ---")
    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        SpeechToText(api_key=None)
        print("  FAIL: no exception raised")
        return False
    except ValueError:
        print("  OK: ValueError raised")
        return True
    finally:
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved


def test_transcribe_none():
    """Passing None should return an empty string."""
    print("\n--- None input ---")
    with patch("src.stt.Groq") as mock_groq_cls:
        stt = SpeechToText(api_key="gsk_test")
        result = stt.transcribe(None)
    ok = result == ""
    print(f"  Result: {result!r} — {'OK' if ok else 'FAIL'}")
    return ok


def test_transcribe_empty_audio():
    """Empty audio frame data should return empty string."""
    print("\n--- Empty audio ---")
    with patch("src.stt.Groq") as mock_groq_cls:
        stt = SpeechToText(api_key="gsk_test")
        empty = sr.AudioData(b"", 16000, 2)
        result = stt.transcribe(empty)
    ok = result == ""
    print(f"  Result: {result!r} — {'OK' if ok else 'FAIL'}")
    return ok


def test_transcribe_success():
    """Valid audio should be sent to Groq Whisper and return the transcript."""
    print("\n--- Successful transcription ---")
    mock_groq_instance = MagicMock()
    mock_groq_instance.audio.transcriptions.create.return_value = "hello world"

    with patch("src.stt.Groq", return_value=mock_groq_instance):
        stt = SpeechToText(api_key="gsk_test")
        audio = _make_noisy_audio()
        result = stt.transcribe(audio)

    ok = result == "hello world"
    print(f"  Result: {result!r} — {'OK' if ok else 'FAIL'}")

    mock_groq_instance.audio.transcriptions.create.assert_called_once()
    call_kwargs = mock_groq_instance.audio.transcriptions.create.call_args[1]
    print(f"  Model: {call_kwargs.get('model')}")
    print(f"  Language: {call_kwargs.get('language')}")
    print(f"  Format: {call_kwargs.get('response_format')}")
    ok = ok and call_kwargs.get("model") == "whisper-large-v3-turbo"
    ok = ok and call_kwargs.get("language") == "en"
    ok = ok and call_kwargs.get("response_format") == "text"
    return ok


def test_transcribe_empty_response():
    """When Groq Whisper returns empty string, transcribe should return empty."""
    print("\n--- Empty Groq response ---")
    mock_groq_instance = MagicMock()
    mock_groq_instance.audio.transcriptions.create.return_value = ""

    with patch("src.stt.Groq", return_value=mock_groq_instance):
        stt = SpeechToText(api_key="gsk_test")
        audio = _make_noisy_audio()
        result = stt.transcribe(audio)

    ok = result == ""
    print(f"  Result: {result!r} — {'OK' if ok else 'FAIL'}")
    return ok


def test_transcribe_file_not_found():
    """A missing file path should raise FileNotFoundError."""
    print("\n--- Missing file ---")
    with patch("src.stt.Groq") as mock_groq_cls:
        stt = SpeechToText(api_key="gsk_test")
        try:
            stt.transcribe_file("does_not_exist.wav")
            print("  FAIL: no exception raised")
            return False
        except FileNotFoundError:
            print("  OK: FileNotFoundError raised")
            return True


def test_transcribe_file_success():
    """Valid file should be sent to Groq Whisper and return transcript."""
    print("\n--- Transcribe file ---")
    import wave
    import numpy as np

    wav_path = Path("output/test_groq_stt.wav")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    samples = (np.random.randn(sample_rate * 2) * 5000).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    mock_groq_instance = MagicMock()
    mock_groq_instance.audio.transcriptions.create.return_value = "file transcript"

    with patch("src.stt.Groq", return_value=mock_groq_instance):
        stt = SpeechToText(api_key="gsk_test")
        result = stt.transcribe_file(str(wav_path))

    ok = result == "file transcript"
    print(f"  Result: {result!r} — {'OK' if ok else 'FAIL'}")

    mock_groq_instance.audio.transcriptions.create.assert_called_once()
    print("  OK — Groq API called")
    return ok


def main():
    print("=== Nova AI Voice Agent — Speech-to-Text Test ===")

    tests = [
        ("test_invalid_model", test_invalid_model),
        ("test_missing_api_key", test_missing_api_key),
        ("test_transcribe_none", test_transcribe_none),
        ("test_transcribe_empty_audio", test_transcribe_empty_audio),
        ("test_transcribe_success", test_transcribe_success),
        ("test_transcribe_empty_response", test_transcribe_empty_response),
        ("test_transcribe_file_not_found", test_transcribe_file_not_found),
        ("test_transcribe_file_success", test_transcribe_file_success),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  UNEXPECTED ERROR: {exc}")
            results.append((name, False))

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n{passed}/{total} tests passed")


if __name__ == "__main__":
    main()
