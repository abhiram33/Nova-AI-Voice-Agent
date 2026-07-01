import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import VoicePipeline, TurnResult, ConversationMemory


def test_turn_result_dataclass():
    """Verify TurnResult fields default correctly."""
    print("\n--- TurnResult defaults ---")

    r = TurnResult()
    assert r.success is False
    assert r.transcript == ""
    assert r.response == ""
    assert r.error is None
    print("  OK — all defaults correct")

    r2 = TurnResult(success=True, transcript="hi", response="hello")
    assert r2.success is True
    assert r2.transcript == "hi"
    assert r2.response == "hello"
    assert r2.error is None
    print("  OK — explicit fields work")
    return True


def test_conversation_memory():
    """Verify ConversationMemory add, get, clear, and max limit."""
    print("\n--- ConversationMemory ---")

    mem = ConversationMemory(max_messages=4)
    assert mem.count == 0
    assert len(mem) == 0

    mem.add("user", "hi")
    mem.add("assistant", "hello")
    assert mem.count == 2
    assert len(mem.get_history()) == 2

    mem.add("user", "how are you")
    mem.add("assistant", "good")
    assert mem.count == 4

    mem.add("user", "whats up")
    assert mem.count == 4
    assert mem.get_history()[0] == {"role": "assistant", "content": "hello"}
    assert mem.get_history()[-1]["content"] == "whats up"

    mem.clear()
    assert mem.count == 0
    assert mem.get_history() == []

    print("  OK — add, get, clear, and max limit all work")
    return True


@patch("src.pipeline.SpeechToText")
@patch("src.pipeline.GroqClient")
@patch("src.pipeline.TextToSpeech")
@patch("src.pipeline.save_audio", return_value=True)
def test_successful_turn(mock_save, mock_tts_cls, mock_groq_cls, mock_stt_cls):
    """A complete turn should capture → transcribe → LLM → speak."""
    print("\n--- Successful turn (mocked) ---")

    stt_instance = mock_stt_cls.return_value
    stt_instance.transcribe.return_value = "what is the weather"

    groq_instance = mock_groq_cls.return_value
    groq_instance.generate_with_tools.return_value = (
        "It is sunny today."
    )

    tts_instance = mock_tts_cls.return_value
    tts_instance.speak.return_value = True

    with patch("src.pipeline.capture_audio") as mock_capture:
        mock_audio = MagicMock()
        mock_audio.frame_data = b"\x00\x00" * 16000
        mock_audio.sample_rate = 16000
        mock_audio.sample_width = 2
        mock_capture.return_value = mock_audio

        pipeline = VoicePipeline()
        result = pipeline.run_once()

    assert result.success is True
    assert result.transcript == "what is the weather"
    assert result.response == "It is sunny today."
    assert result.error is None

    stt_instance.transcribe.assert_called_once_with(mock_audio)
    groq_instance.generate_with_tools.assert_called_once()
    tts_instance.speak.assert_called_once_with("It is sunny today.")

    assert pipeline.memory.count == 2
    assert pipeline.memory.get_history()[0] == {"role": "user", "content": "what is the weather"}
    assert pipeline.memory.get_history()[1] == {"role": "assistant", "content": "It is sunny today."}

    print("  OK — pipeline orchestrated all 4 steps and saved to memory")
    return True


@patch("src.pipeline.SpeechToText")
@patch("src.pipeline.GroqClient")
@patch("src.pipeline.TextToSpeech")
def test_no_speech_detected(mock_tts_cls, mock_groq_cls, mock_stt_cls):
    """When capture_audio returns None, the pipeline should abort early."""
    print("\n--- No speech detected ---")

    with patch("src.pipeline.capture_audio") as mock_capture:
        mock_capture.return_value = None
        pipeline = VoicePipeline()
        result = pipeline.run_once()

    assert result.success is False
    assert result.transcript == ""
    assert result.response == ""
    assert result.error is not None
    assert "speech" in (result.error or "").lower()

    mock_stt_cls.return_value.transcribe.assert_not_called()
    mock_groq_cls.return_value.generate_with_tools.assert_not_called()
    mock_tts_cls.return_value.speak.assert_not_called()

    print("  OK — pipeline aborted before STT")
    return True


@patch("src.pipeline.SpeechToText")
@patch("src.pipeline.GroqClient")
@patch("src.pipeline.TextToSpeech")
@patch("src.pipeline.save_audio", return_value=True)
def test_transcription_empty(mock_save, mock_tts_cls, mock_groq_cls, mock_stt_cls):
    """When STT returns empty string, the pipeline should abort."""
    print("\n--- Empty transcription ---")

    stt_instance = mock_stt_cls.return_value
    stt_instance.transcribe.return_value = ""

    with patch("src.pipeline.capture_audio") as mock_capture:
        mock_audio = MagicMock()
        mock_audio.frame_data = b"\x00\x00" * 16000
        mock_audio.sample_rate = 16000
        mock_audio.sample_width = 2
        mock_capture.return_value = mock_audio

        pipeline = VoicePipeline()
        result = pipeline.run_once()

    assert result.success is False
    assert result.transcript == ""
    assert result.error is not None

    mock_groq_cls.return_value.generate_with_tools.assert_not_called()
    mock_tts_cls.return_value.speak.assert_not_called()

    print("  OK — pipeline aborted before Groq")
    return True


@patch("src.pipeline.SpeechToText")
@patch("src.pipeline.GroqClient")
@patch("src.pipeline.TextToSpeech")
@patch("src.pipeline.save_audio", return_value=True)
def test_groq_failure(mock_save, mock_tts_cls, mock_groq_cls, mock_stt_cls):
    """When Groq raises, the pipeline should capture transcript but abort."""
    print("\n--- Groq API failure ---")

    stt_instance = mock_stt_cls.return_value
    stt_instance.transcribe.return_value = "hello"

    groq_instance = mock_groq_cls.return_value
    groq_instance.generate_with_tools.side_effect = RuntimeError("API down")

    with patch("src.pipeline.capture_audio") as mock_capture:
        mock_audio = MagicMock()
        mock_audio.frame_data = b"\x00\x00" * 16000
        mock_audio.sample_rate = 16000
        mock_audio.sample_width = 2
        mock_capture.return_value = mock_audio

        pipeline = VoicePipeline()
        result = pipeline.run_once()

    assert result.success is False
    assert result.transcript == "hello"
    assert result.response == ""
    assert result.error is not None

    mock_tts_cls.return_value.speak.assert_not_called()

    print("  OK — Groq error caught, transcript preserved")
    return True


@patch("src.pipeline.SpeechToText")
@patch("src.pipeline.GroqClient")
@patch("src.pipeline.TextToSpeech")
@patch("src.pipeline.save_audio", return_value=True)
def test_tts_failure(mock_save, mock_tts_cls, mock_groq_cls, mock_stt_cls):
    """TTS failure should still yield a successful TurnResult with text."""
    print("\n--- TTS failure (graceful degradation) ---")

    stt_instance = mock_stt_cls.return_value
    stt_instance.transcribe.return_value = "hello"

    groq_instance = mock_groq_cls.return_value
    groq_instance.generate_with_tools.return_value = "Hi there!"

    tts_instance = mock_tts_cls.return_value
    tts_instance.speak.return_value = False

    with patch("src.pipeline.capture_audio") as mock_capture:
        mock_audio = MagicMock()
        mock_audio.frame_data = b"\x00\x00" * 16000
        mock_audio.sample_rate = 16000
        mock_audio.sample_width = 2
        mock_capture.return_value = mock_audio

        pipeline = VoicePipeline()
        result = pipeline.run_once()

    assert result.success is True
    assert result.transcript == "hello"
    assert result.response == "Hi there!"
    assert result.error is None

    print("  OK — pipeline succeeded despite TTS failure")
    return True


@patch("src.pipeline.SpeechToText")
@patch("src.pipeline.GroqClient")
@patch("src.pipeline.TextToSpeech")
@patch("src.pipeline.save_audio", return_value=True)
def test_memory_accumulates_across_turns(mock_save, mock_tts_cls, mock_groq_cls, mock_stt_cls):
    """Multiple turns should accumulate messages in memory."""
    print("\n--- Memory accumulates across turns ---")

    stt_instance = mock_stt_cls.return_value
    groq_instance = mock_groq_cls.return_value
    tts_instance = mock_tts_cls.return_value

    stt_instance.transcribe.side_effect = ["first turn", "second turn", "goodbye"]
    groq_instance.generate_with_tools.side_effect = [
        "first reply",
        "second reply",
        "bye",
    ]
    tts_instance.speak.return_value = True

    with patch("src.pipeline.capture_audio") as mock_capture:
        mock_audio = MagicMock()
        mock_audio.frame_data = b"\x00\x00" * 16000
        mock_audio.sample_rate = 16000
        mock_audio.sample_width = 2
        mock_capture.return_value = mock_audio

        pipeline = VoicePipeline()
        pipeline.run_interactive(turns=3)

    assert pipeline.memory.count == 6
    assert pipeline.memory.get_history()[0]["content"] == "first turn"
    assert pipeline.memory.get_history()[1]["content"] == "first reply"
    assert pipeline.memory.get_history()[4]["content"] == "goodbye"
    assert pipeline.memory.get_history()[5]["content"] == "bye"

    print("  OK — memory has 6 messages across 3 turns")
    return True


@patch("src.pipeline.SpeechToText")
@patch("src.pipeline.GroqClient")
@patch("src.pipeline.TextToSpeech")
@patch("src.pipeline.save_audio", return_value=True)
def test_clear_memory(mock_save, mock_tts_cls, mock_groq_cls, mock_stt_cls):
    """clear_memory() should erase all stored messages."""
    print("\n--- Clear memory ---")

    stt_instance = mock_stt_cls.return_value
    groq_instance = mock_groq_cls.return_value
    tts_instance = mock_tts_cls.return_value

    stt_instance.transcribe.return_value = "hello"
    groq_instance.generate_with_tools.return_value = "hi"
    tts_instance.speak.return_value = True

    with patch("src.pipeline.capture_audio") as mock_capture:
        mock_audio = MagicMock()
        mock_audio.frame_data = b"\x00\x00" * 16000
        mock_audio.sample_rate = 16000
        mock_audio.sample_width = 2
        mock_capture.return_value = mock_audio

        pipeline = VoicePipeline()
        pipeline.run_once()
        assert pipeline.memory.count == 2

        pipeline.clear_memory()
        assert pipeline.memory.count == 0
        assert pipeline.memory.get_history() == []

    print("  OK — memory cleared after 1 turn")
    return True


def test_interactive_stop_phrase():
    """run_interactive should exit on a stop phrase."""
    print("\n--- Interactive mode stop phrase ---")
    with (
        patch("src.pipeline.SpeechToText") as mock_stt_cls,
        patch("src.pipeline.GroqClient") as mock_groq_cls,
        patch("src.pipeline.TextToSpeech") as mock_tts_cls,
        patch("src.pipeline.capture_audio") as mock_capture,
        patch("src.pipeline.save_audio", return_value=True),
    ):
        stt_instance = mock_stt_cls.return_value
        stt_instance.transcribe.return_value = "goodbye"

        groq_instance = mock_groq_cls.return_value
        groq_instance.generate_with_tools.return_value = "Goodbye!"

        tts_instance = mock_tts_cls.return_value
        tts_instance.speak.return_value = True

        mock_audio = MagicMock()
        mock_audio.frame_data = b"\x00\x00" * 16000
        mock_audio.sample_rate = 16000
        mock_audio.sample_width = 2
        mock_capture.return_value = mock_audio

        pipeline = VoicePipeline()
        pipeline.run_interactive(turns=10)

        assert tts_instance.speak.call_count >= 2

    print("  OK — interactive loop stopped on 'goodbye'")
    return True


def main():
    print("=== Nova AI Voice Agent — Pipeline Test ===")

    tests = [
        ("test_turn_result_dataclass", test_turn_result_dataclass),
        ("test_conversation_memory", test_conversation_memory),
        ("test_successful_turn", test_successful_turn),
        ("test_no_speech_detected", test_no_speech_detected),
        ("test_transcription_empty", test_transcription_empty),
        ("test_groq_failure", test_groq_failure),
        ("test_tts_failure", test_tts_failure),
        ("test_memory_accumulates_across_turns", test_memory_accumulates_across_turns),
        ("test_clear_memory", test_clear_memory),
        ("test_interactive_stop_phrase", test_interactive_stop_phrase),
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
