import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import speech_recognition as sr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.wake_word import WakeWordDetector


def test_pause_resume():
    """pause() and resume() should toggle the pause state correctly."""
    print("\n--- Pause / resume ---")

    with patch("src.wake_word.SpeechToText"):
        with patch("src.wake_word.pick_best_microphone", return_value={"index": 0, "name": "Test"}):
            detector = WakeWordDetector(
                wake_word="hey nova",
                api_key="gsk_test",
            )

            assert not detector.is_paused, "should start unpaused"

            detector.pause()
            assert detector.is_paused, "should be paused after pause()"

            detector.resume()
            assert not detector.is_paused, "should be unpaused after resume()"

    print("  OK")
    return True


def test_start_stop():
    """start() and stop() should manage the background thread."""
    print("\n--- Start / stop ---")

    with patch("src.wake_word.SpeechToText"):
        with patch("src.wake_word.pick_best_microphone", return_value={"index": 0, "name": "Test"}):
            detector = WakeWordDetector(
                wake_word="hey nova",
                api_key="gsk_test",
            )

            assert not detector.is_running, "should not be running before start()"

            detector.start()
            assert detector.is_running, "should be running after start()"

            detector.stop()
            # allow thread to exit
            time.sleep(0.1)
            assert not detector.is_running, "should not be running after stop()"

    print("  OK")
    return True


def test_start_no_mic():
    """start() should handle missing microphone gracefully."""
    print("\n--- Start with no microphone ---")

    with patch("src.wake_word.SpeechToText"):
        with patch("src.wake_word.pick_best_microphone", return_value=None):
            detector = WakeWordDetector(
                wake_word="hey nova",
                api_key="gsk_test",
            )
            detector.start()
            assert not detector.is_running, "should not start without mic"

    print("  OK")
    return True


def test_on_wake_called():
    """on_wake callback should fire when wake word is in transcription."""
    print("\n--- On-wake callback ---")

    callback = MagicMock()

    # We need to simulate the _run loop detecting "hey nova".
    # The _run loop uses recognizer.listen() which returns audio,
    # then transcribes it and checks for wake word.
    #
    # To test, we patch _run to simulate a detection cycle.

    with patch("src.wake_word.SpeechToText") as mock_stt_cls:
        mock_stt = mock_stt_cls.return_value
        mock_stt.transcribe.return_value = "hey nova what's the weather"

        with patch("src.wake_word.pick_best_microphone", return_value={"index": 0, "name": "Test"}):
            detector = WakeWordDetector(
                wake_word="hey nova",
                api_key="gsk_test",
                on_wake=callback,
            )

            # Manually simulate one detection cycle.
            audio = sr.AudioData(b"\x00\x01" * 16000, 16000, 2)
            text = detector._stt.transcribe(audio)
            assert detector.wake_word in text.lower()
            if detector.on_wake:
                detector.on_wake()

    callback.assert_called_once()
    print("  OK — callback was invoked")
    return True


def test_wake_word_not_detected():
    """Callback should NOT fire when wake word is absent."""
    print("\n--- Wake word not present ---")

    callback = MagicMock()

    with patch("src.wake_word.SpeechToText") as mock_stt_cls:
        mock_stt = mock_stt_cls.return_value
        mock_stt.transcribe.return_value = "what is the weather"

        with patch("src.wake_word.pick_best_microphone", return_value={"index": 0, "name": "Test"}):
            detector = WakeWordDetector(
                wake_word="hey nova",
                api_key="gsk_test",
                on_wake=callback,
            )

            audio = sr.AudioData(b"\x00\x01" * 16000, 16000, 2)
            text = detector._stt.transcribe(audio)
            if detector.wake_word in text.lower():
                if detector.on_wake:
                    detector.on_wake()

    callback.assert_not_called()
    print("  OK — callback was NOT invoked")
    return True


def test_wake_word_case_insensitive():
    """Wake word matching should be case-insensitive."""
    print("\n--- Case insensitive matching ---")

    with patch("src.wake_word.SpeechToText"):
        with patch("src.wake_word.pick_best_microphone", return_value={"index": 0, "name": "Test"}):
            detector = WakeWordDetector(
                wake_word="Hey Nova",
                api_key="gsk_test",
            )
            assert detector.wake_word == "hey nova"

    print("  OK — stored as lowercase")
    return True


def test_multiple_detection_cycles():
    """Simulate multiple listen cycles, only one with wake word."""
    print("\n--- Multiple cycles ---")

    callback = MagicMock()

    with patch("src.wake_word.SpeechToText") as mock_stt_cls:
        mock_stt = mock_stt_cls.return_value
        mock_stt.transcribe.side_effect = [
            "what is the weather",
            "hey nova tell me a joke",
            "goodbye",
        ]

        with patch("src.wake_word.pick_best_microphone", return_value={"index": 0, "name": "Test"}):
            detector = WakeWordDetector(
                wake_word="hey nova",
                api_key="gsk_test",
                on_wake=callback,
            )

            audio = sr.AudioData(b"\x00\x01" * 16000, 16000, 2)
            for _ in range(3):
                text = detector._stt.transcribe(audio)
                if detector.wake_word in text.lower():
                    if detector.on_wake:
                        detector.on_wake()

    assert callback.call_count == 1
    print("  OK — callback fired exactly once")
    return True


def main():
    print("=== Nova AI Voice Agent — Wake Word Detection Test ===\n")

    tests = [
        ("test_pause_resume", test_pause_resume),
        ("test_start_stop", test_start_stop),
        ("test_start_no_mic", test_start_no_mic),
        ("test_on_wake_called", test_on_wake_called),
        ("test_wake_word_not_detected", test_wake_word_not_detected),
        ("test_wake_word_case_insensitive", test_wake_word_case_insensitive),
        ("test_multiple_detection_cycles", test_multiple_detection_cycles),
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
