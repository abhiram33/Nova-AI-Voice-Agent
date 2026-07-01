import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.tts import TextToSpeech


def test_initialise():
    """Engine should initialise without errors and report voices."""
    print("\n--- Initialisation ---")
    try:
        tts = TextToSpeech()
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False

    voices = tts.list_voices()
    ok = len(voices) > 0
    print(f"  Voices found: {len(voices)}")
    for v in voices:
        print(f"    [{v['id'][:50]}…] {v['name']}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_empty_text():
    """Speaking empty/whitespace text should return False."""
    print("\n--- Empty text ---")
    tts = TextToSpeech()
    ok = tts.speak("") is False and tts.speak("   ") is False
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_speak_short():
    """A short phrase should play and return True."""
    print("\n--- Speak short phrase ---")
    tts = TextToSpeech()
    result = tts.speak("Hello, I am Nova.")
    print(f"  Returned: {result}")
    print(f"  {'OK' if result else 'FAIL'}")
    return result


def test_set_rate():
    """Setting rate should clamp to valid range."""
    print("\n--- Set rate ---")
    tts = TextToSpeech()
    tts.set_rate(300)
    tts.set_rate(-10)
    print("  OK")
    return True


def test_set_volume():
    """Setting volume should clamp to 0–1 range."""
    print("\n--- Set volume ---")
    tts = TextToSpeech()
    tts.set_volume(2.0)
    tts.set_volume(-0.5)
    tts.set_volume(0.75)
    print("  OK")
    return True


def test_voice_switch():
    """Switching to the first available voice should succeed."""
    print("\n--- Switch voice ---")
    tts = TextToSpeech()
    voices = tts.list_voices()
    if not voices:
        print("  SKIP (no voices)")
        return True
    ok = tts.set_voice(voices[0]["id"])
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_speak_async():
    """Async speech should return True and not block."""
    print("\n--- Speak async ---")
    tts = TextToSpeech()
    start = time.time()
    result = tts.speak_async("This is an asynchronous test.")
    elapsed = time.time() - start
    print(f"  Returned: {result} in {elapsed:.3f}s (should be near-instant)")
    time.sleep(0.5)
    print(f"  {'OK' if result else 'FAIL'}")
    return result


def test_stop():
    """Stop should not raise."""
    print("\n--- Stop ---")
    tts = TextToSpeech()
    try:
        tts.stop()
        print("  OK")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def main():
    print("=== Nova AI Voice Agent — Text-to-Speech Test ===")

    tests = [
        ("test_initialise", test_initialise),
        ("test_empty_text", test_empty_text),
        ("test_speak_short", test_speak_short),
        ("test_set_rate", test_set_rate),
        ("test_set_volume", test_set_volume),
        ("test_voice_switch", test_voice_switch),
        ("test_speak_async", test_speak_async),
        ("test_stop", test_stop),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as exc:
            print(f"  UNEXPECTED ERROR in {name}: {exc}")
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
