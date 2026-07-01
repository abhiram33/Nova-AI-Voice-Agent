import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.microphone import (
    list_microphones,
    pick_best_microphone,
    get_microphone,
    capture_audio,
    save_audio,
    record_and_save,
)


def test_list_microphones() -> list[dict]:
    """Display every available microphone device."""
    devices = list_microphones()
    if devices:
        print(f"\nFound {len(devices)} microphone(s):")
        for d in devices:
            print(f"  [{d['index']}] {d['name']}")
    else:
        print("\nNo microphones detected.")
    return devices


def test_pick_best(devices: list[dict]):
    """Auto-detect the best physical microphone."""
    print("\n--- Auto-detect best microphone ---")
    best = pick_best_microphone()
    if best:
        print(f"  Selected: [{best['index']}] {best['name']}")
        for d in devices:
            if d["index"] == best["index"]:
                print(f"    -> {d['name']}")
                break
    else:
        print("  No suitable microphone found.")
    return best


def test_get_microphone(device_index=None):
    """Confirm we can open a microphone handle."""
    mic = get_microphone(device_index)
    if mic:
        name = Path(mic.__repr__())
        print(f"  Microphone opened: device_index={mic.device_index}")
    else:
        print("  Failed to open microphone.")
    return mic


def test_capture_audio(duration=6, device_index=None):
    """Record a fixed-duration clip and return the AudioData."""
    print(f"\nRecording for {duration} seconds — speak now...")
    audio = capture_audio(
        duration=duration,
        device_index=device_index,
    )
    if audio:
        dur = len(audio.frame_data) / audio.sample_width / audio.sample_rate
        print(f"  Captured {len(audio.frame_data)} bytes @ {audio.sample_rate} Hz ({dur:.1f}s)")
    else:
        print("  Audio RMS below threshold — likely silence.")
        print("  Listen to output/debug_ambient.wav to verify mic signal.")
    return audio


def test_save_audio(audio_data, path="output/test_recording.wav"):
    """Write AudioData to a WAV file and verify it exists."""
    ok = save_audio(audio_data, path)
    if ok:
        size = Path(path).stat().st_size
        print(f"  Saved {size} bytes to {path}")
    else:
        print("  Save failed.")
    return ok


def test_record_and_save():
    """Exercise the convenience wrapper."""
    print("\n--- record_and_save ---")
    path = "output/test_convenience.wav"
    ok = record_and_save(filepath=path, duration=10)
    if ok:
        size = Path(path).stat().st_size
        print(f"  Saved {size} bytes to {path}")
    else:
        print("  record_and_save returned False")
    return ok


def test_invalid_device():
    """Verify graceful handling of a non-existent device index."""
    print("\n--- Invalid device test ---")
    mic = get_microphone(device_index=9999)
    if mic is None:
        print("  Correctly returned None for invalid device index.")
    else:
        print("  Unexpectedly got a microphone (unusual).")
    return mic


def main():
    print("=== Nova AI Voice Agent — Microphone Test ===")

    devices = test_list_microphones()
    best = test_pick_best(devices)

    print("\n--- Opening best microphone ---")
    test_get_microphone(best["index"] if best else None)

    print("\n--- Opening invalid device ---")
    test_invalid_device()

    print("\n" + "=" * 50)
    print("CAPTURE TEST — speak when prompted")
    print("=" * 50)

    audio = test_capture_audio(
        duration=6,
        device_index=best["index"] if best else None,
    )
    if audio:
        test_save_audio(audio)

    print("\n=== All test functions completed ===")
    print("If capture failed, check:")
    print("  1. output/debug_ambient.wav — does it contain audio?")
    print("  2. The selected microphone index matches your actual mic.")
    print("  3. Windows privacy settings allow mic access.")


if __name__ == "__main__":
    main()
