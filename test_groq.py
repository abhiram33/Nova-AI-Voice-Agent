import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.groq_client import GroqClient
from groq import AuthenticationError, RateLimitError, APIError


def test_missing_api_key():
    """Construction without a key and without env var should raise ValueError."""
    print("\n--- Missing API key ---")
    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        GroqClient(api_key=None)
        print("  FAIL: no exception raised")
        return False
    except ValueError:
        print("  OK: ValueError raised")
        return True
    finally:
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved


def test_invalid_api_key():
    """An invalid key should raise AuthenticationError."""
    print("\n--- Invalid API key ---")
    try:
        client = GroqClient(api_key="gsk_invalid")
        client.generate_response("Hello")
        print("  FAIL: no exception raised")
        return False
    except AuthenticationError:
        print("  OK: AuthenticationError raised")
        return True
    except Exception as exc:
        print(f"  UNEXPECTED: {type(exc).__name__}: {exc}")
        return False


def test_empty_prompt():
    """An empty prompt should raise ValueError locally (no API call)."""
    print("\n--- Empty prompt ---")
    try:
        client = GroqClient()
        client.generate_response("   ")
        print("  FAIL: no exception raised")
        return False
    except ValueError:
        print("  OK: ValueError raised locally")
        return True


def test_basic_response():
    """A simple prompt should return a non-empty string."""
    print("\n--- Basic response ---")
    try:
        client = GroqClient()
        reply = client.generate_response(
            "Say exactly: hello world",
            max_tokens=50,
            temperature=0.0,
        )
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return False

    ok = bool(reply) and len(reply) > 0
    print(f"  Reply: {reply!r}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_with_system_prompt():
    """A system prompt should influence the model behaviour."""
    print("\n--- System prompt ---")
    try:
        client = GroqClient()
        reply = client.generate_response(
            prompt="What is 2+2?",
            system_prompt="You only answer with a single number, nothing else.",
            max_tokens=20,
            temperature=0.0,
        )
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return False

    ok = "4" in reply
    print(f"  Reply: {reply!r}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_model_property():
    """The model property should return the expected model name."""
    print("\n--- Model property ---")
    client = GroqClient()
    name = client.model
    ok = "llama" in name and "8b" in name
    print(f"  Model: {name!r} — {'OK' if ok else 'FAIL'}")
    return ok


def test_tool_calling_calculate():
    """The model should call the calculate tool for math questions."""
    print("\n--- Tool calling: calculate ---")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform math",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"},
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    def fake_executor(name: str, args: dict) -> str:
        from src.tools import calculate as calc_fn
        return calc_fn(args.get("expression", ""))

    try:
        client = GroqClient()
        reply = client.generate_with_tools(
            prompt="What is 145 * 37?",
            tools=tools,
            tool_executor=fake_executor,
            system_prompt="Answer concisely.",
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return False

    ok = bool(reply) and "5365" in reply
    print(f"  Reply: {reply!r}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    print("=== Nova AI Voice Agent — Groq Integration Test ===")

    results = []

    tests = [
        ("test_missing_api_key", test_missing_api_key),
        ("test_invalid_api_key", test_invalid_api_key),
        ("test_empty_prompt", test_empty_prompt),
        ("test_basic_response", test_basic_response),
        ("test_with_system_prompt", test_with_system_prompt),
        ("test_model_property", test_model_property),
        ("test_tool_calling_calculate", test_tool_calling_calculate),
    ]

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
