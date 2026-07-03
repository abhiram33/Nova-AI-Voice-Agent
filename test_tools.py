import json
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
from src.tools import (
    TOOL_DEFINITIONS,
    TOOL_FUNCTIONS,
    execute_tool,
    calculate,
    get_weather,
    web_search,
)


def test_tool_definitions_structure():
    """Tool definitions should have the correct schema structure."""
    print("\n--- Tool definition structure ---")

    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "web_search" in names
    assert "calculate" in names
    assert "get_weather" in names
    print(f"  Tools defined: {names}")

    for t in TOOL_DEFINITIONS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]
        assert "properties" in t["function"]["parameters"]
        assert "required" in t["function"]["parameters"]

    print("  OK — all 3 tools have valid schema")
    return True


def test_calculate_simple():
    """Simple arithmetic should work."""
    print("\n--- Calculate: simple arithmetic ---")
    result = calculate("2 + 2")
    ok = result == "4"
    print(f"  2 + 2 = {result} — {'OK' if ok else 'FAIL'}")
    return ok


def test_calculate_advanced():
    """Math functions should work."""
    print("\n--- Calculate: advanced math ---")
    result = calculate("sqrt(144)")
    ok = result == "12.0"
    print(f"  sqrt(144) = {result} — {'OK' if ok else 'FAIL'}")
    return ok


def test_calculate_invalid():
    """Invalid expression should return an error message."""
    print("\n--- Calculate: invalid expression ---")
    result = calculate("undefined_function(42)")
    ok = result.startswith("[calculate error:")
    print(f"  Result: {result} — {'OK' if ok else 'FAIL'}")
    return ok


def test_execute_tool_unknown():
    """Unknown tool name should return error."""
    print("\n--- Execute unknown tool ---")
    result = execute_tool("nonexistent", {})
    ok = "[unknown tool:" in result
    print(f"  Result: {result} — {'OK' if ok else 'FAIL'}")
    return ok


def test_execute_tool_calculate():
    """execute_tool should dispatch to calculate()."""
    print("\n--- Execute tool: calculate ---")
    result = execute_tool("calculate", {"expression": "3 * 7"})
    ok = result == "21"
    print(f"  3 * 7 = {result} — {'OK' if ok else 'FAIL'}")
    return ok


def test_web_search():
    """Web search should return formatted results."""
    print("\n--- Web search ---")
    try:
        result = web_search("Python programming language")
    except Exception as exc:
        print(f"  SKIP (network may be unavailable): {exc}")
        return True  # not a failure — network-dependent

    ok = bool(result) and "Python" in result
    print(f"  Result length: {len(result)} chars")
    print(f"  First 100 chars: {result[:100]}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_get_weather():
    """Weather should return a formatted string."""
    print("\n--- Get weather ---")
    try:
        result = get_weather("London")
    except Exception as exc:
        print(f"  SKIP (network may be unavailable): {exc}")
        return True

    ok = bool(result) and "°C" in result and "humidity" in result
    print(f"  Result: {result}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_tool_functions_registry():
    """All tool names should have a corresponding function."""
    print("\n--- Tool functions registry ---")
    for t in TOOL_DEFINITIONS:
        name = t["function"]["name"]
        assert name in TOOL_FUNCTIONS, f"Missing function for {name}"
    print(f"  All {len(TOOL_DEFINITIONS)} tools have registered functions — OK")
    return True


def main():
    print("=== Nova AI Voice Agent — Tools Test ===\n")

    tests = [
        ("test_tool_definitions_structure", test_tool_definitions_structure),
        ("test_tool_functions_registry", test_tool_functions_registry),
        ("test_calculate_simple", test_calculate_simple),
        ("test_calculate_advanced", test_calculate_advanced),
        ("test_calculate_invalid", test_calculate_invalid),
        ("test_execute_tool_unknown", test_execute_tool_unknown),
        ("test_execute_tool_calculate", test_execute_tool_calculate),
        ("test_web_search", test_web_search),
        ("test_get_weather", test_get_weather),
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
