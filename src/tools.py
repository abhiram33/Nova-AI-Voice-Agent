import json
import logging
import math
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tool definitions (OpenAI-compatible function-calling schema)
# ------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation (arithmetic, trigonometry, logarithms, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2', 'sqrt(144)', 'sin(pi/4)'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name and optional country/region, e.g. 'Tokyo' or 'London, UK'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------


def web_search(query: str) -> str:
    """
    Search the web via DuckDuckGo and return a summary of results.
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return f"[web_search error: {exc}]"

    if not results:
        return "[web_search: no results found]"

    snippets = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")[:200]
        href = r.get("href", "")
        snippets.append(f"{i}. {title}\n   {body}\n   Source: {href}")

    return "\n\n".join(snippets)


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Uses the ``math`` module for functions (sin, cos, sqrt, log, etc.)
    and standard arithmetic operators.
    """
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(
        {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    )
    safe_globals.update(
        {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "int": int,
            "float": float,
        }
    )

    try:
        result = eval(expression, safe_globals)  # noqa: S307 — safe_globals has no __builtins__
    except Exception as exc:
        logger.warning("Calculation failed: %s", exc)
        return f"[calculate error: {exc}]"

    return str(result)


def get_weather(location: str) -> str:
    """
    Fetch current weather from wttr.in (free, no API key required).
    """
    try:
        resp = httpx.get(
            f"https://wttr.in/{location}?format=j1",
            timeout=10.0,
            headers={"User-Agent": "curl/8.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Weather fetch failed: %s", exc)
        return f"[get_weather error: {exc}]"

    try:
        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
    except (KeyError, IndexError) as exc:
        logger.warning("Weather parse failed: %s", exc)
        return f"[get_weather parse error: {exc}]"

    return (
        f"Weather in {city}, {country}: {desc}, "
        f"{temp_c}°C (feels like {feels_like}°C), "
        f"humidity {humidity}%, wind {wind} km/h."
    )

# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "web_search": web_search,
    "calculate": calculate,
    "get_weather": get_weather,
}


def execute_tool(name: str, args: dict) -> str:
    """Execute a named tool with the given arguments and return the result string."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"[unknown tool: {name}]"
    try:
        return fn(**args)
    except Exception as exc:
        logger.error("Tool %s failed: %s", name, exc, exc_info=True)
        return f"[tool error: {exc}]"
