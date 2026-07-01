import json
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq
from groq import APIError, AuthenticationError, RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "llama-3.1-8b-instant"


class GroqClient:
    """
    Wrapper around the Groq API for LLM inference.

    Usage
    -----
    >>> client = GroqClient()
    >>> reply = client.generate_response("Hello!")
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        api_key : str or None
            Groq API key.  Falls back to the ``GROQ_API_KEY`` environment
            variable when ``None``.
        """
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Provide it via the constructor, set the environment variable, "
                "or add it to a .env file."
            )
        self._client = Groq(api_key=key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Send a prompt (with optional conversation history) to the Groq LLM.

        Parameters
        ----------
        prompt : str
            The user message to send.
        system_prompt : str or None
            Optional system-level instruction.  When ``None`` a neutral
            assistant persona is used.
        history : list[dict] or None
            Previous ``{"role": ..., "content": ...}`` messages to include
            for context.  Inserted between the system message and the
            current user message.
        temperature : float
            Sampling temperature (0.0 – 2.0).  Default ``0.7``.
        max_tokens : int
            Maximum number of tokens in the response.  Default ``512``.

        Returns
        -------
        str
            The model's response text.  May be empty if the model returns
            no content.

        Raises
        ------
        ValueError
            If ``prompt`` is empty or only whitespace.
        AuthenticationError
            If the API key is invalid or revoked.
        RateLimitError
            If the account has exceeded its rate / quota limit.
        APIError
            For other API-level failures (server errors, etc.).
        """
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("Prompt must be non-empty.")

        messages = self._build_messages(prompt=cleaned, system_prompt=system_prompt, history=history)

        logger.debug("Sending %d messages to %s", len(messages), MODEL)

        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except AuthenticationError:
            logger.error("Groq authentication failed — check your API key.")
            raise
        except RateLimitError:
            logger.error("Groq rate limit exceeded.")
            raise
        except APIError as exc:
            logger.error("Groq API error: %s", exc)
            raise

        content = response.choices[0].message.content
        return (content or "").strip()

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_executor: Any,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a prompt with tool-calling support.

        If the model decides to call a tool, the *tool_executor* is
        invoked, the result is fed back, and the model produces the
        final answer — all in a single call (multiple API round-trips
        if needed).

        Parameters
        ----------
        prompt : str
            The user message.
        tools : list[dict]
            OpenAI-compatible tool definitions.
        tool_executor : callable ``(name, args) -> str``
            Function that executes a tool and returns a result string.
        system_prompt : str or None
            System-level instruction.
        history : list[dict] or None
            Previous conversation messages.
        temperature : float
            Sampling temperature (default ``0.3``).
        max_tokens : int
            Max tokens per response (default ``1024``).

        Returns
        -------
        str
            The model's final response after any tool calls.
        """
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("Prompt must be non-empty.")

        messages = self._build_messages(prompt=cleaned, system_prompt=system_prompt, history=history)

        for turn in range(3):  # safety limit
            try:
                response = self._client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except AuthenticationError:
                logger.error("Groq auth failed during tool call.")
                raise
            except RateLimitError:
                logger.error("Groq rate limit exceeded during tool call.")
                raise
            except APIError as exc:
                logger.error("Groq API error during tool call: %s", exc)
                raise

            choice = response.choices[0]
            msg = choice.message

            # No tool calls → final answer.
            if not msg.tool_calls:
                return (msg.content or "").strip()

            # Process each tool call the model requested.
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info("Tool call: %s(%s)", name, args)
                result = tool_executor(name, args)
                logger.info("Tool result (%s): %.200s", name, result)

                # The assistant message with tool_calls must be added
                # before tool response messages.
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {"name": name, "arguments": tc.function.arguments},
                            "type": "function",
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        logger.warning("Tool call loop limit reached — forcing final answer without tools.")
        messages.append({
            "role": "user",
            "content": "Please provide a clear, concise final answer based only on the information above. "
                       "Do NOT include any function calls or XML tags in your response.",
        })
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("Final fallback request failed: %s", exc)
            return (msg.content or "").strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Build the message list for a chat completion request."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """The model identifier used by this client."""
        return MODEL
