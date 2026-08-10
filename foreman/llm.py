"""
The one place that talks to the model.

Returns usage on every call so the ledger's spend events are real rather than
estimated. Retries transient errors so a long run does not die on a blip.
"""
import time

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from . import config

_client = None

# $ per 1M tokens. Update if pricing moves.
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}


class LLMError(RuntimeError):
    pass


def client() -> OpenAI:
    """Lazily built singleton, so importing this module never needs an API key
    (the test suite imports it constantly and never calls out)."""
    global _client
    if _client is None:
        config.require()
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def cents(model: str, tin: int, tout: int) -> float:
    """Convert token usage to cents. Real numbers, from the API's own usage block —
    the ledger's spend events are measured, not estimated."""
    pin, pout = PRICES.get(model, PRICES["gpt-4o"])
    return (tin / 1e6 * pin + tout / 1e6 * pout) * 100


def call(messages, tools=None, tool_choice=None, model=None, max_retries=4):
    """Returns (message, tokens_in, tokens_out, cents)."""
    model = model or config.STRONG_MODEL
    kwargs = {"model": model, "messages": messages, "temperature": config.TEMPERATURE}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    delay, last = 2.0, None
    for _ in range(max_retries):
        try:
            r = client().chat.completions.create(**kwargs)
            u = r.usage
            tin, tout = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
            return r.choices[0].message, tin, tout, cents(model, tin, tout)
        except (RateLimitError, APIConnectionError) as exc:
            last = exc
            time.sleep(delay); delay *= 2
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            if status and 500 <= status < 600:
                last = exc
                time.sleep(delay); delay *= 2
            else:
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    raise LLMError(f"gave up after {max_retries} attempts: {last}")
