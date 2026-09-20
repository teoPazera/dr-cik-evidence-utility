"""CiK-compatible direct LLM forecaster for U1.

This module preserves the behavioral core of CiK's ``DirectPrompt`` baseline:
its prompt template, ``temperature=1.0`` default, tagged forecast parsing,
timestamp validation, and rejection sampling. The transport is adapted only for
the configured company LiteLLM proxy and Gemini 3.1 Flash-Lite constraints:
the route returns one candidate per request, so samples are collected through
sequential calls.

Provider-side Gemini prompt caching is enabled with LiteLLM's documented
``cache_control: {"type": "ephemeral"}`` marker. LiteLLM response caching must
not be enabled because independent forecast samples are required.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from utrack.data.schema import ForecastInput
from utrack.forecasters.base import ForecasterOutput
from utrack.forecasters.llm_prompt import SYSTEM_MESSAGE, TEMPLATE_VERSION, render_user_prompt

_FORECAST_RE = re.compile(r"<forecast>\s*(.*?)\s*</forecast>", re.IGNORECASE | re.DOTALL)


class CostCapExceeded(RuntimeError):
    """Raised before an API request when it could exceed the configured spend cap."""


class ChatCompletionClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class RawChatCompletionClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass
class CostLedger:
    """Running U1 cost ledger, preferring proxy-reported cost when available."""

    cap_usd: float
    spent_usd: float = 0.0
    request_costs: list[float] = field(default_factory=list)

    def ensure_can_spend(self, estimated_cost_usd: float) -> None:
        if estimated_cost_usd < 0:
            raise ValueError("estimated cost cannot be negative")
        if self.spent_usd + estimated_cost_usd > self.cap_usd + 1e-12:
            raise CostCapExceeded(
                f"cost cap ${self.cap_usd:.6f} would be exceeded: "
                f"spent=${self.spent_usd:.6f}, estimated next=${estimated_cost_usd:.6f}"
            )

    def record(self, cost_usd: float) -> None:
        if cost_usd < 0:
            raise ValueError("recorded cost cannot be negative")
        self.spent_usd += cost_usd
        self.request_costs.append(cost_usd)


def _object_or_mapping(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _header_value(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return headers.get(name)
    except AttributeError:
        return None


def parse_cik_forecast(content: str, expected_timestamps: list[str]) -> np.ndarray:
    """Parse CiK's ``<forecast>`` format and require every expected timestamp.

    The parsing intentionally mirrors CiK's DirectPrompt behavior: remove
    parentheses, split one pair per line, construct a timestamp-to-value map,
    and retrieve values in the required timestamp order. Additional timestamps
    are harmless; malformed lines, duplicate keys or missing expected values are
    rejected.
    """
    match = _FORECAST_RE.search(content or "")
    if not match:
        raise ValueError("response has no <forecast>...</forecast> block")

    values_by_timestamp: dict[str, float] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not (line.startswith("(") and line.endswith(")")):
            raise ValueError(f"invalid forecast line: {raw_line!r}")
        inside = line[1:-1]
        if "," not in inside:
            raise ValueError(f"forecast line lacks comma: {raw_line!r}")
        timestamp, value_text = inside.split(",", 1)
        timestamp = timestamp.strip().strip("'\"")
        if not timestamp:
            raise ValueError("forecast timestamp is empty")
        if timestamp in values_by_timestamp:
            raise ValueError(f"duplicate forecast timestamp: {timestamp}")
        try:
            value = float(value_text.strip())
        except ValueError as exc:
            raise ValueError(f"forecast value is not numeric for {timestamp}: {value_text!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"forecast value is not finite for {timestamp}: {value_text!r}")
        values_by_timestamp[timestamp] = value

    if not values_by_timestamp:
        raise ValueError("forecast contains no rows")
    missing = [timestamp for timestamp in expected_timestamps if timestamp not in values_by_timestamp]
    if missing:
        raise ValueError(f"forecast missing expected timestamps: {missing[:3]}")
    return np.asarray([values_by_timestamp[timestamp] for timestamp in expected_timestamps], dtype=float)


class LiteLLMDirectForecaster:
    """CiK DirectPrompt adapter over the company LiteLLM proxy."""

    name = "llm_direct"
    version = "1.0.0-cik-litellm"

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 1.0,
        n_retries: int = 3,
        cost_cap_usd: float = 5.0,
        input_price_per_million: float = 0.25,
        cached_input_price_per_million: float = 0.025,
        output_price_per_million: float = 1.50,
        max_output_tokens: int = 8000,
        client: ChatCompletionClient | None = None,
        raw_client: RawChatCompletionClient | None = None,
        ledger: CostLedger | None = None,
    ) -> None:
        if n_retries < 1:
            raise ValueError("n_retries must be at least 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        self.model = model
        self.temperature = temperature
        self.n_retries = n_retries
        if input_price_per_million < 0 or cached_input_price_per_million < 0 or output_price_per_million < 0:
            raise ValueError("token prices cannot be negative")
        self.input_price_per_million = input_price_per_million
        self.cached_input_price_per_million = cached_input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.max_output_tokens = max_output_tokens
        self.ledger = ledger or CostLedger(cap_usd=cost_cap_usd)
        self.client = client
        self.raw_client = raw_client
        if self.client is None and self.raw_client is None:
            self.client, self.raw_client = self._clients_from_env()

    @staticmethod
    def _clients_from_env() -> tuple[ChatCompletionClient, RawChatCompletionClient]:
        load_dotenv(dotenv_path=".env")
        proxy_url = os.getenv("LITELLM_PROXY_URL")
        api_key = os.getenv("LITELLM_API_KEY")
        if not proxy_url or not api_key:
            raise RuntimeError("LITELLM_PROXY_URL and LITELLM_API_KEY must be set in .env")
        completions = OpenAI(base_url=proxy_url, api_key=api_key).chat.completions
        return completions, completions.with_raw_response

    @staticmethod
    def _messages(user_prompt: str) -> list[dict[str, Any]]:
        # Gemini explicit CachedContent cannot be combined with a separate
        # system_instruction. Keep the CiK system text as the first line of the
        # cached content, then preserve the DirectPrompt text byte-for-byte.
        # The final user turn repeats the original output contract so every
        # request remains an independent completion rather than a response-cache hit.
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_MESSAGE + "\n\n" + user_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": (
                    "Now produce one independent forecast. Return only the forecast in "
                    "(timestamp, value) format between <forecast> and </forecast> tags; "
                    "do not include comments or other text."
                ),
            },
        ]

    def _estimated_request_cost(self, prompt_tokens: int, max_output_tokens: int) -> float:
        """Conservative pre-request cost.

        Before a response arrives we cannot know whether Gemini will return a cache
        hit, so budget all input at the higher uncached rate. Actual fallback
        accounting below uses the reported cached-token count.
        """
        return (
            prompt_tokens * self.input_price_per_million / 1_000_000
            + max_output_tokens * self.output_price_per_million / 1_000_000
        )

    def _extract_usage_and_cost(self, response: Any) -> tuple[dict[str, int], float, dict[str, Any]]:
        usage = _object_or_mapping(response, "usage", {}) or {}
        prompt_tokens = int(_object_or_mapping(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(_object_or_mapping(usage, "completion_tokens", 0) or 0)
        total_tokens = int(_object_or_mapping(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        details = _object_or_mapping(usage, "prompt_tokens_details", {}) or {}
        cached_tokens = int(_object_or_mapping(details, "cached_tokens", 0) or 0)

        response_cost_text = _header_value(response, "x-litellm-response-cost")
        uncached_tokens = max(prompt_tokens - cached_tokens, 0)
        if response_cost_text is not None:
            cost = float(response_cost_text)
            source = "litellm_response_header"
            input_cost = _float_header(response, "x-litellm-response-cost-input")
            cache_read_cost = _float_header(response, "x-litellm-response-cost-cache-read")
            output_cost = _float_header(response, "x-litellm-response-cost-output")
        else:
            input_cost = uncached_tokens * self.input_price_per_million / 1_000_000
            cache_read_cost = cached_tokens * self.cached_input_price_per_million / 1_000_000
            output_cost = completion_tokens * self.output_price_per_million / 1_000_000
            cost = input_cost + cache_read_cost + output_cost
            source = "configured_token_prices_cache_aware"

        components = {
            "input_cost_usd": input_cost,
            "cache_read_cost_usd": cache_read_cost,
            "output_cost_usd": output_cost,
            "call_id": _header_value(response, "x-litellm-call-id"),
            "cost_source": source,
        }
        return (
            {
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": total_tokens,
                "cached_input": cached_tokens,
            },
            cost,
            components,
        )

    def forecast(
        self,
        forecast_input: ForecastInput,
        context: str | None,
        n_samples: int,
        seed: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ForecasterOutput:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        user_prompt = render_user_prompt(forecast_input, context)
        messages = self._messages(user_prompt)
        expected = list(forecast_input.future_timestamps)
        # Configured at 8,000 for U1 because a 100-step timestamp/value trajectory
        # exceeded the earlier 2,400-token bound and was truncated before </forecast>.
        max_output_tokens = self.max_output_tokens
        estimated_cost = self._estimated_request_cost(len(user_prompt) // 4 + 32, max_output_tokens)

        valid: list[np.ndarray] = []
        notes: list[str] = []
        token_counts = {"input": 0, "output": 0, "total": 0, "cached_input": 0}
        request_costs: list[dict[str, Any]] = []
        raw_outputs: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = n_samples + self.n_retries
        started = time.monotonic()

        def report_progress(event: str, **details: Any) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "event": event,
                    "benchmark_id": forecast_input.benchmark_id,
                    "requested_samples": n_samples,
                    "valid_samples": len(valid),
                    "attempts": attempts,
                    "max_attempts": max_attempts,
                    "spent_usd": self.ledger.spent_usd,
                    "cost_cap_usd": self.ledger.cap_usd,
                    "elapsed_seconds": time.monotonic() - started,
                    **details,
                }
            )

        report_progress("cell_started")
        while len(valid) < n_samples and attempts < max_attempts:
            try:
                self.ledger.ensure_can_spend(estimated_cost)
            except CostCapExceeded as exc:
                notes.append(str(exc))
                report_progress("cost_cap_blocked", message=str(exc))
                break

            attempts += 1
            report_progress("attempt_started")
            try:
                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": max_output_tokens,
                }
                if self.raw_client is not None:
                    raw_response = self.raw_client.create(**request_kwargs)
                    response = raw_response.parse()
                    # Attach headers for the shared accounting function. The parsed
                    # OpenAI completion object does not retain proxy HTTP headers.
                    setattr(response, "headers", raw_response.headers)
                elif self.client is not None:
                    response = self.client.create(**request_kwargs)
                else:  # defensive: constructor guarantees one client exists
                    raise RuntimeError("no LiteLLM client configured")
            except Exception as exc:  # transport failures consume unknown provider cost; record and retry.
                message = f"request {attempts} failed: {type(exc).__name__}: {exc}"
                notes.append(message)
                report_progress("attempt_failed", message=message)
                continue

            usage, request_cost, components = self._extract_usage_and_cost(response)
            self.ledger.record(request_cost)
            for key in token_counts:
                token_counts[key] += usage[key]

            choices = _object_or_mapping(response, "choices", []) or []
            if not choices:
                notes.append(f"request {attempts} returned no choices")
                request_costs.append({"attempt": attempts, "cost_usd": request_cost, **components, **usage})
                raw_outputs.append({"attempt": attempts, "content": None, "valid": False, "rejection_reason": "no choices"})
                report_progress("attempt_rejected", message="request returned no choices", request_cost_usd=request_cost)
                continue

            content = _object_or_mapping(_object_or_mapping(choices[0], "message", {}), "content", None)
            request_costs.append({"attempt": attempts, "cost_usd": request_cost, **components, **usage})
            raw_record = {"attempt": attempts, "content": content, "valid": False, "rejection_reason": None}
            try:
                valid.append(parse_cik_forecast(str(content or ""), expected))
                raw_record["valid"] = True
                report_progress("attempt_accepted", request_cost_usd=request_cost)
            except ValueError as exc:
                raw_record["rejection_reason"] = str(exc)
                message = f"request {attempts} rejected: {exc}"
                notes.append(message)
                report_progress("attempt_rejected", message=message, request_cost_usd=request_cost)
            raw_outputs.append(raw_record)

        samples = np.asarray(valid, dtype=float)
        if samples.size == 0:
            samples = np.empty((0, forecast_input.prediction_length), dtype=float)

        sampling_params = {
            "temperature": self.temperature,
            "template_version": TEMPLATE_VERSION,
            "provider": "litellm",
            "api_style": "openai-compatible",
            "candidates_per_request": 1,
            "provider_prompt_cache": {"enabled": True, "type": "ephemeral"},
            "response_cache": {"enabled": False},
            "attempts": attempts,
            "max_attempts": max_attempts,
            "request_costs": request_costs,
            "raw_outputs": raw_outputs,
            "elapsed_seconds": time.monotonic() - started,
        }
        if len(valid) < n_samples:
            notes.append(f"incomplete cell: requested {n_samples}, valid {len(valid)}")
        report_progress("cell_finished", complete=len(valid) == n_samples)

        return ForecasterOutput(
            samples=samples,
            forecaster_name=self.name,
            forecaster_version=self.version,
            model_identifier=self.model,
            n_requested=n_samples,
            n_valid=len(valid),
            seed=seed,
            sampling_params=sampling_params,
            token_counts=token_counts,
            cost_usd=sum(item["cost_usd"] for item in request_costs),
            notes=notes,
        )


def _float_header(response: Any, name: str) -> float | None:
    value = _header_value(response, name)
    return float(value) if value is not None else None
