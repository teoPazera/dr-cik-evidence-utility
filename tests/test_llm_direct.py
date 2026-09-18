from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from utrack.data.schema import ForecastInput
from utrack.forecasters.llm_direct import CostCapExceeded, CostLedger, LiteLLMDirectForecaster, parse_cik_forecast
from utrack.forecasters.llm_prompt import SYSTEM_MESSAGE, render_user_prompt


def _forecast_input() -> ForecastInput:
    return ForecastInput(
        benchmark_id="task_test",
        origin="synthetic",
        frequency="1 hour",
        prediction_length=2,
        seasonal_period=None,
        history_timestamps=["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
        history_values=[1.0, 2.0],
        future_timestamps=["2026-01-01 02:00:00", "2026-01-01 03:00:00"],
        entity_name="entity",
        entity_type="type",
        profile_id="1",
        profile_name="profile",
        profile_details={},
        time_series_variable="value",
        target_description="target",
        reasoning_hops=1,
    )


def _response(content: str, *, prompt_tokens: int = 100, completion_tokens: int = 10, cost: float = 0.001):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=prompt_tokens - 3),
        ),
        headers={
            "x-litellm-response-cost": str(cost),
            "x-litellm-response-cost-input": "0.0001",
            "x-litellm-response-cost-cache-read": "0.0002",
            "x-litellm-response-cost-output": "0.0007",
            "x-litellm-call-id": "call-test",
        },
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected extra request")
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


def test_parse_cik_forecast_requires_expected_timestamps_and_finite_values() -> None:
    expected = ["2026-01-01 02:00:00", "2026-01-01 03:00:00"]
    text = "<forecast>\n(2026-01-01 02:00:00, 2.5)\n(2026-01-01 03:00:00, 3.5)\n</forecast>"
    np.testing.assert_allclose(parse_cik_forecast(text, expected), [2.5, 3.5])

    with pytest.raises(ValueError, match="missing expected"):
        parse_cik_forecast("<forecast>\n(2026-01-01 02:00:00, 2.5)\n</forecast>", expected)
    with pytest.raises(ValueError, match="not finite"):
        parse_cik_forecast(
            "<forecast>\n(2026-01-01 02:00:00, nan)\n(2026-01-01 03:00:00, 3)\n</forecast>", expected
        )


def test_direct_forecaster_reproduces_cik_prompt_and_retries_invalid_output() -> None:
    fi = _forecast_input()
    invalid = _response("not a forecast")
    valid_1 = _response("<forecast>\n(2026-01-01 02:00:00, 2.5)\n(2026-01-01 03:00:00, 3.5)\n</forecast>")
    valid_2 = _response("<forecast>\n(2026-01-01 02:00:00, 4.5)\n(2026-01-01 03:00:00, 5.5)\n</forecast>")
    client = FakeClient([invalid, valid_1, valid_2])
    forecaster = LiteLLMDirectForecaster(client=client, n_retries=3, cost_cap_usd=1.0)

    out = forecaster.forecast(fi, context="evidence", n_samples=2, seed=3)

    assert out.samples.shape == (2, 2)
    np.testing.assert_allclose(out.samples, [[2.5, 3.5], [4.5, 5.5]])
    assert out.n_requested == 2
    assert out.n_valid == 2
    assert out.token_counts == {"input": 300, "output": 30, "total": 330, "cached_input": 291}
    assert out.cost_usd == pytest.approx(0.003)
    assert any("rejected" in note for note in out.notes)
    assert len(client.calls) == 3
    assert all(call["model"] == "gemini-3.1-flash-lite" for call in client.calls)
    assert all(call["temperature"] == 1.0 for call in client.calls)
    assert all("n" not in call for call in client.calls)

    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == SYSTEM_MESSAGE + "\n\n" + render_user_prompt(fi, "evidence")
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert messages[1]["role"] == "user"
    assert "<forecast>" in messages[1]["content"]
    raw_outputs = out.sampling_params["raw_outputs"]
    assert len(raw_outputs) == 3
    assert raw_outputs[0]["valid"] is False
    assert raw_outputs[0]["content"] == "not a forecast"
    assert raw_outputs[-1]["valid"] is True
    assert out.sampling_params["candidates_per_request"] == 1
    assert out.sampling_params["provider_prompt_cache"]["enabled"] is True


def test_direct_forecaster_returns_incomplete_cell_when_cap_blocks_next_call() -> None:
    fi = _forecast_input()
    client = FakeClient([
        _response("<forecast>\n(2026-01-01 02:00:00, 2)\n(2026-01-01 03:00:00, 3)\n</forecast>", cost=0.007)
    ])
    # Cap permits the first conservative request estimate but not the second.
    forecaster = LiteLLMDirectForecaster(
        client=client,
        n_retries=3,
        cost_cap_usd=0.01,
        input_price_per_million=0.0,
        output_price_per_million=100.0,
        max_output_tokens=64,
    )
    out = forecaster.forecast(fi, context=None, n_samples=2, seed=1)

    assert out.n_valid == 1
    assert out.cost_usd == pytest.approx(0.007)
    assert len(client.calls) == 1
    assert any("cost cap" in note for note in out.notes)
    assert any("incomplete cell" in note for note in out.notes)


def test_cost_ledger_raises_before_overspend() -> None:
    ledger = CostLedger(cap_usd=0.01, spent_usd=0.009)
    with pytest.raises(CostCapExceeded):
        ledger.ensure_can_spend(0.002)


class FakeRawResponse:
    def __init__(self, parsed, headers):
        self._parsed = parsed
        self.headers = headers

    def parse(self):
        return self._parsed


class FakeRawClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_direct_forecaster_uses_raw_response_headers_for_exact_cost() -> None:
    fi = _forecast_input()
    parsed = _response("<forecast>\n(2026-01-01 02:00:00, 2)\n(2026-01-01 03:00:00, 3)\n</forecast>", cost=999.0)
    raw = FakeRawClient(FakeRawResponse(parsed, {
        "x-litellm-response-cost": "0.000123",
        "x-litellm-response-cost-input": "0.000010",
        "x-litellm-response-cost-cache-read": "0.000100",
        "x-litellm-response-cost-output": "0.000013",
        "x-litellm-call-id": "raw-call-1",
    }))
    out = LiteLLMDirectForecaster(raw_client=raw, n_retries=1, cost_cap_usd=1.0).forecast(
        fi, context=None, n_samples=1, seed=0
    )
    assert out.n_valid == 1
    assert out.cost_usd == pytest.approx(0.000123)
    request = out.sampling_params["request_costs"][0]
    assert request["cost_source"] == "litellm_response_header"
    assert request["cache_read_cost_usd"] == pytest.approx(0.0001)
    assert request["call_id"] == "raw-call-1"
    assert len(raw.calls) == 1
