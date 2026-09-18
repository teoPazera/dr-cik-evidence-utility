import ast
import dataclasses
import inspect

import pytest
from test_conditions import _dataset, _task

from utrack.conditions import build_condition
from utrack.conditions import leakage as leakage_mod
from utrack.data.schema import EvidenceSpan, ForecastInput
from utrack.forecasters import llm_prompt as llm_prompt_mod
from utrack.forecasters.llm_prompt import (
    SYSTEM_MESSAGE,
    TEMPLATE_VERSION,
    build_request,
    expected_output_text,
    format_history,
    render_user_prompt,
)
from utrack.reports import cost_estimate as cost_mod
from utrack.reports import leakage as leakage_report_mod


def _forecast_input(**overrides) -> ForecastInput:
    defaults = dict(
        benchmark_id="task_x",
        origin="synthetic",
        frequency="1 hour",
        prediction_length=2,
        seasonal_period=None,
        history_timestamps=["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 02:00:00"],
        history_values=[1.5, 2000000.0, 0.1234567],
        future_timestamps=["2026-01-01 03:00:00", "2026-01-01 04:00:00"],
        entity_name="e",
        entity_type="t",
        profile_id="1",
        profile_name="p",
        profile_details={},
        time_series_variable="v",
        target_description="d",
        reasoning_hops=1,
    )
    defaults.update(overrides)
    return ForecastInput(**defaults)


EXPECTED_PROMPT = """
I have a time series forecasting task for you.

Here is some context about the task. Make sure to factor in any background knowledge,
satisfy any constraints, and respect any scenarios.
<context>
CTX
</context>

Here is a historical time series in (timestamp, value) format:
<history>
(2026-01-01 00:00:00, 1.5)
(2026-01-01 01:00:00, 2000000)
(2026-01-01 02:00:00, 0.123457)
</history>

Now please predict the value at the following timestamps: ['2026-01-01 03:00:00' '2026-01-01 04:00:00'].

Return the forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
Do not include any other information (e.g., comments) in the forecast.

Example:
<history>
(t1, v1)
(t2, v2)
(t3, v3)
</history>
<forecast>
(t4, v4)
(t5, v5)
</forecast>

"""


# --- prompt template ---------------------------------------------------------


def test_prompt_matches_the_draft_template_exactly() -> None:
    """Golden text. If this fails on purpose, bump TEMPLATE_VERSION (plan_a.md Decision D)."""
    assert TEMPLATE_VERSION == "d1-draft-v0"
    assert render_user_prompt(_forecast_input(), "CTX") == EXPECTED_PROMPT


def test_c0_leaves_the_context_slot_empty() -> None:
    prompt = render_user_prompt(_forecast_input(), None)
    assert "<context>\n\n</context>" in prompt


def test_history_uses_six_significant_digits_and_no_scientific_notation() -> None:
    lines = format_history(_forecast_input()).split("\n")
    assert lines == ["(2026-01-01 00:00:00, 1.5)", "(2026-01-01 01:00:00, 2000000)", "(2026-01-01 02:00:00, 0.123457)"]


def test_missing_history_values_are_filled_not_printed_as_nan() -> None:
    fi = _forecast_input(history_values=[None, 5.0, None])
    assert "nan" not in format_history(fi).lower()


def test_request_carries_template_version_samples_and_condition_seed() -> None:
    ds = _dataset()
    condition = build_condition("C2", ds, "task_1", 1)
    request = build_request(ds.forecast_input("task_1"), condition, repeat=1, n_samples=25)
    assert (request.condition_id, request.repeat, request.n_samples) == ("C2", 1, 25)
    assert request.template_version == TEMPLATE_VERSION and request.system_message == SYSTEM_MESSAGE
    assert request.condition_seed == condition.seed
    assert condition.context in request.user_prompt


def test_expected_output_is_one_line_per_horizon_step() -> None:
    lines = expected_output_text(_forecast_input()).split("\n")
    assert lines[0] == "<forecast>" and lines[-1] == "</forecast>"
    assert len(lines) == 2 + 2 and lines[1].startswith("(2026-01-01 03:00:00, ")


# --- token estimates and totals ----------------------------------------------


def test_token_models_low_is_chars_over_four_and_high_counts_digits_singly() -> None:
    assert cost_mod.estimate_tokens("abcdefgh", 4.0, 4.0) == 2
    assert cost_mod.estimate_tokens("12345678", 4.0, 4.0) == 2  # low: digits like any character
    assert cost_mod.estimate_tokens("12345678", 4.0, 1.0) == 8  # high: one token per digit
    assert cost_mod.estimate_tokens("ab12", 4.0, 1.0) == 3  # ceil(2/4 + 2)


def _config(prices=None) -> dict:
    return {
        "cost_estimate": {
            "n_samples": 25,
            "repeats": 2,
            "valid_rate_floor": 0.8,
            "token_models": {
                "low": {"chars_per_token": 4.0, "digits_per_token": 4.0},
                "high": {"chars_per_token": 4.0, "digits_per_token": 1.0},
            },
            "price_usd_per_million_tokens": prices or {"input": None, "output": None},
        }
    }


def _requests_and_frame(repeats: int = 2, n_samples: int = 25):
    ds = _dataset()
    requests = cost_mod.build_u1_requests(ds, ["task_1", "task_2"], ["C0", "C1", "C2", "C3"], repeats, n_samples, 5)
    return ds, requests, cost_mod.estimate_rows(ds, requests, _config()["cost_estimate"]["token_models"])


def test_build_u1_requests_covers_every_cell_and_repeat() -> None:
    _, requests, _ = _requests_and_frame()
    assert len(requests) == 2 * 4 * 2
    assert {(r.benchmark_id, r.condition_id, r.repeat) for r in requests} == {
        (t, c, k) for t in ("task_1", "task_2") for c in ("C0", "C1", "C2", "C3") for k in (0, 1)
    }
    assert all(r.n_samples == 25 for r in requests)


def test_high_estimate_is_never_below_low_and_context_adds_input_tokens() -> None:
    _, _, df = _requests_and_frame()
    assert (df["input_tokens_high"] >= df["input_tokens_low"]).all()
    per = df[df["repeat"] == 0].set_index(["benchmark_id", "condition_id"])["input_tokens_low"]
    assert per[("task_1", "C2")] > per[("task_1", "C1")] > per[("task_1", "C0")]


def test_billing_b_bills_input_once_per_sample_and_output_unchanged() -> None:
    _, _, df = _requests_and_frame()
    a_requests, a_in, a_out = cost_mod._totals(df, "low", per_request=True)
    b_requests, b_in, b_out = cost_mod._totals(df, "low", per_request=False)
    assert (a_requests, b_requests) == (len(df), len(df) * 25)
    assert b_in == a_in * 25
    assert a_out == b_out == int((df["output_tokens_per_sample_low"] * 25).sum())


def test_report_without_prices_says_so_and_with_prices_computes_cost(tmp_path) -> None:
    ds, requests, df = _requests_and_frame()
    out = tmp_path / "estimate.md"
    cost_mod.write_cost_estimate(df, _config(), [], [], requests[0], out)
    text = out.read_text(encoding="utf-8")
    assert "No request was sent" in text and "price not set" in text
    assert "0 of 16 requests flagged" in text and "No request carries a true future value" in text

    prices = {"input": 3.0, "output": 15.0}
    cost_mod.write_cost_estimate(df, _config(prices), [], [], requests[0], out)
    _, a_in, a_out = cost_mod._totals(df, "low", per_request=True)
    expected = f"${(a_in * 3.0 + a_out * 15.0) / 1e6:,.2f}"
    assert expected in out.read_text(encoding="utf-8")


def test_report_lists_the_example_request_and_pair_matches(tmp_path) -> None:
    _, requests, df = _requests_and_frame()
    out = tmp_path / "estimate.md"
    cost_mod.write_cost_estimate(df, _config(), [], [("task_9", "C1", 7, 2, 24)], requests[0], out)
    text = out.read_text(encoding="utf-8")
    assert "| task_9 | C1 | 7 | 2 | 24 |" in text
    assert "Appendix: request for task_1 / C0" in text and "<history>" in text


# --- leak checks on request text ---------------------------------------------


def test_request_leaks_flags_a_planted_run_but_not_a_history_repeat() -> None:
    ds = _dataset()
    clean = cost_mod.build_u1_requests(ds, ["task_1"], ["C0", "C1"], 1, 25, 5)
    assert cost_mod.request_leaks(ds, clean, 4) == []

    task = ds.tasks["task_1"]
    leaked = "Prices will be " + ", ".join(str(v) for v in task.future_values) + "."
    ds.tasks["task_1"] = dataclasses.replace(task, gt_evidence=[EvidenceSpan("E1", leaked)])
    flagged = cost_mod.request_leaks(ds, cost_mod.build_u1_requests(ds, ["task_1"], ["C0", "C1"], 1, 25, 5), 4)
    assert [(t, c) for t, c, _ in flagged] == [("task_1", "C1")]

    # a future that repeats the history is visible anyway, so it is not flagged
    repeated = dataclasses.replace(_task(2), future_values=list(_task(2).history_values[-8:]))
    ds2 = _dataset()
    ds2.tasks["task_2"] = repeated
    assert cost_mod.request_leaks(ds2, cost_mod.build_u1_requests(ds2, ["task_2"], ["C0"], 1, 25, 5), 4) == []


def test_timestamped_values_reads_the_pair_shapes_used_in_evidence() -> None:
    text = "(2022-02-09 12:00:00, 903) then 2022-02-09 15:00:00: 608, and from 2022-02-10 00:00:00 to 2022-02-11 00:00:00 it rose"
    assert leakage_mod.timestamped_values(text) == [("2022-02-09 12:00:00", 903.0), ("2022-02-09 15:00:00", 608.0)]


def test_matching_future_pairs_counts_exact_and_non_zero_values() -> None:
    future_ts = ["2022-02-09 00:00:00", "2022-02-09 12:00:00", "2022-02-09 15:00:00"]
    future = [0.0, 903.0, 610.0]
    text = "(2022-02-09 00:00:00, 0) (2022-02-09 12:00:00, 903) (2022-02-09 15:00:00, 608) (2022-02-01 12:00:00, 903)"
    # 3 pairs sit at future timestamps, 2 carry the true value, 1 of those is non-zero
    assert leakage_mod.matching_future_pairs(text, future_ts, future) == (3, 2, 1)
    assert leakage_mod.matching_future_pairs("no pairs here", future_ts, future) == (0, 0, 0)


def test_scan_reports_scattered_exact_pairs_that_the_run_detector_misses() -> None:
    ds = _dataset()
    task = ds.tasks["task_1"]
    every_third = [(ts, v) for i, (ts, v) in enumerate(zip(task.future_timestamps, task.future_values)) if i % 3 == 0]
    evidence = "Reference: " + " ".join(f"({ts}, {v})" for ts, v in every_third)
    ds.tasks["task_1"] = dataclasses.replace(task, gt_evidence=[EvidenceSpan("E1", evidence)])
    scan = leakage_report_mod.scan_dev_tasks(ds, 1)
    c1 = scan[(scan["benchmark_id"] == "task_1") & (scan["surface"] == "C1")].iloc[0]
    assert c1["run_length"] == 0  # not consecutive, so the run detector is blind to it
    assert c1["exact_pairs"] == len(every_third) and c1["exact_nonzero_pairs"] == len(every_third)
    assert scan[scan["surface"] == "metadata"]["exact_pairs"].sum() == 0


# --- nothing is sent ---------------------------------------------------------

NETWORK_MODULES = {"requests", "httpx", "urllib", "urllib3", "socket", "http", "aiohttp", "openai", "anthropic", "ssl"}


@pytest.mark.parametrize("module", [llm_prompt_mod, cost_mod])
def test_dry_run_modules_import_no_network_library(module) -> None:
    imported = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(NETWORK_MODULES), imported & NETWORK_MODULES


def test_llm_prompt_never_references_labels() -> None:
    assert leakage_mod.label_references_in_source(llm_prompt_mod) == []
