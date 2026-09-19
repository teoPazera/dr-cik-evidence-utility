"""U0.6: dry-run cost estimate for U1 (plan_a.md U0.6).

Builds every U1 request (task x condition x repeat) with the draft prompt template and counts
tokens. Nothing is sent. Decisions C (model), D (template), E (samples and repeats) and I (cost cap)
are open, so every figure here rests on stated assumptions:

- tokens are estimated, not tokenised (no model is chosen): a low figure (plain characters / 4, the
  same rule as the U0.2 index) and a high figure that counts every digit as its own token. History and
  timestamps are digit-heavy, so the two differ a lot; a real tokenizer replaces both once Decision C
  is made;
- output tokens are the length of one forecast in the requested `(timestamp, value)` format times the
  number of samples; reasoning tokens, if the chosen model emits them, are not included;
- the prompt-level leakage check reads `future_values` on purpose (this is the checker side).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from utrack.conditions.builders import build_condition
from utrack.conditions.leakage import history_text, leaks_beyond_history, matching_future_pairs
from utrack.data.loader import Dataset
from utrack.forecasters.llm_prompt import (
    SYSTEM_MESSAGE,
    TEMPLATE_VERSION,
    LLMRequest,
    build_request,
    expected_output_text,
)

TOKEN_MODEL_NAMES = ("low", "high")
EXAMPLE_REQUEST = ("task_200", "C1")  # structural business-change example, shown in full in the report


def estimate_tokens(text: str, chars_per_token: float, digits_per_token: float) -> int:
    digits = sum(ch.isdigit() for ch in text)
    return math.ceil((len(text) - digits) / chars_per_token + digits / digits_per_token)


def _request_text(request: LLMRequest) -> str:
    """Everything sent as input: the system message and the user prompt."""
    return request.system_message + "\n" + request.user_prompt


def build_u1_requests(
    dataset: Dataset, task_ids: list[str], condition_ids: list[str], repeats: int, n_samples: int, base_seed: int
) -> list[LLMRequest]:
    requests = []
    for task_id in task_ids:
        forecast_input = dataset.forecast_input(task_id)
        for condition_id in condition_ids:
            condition = build_condition(condition_id, dataset, task_id, base_seed)
            for repeat in range(repeats):
                requests.append(build_request(forecast_input, condition, repeat, n_samples))
    return requests


def request_leaks(dataset: Dataset, requests: list[LLMRequest], min_run: int) -> list[tuple[str, str, int]]:
    """(task, condition, run length) for each request whose text holds a run of future values that
    the history does not explain."""
    found = []
    for r in requests:
        hit = leaks_beyond_history(
            _request_text(r),
            history_text(dataset.forecast_input(r.benchmark_id)),
            dataset.tasks[r.benchmark_id].future_values,
            min_run,
        )
        if hit:
            found.append((r.benchmark_id, r.condition_id, hit.run_length))
    return found


def request_pair_matches(dataset: Dataset, requests: list[LLMRequest]) -> list[tuple[str, str, int, int, int]]:
    """(task, condition, future steps given exactly, of which non-zero, horizon) for each distinct request text that
    carries the true value at some future timestamps, e.g. a clear-sky series in the evidence."""
    found = {}
    for r in requests:
        task = dataset.tasks[r.benchmark_id]
        _, exact, nonzero = matching_future_pairs(_request_text(r), task.future_timestamps, task.future_values)
        if exact:
            found[(r.benchmark_id, r.condition_id)] = (
                r.benchmark_id, r.condition_id, exact, nonzero, len(task.future_values)
            )
    return list(found.values())


def estimate_rows(dataset: Dataset, requests: list[LLMRequest], token_models: dict) -> pd.DataFrame:
    rows = []
    for r in requests:
        output_text = expected_output_text(dataset.forecast_input(r.benchmark_id))
        row = {"benchmark_id": r.benchmark_id, "condition_id": r.condition_id, "repeat": r.repeat, "n_samples": r.n_samples}
        for name in TOKEN_MODEL_NAMES:
            model = token_models[name]
            args = (model["chars_per_token"], model["digits_per_token"])
            row[f"input_tokens_{name}"] = estimate_tokens(_request_text(r), *args)
            row[f"output_tokens_per_sample_{name}"] = estimate_tokens(output_text, *args)
        rows.append(row)
    return pd.DataFrame(rows)


def _totals(df: pd.DataFrame, name: str, per_request: bool) -> tuple[int, int, int]:
    """(requests, input tokens, output tokens) if each request carries all its samples (`per_request`),
    or if every sample is its own request and the input is billed once per sample."""
    n = df["n_samples"]
    output = int((df[f"output_tokens_per_sample_{name}"] * n).sum())
    if per_request:
        return len(df), int(df[f"input_tokens_{name}"].sum()), output
    return int(n.sum()), int((df[f"input_tokens_{name}"] * n).sum()), output


def _millions(tokens: int) -> str:
    return f"{tokens / 1e6:.3f}"


def _cost(tokens_in: int, tokens_out: int, prices: dict) -> str:
    if prices.get("input") is None or prices.get("output") is None:
        return "price not set"
    return f"${(tokens_in * prices['input'] + tokens_out * prices['output']) / 1e6:,.2f}"


def write_cost_estimate(
    df: pd.DataFrame,
    cfg: dict,
    leaks: list[tuple[str, str, int]],
    pair_matches: list[tuple[str, str, int, int, int]],
    example: LLMRequest | None,
    out_path: Path,
) -> None:
    cost_cfg = cfg["cost_estimate"]
    prices = cost_cfg["price_usd_per_million_tokens"]
    n_samples, repeats = cost_cfg["n_samples"], cost_cfg["repeats"]
    floor = cost_cfg["valid_rate_floor"]
    tasks = sorted(df["benchmark_id"].unique(), key=lambda t: int(t.split("_")[1]))
    conditions = sorted(df["condition_id"].unique())
    low, high = (cost_cfg["token_models"][m] for m in TOKEN_MODEL_NAMES)

    lines = [
        "# U0.6 dry-run cost estimate for U1",
        "",
        "Generated by `uv run utrack estimate u1`. **No request was sent.** Every figure is an estimate under "
        "the assumptions in section 1; Decisions C, D, E and I are open.",
        "",
        "## 1. Assumptions",
        "",
        f"- Requests: {len(tasks)} tasks ({', '.join(tasks)}) x {len(conditions)} conditions "
        f"({', '.join(conditions)}) x {repeats} repeats = {len(df)} requests, each asking for {n_samples} samples "
        "(Decision E defaults, still open).",
        f"- Prompt: draft template `{TEMPLATE_VERSION}` (Decision D option D1, a port of CiK's Direct Prompt; not yet "
        f"approved) with system message \"{SYSTEM_MESSAGE}\". The context slot is the only thing that differs "
        "between conditions. Message-framing tokens added by an API are not counted.",
        f"- Tokens are estimated, not tokenised (Decision C picks the model). **Low**: {low['chars_per_token']:g} "
        "characters per token for every character, the same rule as the U0.2 index. **High**: a digit counts as "
        f"{1 / high['digits_per_token']:g} token, all other characters {high['chars_per_token']:g} per token. History and "
        "timestamps are mostly digits, so the high figure is an upper bound for a tokenizer that splits digits singly.",
        "- Output: one sample is the `<forecast>` block in the requested `(timestamp, value)` format, one line per "
        "horizon step, with a typical value. Reasoning tokens, if the model emits any, are not included.",
        "- Billing A: several samples per request (`n > 1`): input billed once per request, output once per sample. "
        "Billing B: one sample per request: input billed once per sample.",
        f"- Retries: figures assume every sample is valid. The U1.2 kill threshold is a valid-sample rate of "
        f"{floor:.0%}; at that rate the extra requests scale cost by up to x{1 / floor:.2f}.",
        "",
        "## 2. Totals by condition",
        "",
        "Tokens in millions, low to high.",
        "",
        "| condition | requests A | requests B | input A | input B | output (A and B) |",
        "|---|---|---|---|---|---|",
    ]
    grand = {"A": [0, 0, 0, 0], "B": [0, 0, 0, 0]}  # low in, low out, high in, high out
    for cond in conditions + ["all"]:
        part = df if cond == "all" else df[df["condition_id"] == cond]
        a_low, a_high = _totals(part, "low", True), _totals(part, "high", True)
        b_low, b_high = _totals(part, "low", False), _totals(part, "high", False)
        lines.append(
            f"| {cond} | {a_low[0]} | {b_low[0]} | {_millions(a_low[1])} to {_millions(a_high[1])} | "
            f"{_millions(b_low[1])} to {_millions(b_high[1])} | {_millions(a_low[2])} to {_millions(a_high[2])} |"
        )
        if cond == "all":
            grand["A"] = [a_low[1], a_low[2], a_high[1], a_high[2]]
            grand["B"] = [b_low[1], b_low[2], b_high[1], b_high[2]]

    lines += [
        "",
        "## 3. Cost",
        "",
        "Cost = input tokens x input price + output tokens x output price, per million tokens. The prices are config "
        "values in `configs/u0.yaml` (`cost_estimate.price_usd_per_million_tokens`); Teo fills them in once Decision C "
        "names a model. With both prices at 1 USD per million, the cost in dollars equals the token counts in section 2 "
        "added together.",
        "",
        f"Prices set: input {prices.get('input')}, output {prices.get('output')} (null = not set).",
        "",
        f"| billing | tokens low (in / out, millions) | tokens high (in / out, millions) | cost low | cost high | cost high at {floor:.0%} valid rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in (("A", "A: several samples per request"), ("B", "B: one sample per request")):
        li, lo, hi, ho = grand[name]
        inflated = _cost(math.ceil(hi / floor), math.ceil(ho / floor), prices)
        lines.append(
            f"| {label} | {_millions(li)} / {_millions(lo)} | {_millions(hi)} / {_millions(ho)} | "
            f"{_cost(li, lo, prices)} | {_cost(hi, ho, prices)} | {inflated} |"
        )

    a_low_in, a_low_out = grand["A"][0], grand["A"][1]
    lines += [
        "",
        "Plan section 8 (U1.0) guessed roughly 0.2 million input and 0.6 million output tokens with several samples per "
        f"request, and about 25 times the input with one sample per request. Measured (low estimate): "
        f"{_millions(a_low_in)} million input and {_millions(a_low_out)} million output under billing A; "
        f"input under billing B is {grand['B'][0] / grand['A'][0]:.1f} times billing A.",
        "",
        "## 4. Input tokens per request, by task and condition",
        "",
        "Low to high, tokens. Repeats of the same task and condition are identical, so one is shown.",
        "",
        "| task | " + " | ".join(conditions) + " |",
        "|---|" + "---|" * len(conditions),
    ]
    first = df[df["repeat"] == 0]
    for task in tasks:
        cells = []
        for cond in conditions:
            row = first[(first["benchmark_id"] == task) & (first["condition_id"] == cond)].iloc[0]
            cells.append(f"{row['input_tokens_low']:,} to {row['input_tokens_high']:,}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    per_step = df.drop_duplicates("benchmark_id")
    lines += [
        "",
        f"Largest request: {int(df['input_tokens_high'].max()):,} tokens at the high estimate "
        f"({int(df['input_tokens_low'].max()):,} low); Decision C requires a context window of at least 32,000.",
        "",
        "Output tokens per sample (low to high), by task: "
        + "; ".join(
            f"{r['benchmark_id']} {r['output_tokens_per_sample_low']:,} to {r['output_tokens_per_sample_high']:,}"
            for _, r in per_step.iterrows()
        )
        + ".",
        "",
        "## 5. Prompt-level leakage check",
        "",
        "The full text of every request (system message and user prompt) was scanned for a run of future values that "
        "the history does not already explain (same detector as `leakage_report.md`): "
        + (
            f"**{len(leaks)} of {len(df)} requests flagged**: {leaks}"
            if leaks
            else f"0 of {len(df)} requests flagged."
        ),
        "",
        "A second check reads every `(timestamp, value)` pair in the request and compares those at a future "
        "timestamp with the true value, which catches values handed over at scattered timestamps that a run "
        "cannot. It is reported, not a failure: such values are part of the benchmark's own evidence (plan_a.md U0.5).",
        "",
    ]
    if pair_matches:
        lines += [
            "| task | condition | future steps given exactly | of which non-zero | horizon |",
            "|---|---|---|---|---|",
        ]
        lines += [f"| {t} | {c} | {n} | {nz} | {h} |" for t, c, n, nz, h in sorted(pair_matches)]
        lines += [
            "",
            "On these requests the model is given part of the true answer verbatim. Zeros are cheap to guess, so "
            "the non-zero count is the informative one: a C1 gain on such a task partly measures copying a given "
            "value, not using the evidence to reason. See `leakage_report.md` for the same pattern across all dev tasks.",
            "",
        ]
    else:
        lines += ["No request carries a true future value at a future timestamp.", ""]
    if example is not None:
        lines += [
            f"## Appendix: request for {example.benchmark_id} / {example.condition_id} (repeat {example.repeat}), template `{example.template_version}`",
            "",
            "This is the exact text that would be sent, for Teo to review under Decision D. The system message is "
            f"\"{example.system_message}\" and the request asks for n = {example.n_samples} samples.",
            "",
            "~~~text",
            example.user_prompt.strip("\n"),
            "~~~",
            "",
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
