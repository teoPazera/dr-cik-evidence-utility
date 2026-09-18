# task_42 cached Gemini smoke: C0 vs C1

This is a 25-sample-per-condition, one-task demonstration; it is not a U1 gate result.

## LLM results

| condition | samples | raw CRPS | scaled A1 | scaled A2 | scaled A3 | cost | input tokens | cached input | cache fraction | output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 25 | 73.173834 | 0.251066 | 0.964356 | 0.147474 | $0.155701 | 178,875 | 177,975 | 99.50% | 73,988 |
| C1 | 25 | 58.747108 | 0.201566 | 0.774226 | 0.118398 | $0.160013 | 184,375 | 183,475 | 99.51% | 75,946 |

## Context difference (positive means C1 improves over C0)

- A1 utility = scaled CRPS(C0) − scaled CRPS(C1) = **0.049499**
- A2 utility = scaled CRPS(C0) − scaled CRPS(C1) = **0.190129**
- A3 utility = scaled CRPS(C0) − scaled CRPS(C1) = **0.029075**

## Statistical C0 baselines on task_42

| forecaster | raw CRPS | scaled A1 | scaled A2 | scaled A3 |
|---|---:|---:|---:|---:|
| last_value_naive | 201.858371 | 0.692593 | 2.660285 | 0.406823 |
| seasonal_naive | 175.616909 | 0.602557 | 2.314450 | 0.353936 |

## Cache accounting

Cached input is reported by the LiteLLM/Gemini response usage. It does not affect CRPS. Costs above are estimated using the configured $0.25/M input and $1.50/M output rates because the OpenAI SDK path did not expose the LiteLLM HTTP cost headers for these calls. Do not subtract cached tokens again from `cost_usd`; any provider cache discount must be obtained from the proxy cost headers in a raw-response transport implementation.
