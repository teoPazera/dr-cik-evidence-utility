# U1.2 smoke-test check — task_42 C0 vs C1

**Run date:** 2026-09-18. This document completes the U1.2 engineering check; it is not a U1 gate result.

## Run scope

- Task: `task_42`; conditions: C0 (no context) and C1 (ground-truth evidence); one repeat.
- Requested samples: 25 per condition. The Gemini/LiteLLM route sends one trajectory per API call, so this execution made 25 calls per condition rather than using `n > 1`.
- Plot: `artifacts/u1/smoke/task42_c0_c1_trajectories.png`.

## Kill-condition evaluation

| condition | valid/requested | valid rate | cost | result |
|---|---:|---:|---:|---|
| C0 | 24/24 | 100.0% | $0.149458 | PASS |
| C1 | 25/25 | 100.0% | $0.160013 | PASS |

**Valid-sample kill condition (≥80%): PASS** — C0 was 24/24 valid and C1 was 25/25 valid in the retained continuation store.

## Actual tokens and cost versus U0.6 dry run

The U0.6 estimate predates the selected Gemini route and assumes `n > 1`; it gives low/high **per-prompt** input ranges for task_42. Since the actual route makes 25 one-sample calls, the comparable total range is the per-prompt range multiplied by 25. Actual provider usage includes message framing and any provider-side tokenisation effects, while U0.6 used character heuristics; this is therefore a calibration comparison, not an exact accounting reconciliation.

| condition | dry-run input range for 25 one-sample calls | actual input | actual / dry-run high | actual output | actual cost |
|---|---:|---:|---:|---:|---:|
| C0 | 48,075–132,275 | 171,720 | 1.30× | 71,019 | $0.149458 |
| C1 | 51,775–137,550 | 184,375 | 1.34× | 75,946 | $0.160013 |

**Cost-per-request kill condition (≤2× dry-run estimate): PASS on the available evidence.** Actual input totals are 1.30× (C0) and 1.34× (C1) of the U0.6 high heuristic bound—below the 2× threshold. The original dry run did not have provider pricing configured, so it cannot supply a literal dollar estimate. Actual spend was $0.309471 total for 49 valid trajectories, or $0.006316 per valid trajectory. This must be used to revise the full-run estimate before U1.3.

## Prompt-label leakage check

| condition | consecutive future-value run | exact future timestamp/value pairs | exact non-zero pairs | result |
|---|---:|---:|---:|---|
| C0 | 0 | 0 | 0 | PASS |
| C1 | 0 | 0 | 0 | PASS |

**Label-leak kill condition: PASS.** Neither exact smoke prompt contained a consecutive run of future values beyond what the history already provides, and neither contained an exact future timestamp/value pair. The structural isolation check remains documented in `artifacts/u0/leakage_report.md`.

## Engineering outcome

All three U1.2 kill conditions pass. C1 also had lower CRPS than C0 in this one-task demonstration, but that is descriptive only and does not evaluate U1 gate criteria.
