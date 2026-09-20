# Decision ledger

Tracks resolution of the open decisions listed in `plan_a.md`, section 6.
The coding agent does not resolve these (plan_a.md section 0, rule 3); this file
is updated only once Teo records a choice, with date and reason.

| Id | Decision | Status | Resolved | Choice | Reason |
|---|---|---|---|---|---|
| A | Scaling for the headline scaled CRPS | resolved | 2026-09-19 | A3: divide by the mean absolute history value | Teo selected an outcome-independent, history-only headline scaling for U1 gates; A1 and A2 remain stored as sensitivity analyses. |
| B | Utility form | resolved | 2026-09-19 | B1: `score_without_context − score_with_context`, the absolute difference of winsorised scaled CRPS | Positive utility means context helped. B2 (`1 − score_ctx / score_base`) remains a supplementary relative diagnostic, not the headline. |
| C | LLM forecaster model and provider for U1 | resolved | 2026-09-18 | `gemini-3.1-flash-lite` through the company LiteLLM OpenAI-compatible proxy | Matches the direct-LLM baseline/backbone named by Dr-CiK while using the approved company access path. |
| D | Forecaster prompt template | resolved | 2026-09-18 | D1: port the CiK Direct Prompt template (`cik_benchmark/baselines/direct_prompt.py`, `make_prompt`) | Teo manually checked that it matches the paper and approved it for U1. |
| E | Samples per forecast and repeats per cell in U1 | resolved | 2026-09-18 | 25 samples and 2 repeats per cell | Confirms the Plan A default for the U1 instrument check. |
| F | U1 task set | resolved | 2026-09-19 | `task_42`, `task_49`, `task_52`, `task_200`, `task_204` | Focus U1 on daily commercial-volume tasks relevant to the downstream Zurich monthly GWP case. The set covers temporary reporting/promotion effects and persistent business-expansion level shifts; `task_42` is retained because it has already been used for U1 smoke-test calls. |
| G | Rendering of contexts | resolved | 2026-09-19 | Evidence spans in natural id order, one per line; concatenated documents canonicalised by document id then seeded-shuffled, with neutral `[Document n]` delimiters | Prevents the released stored document order from revealing document role or subtype while keeping rendering deterministic and auditable. |
| H | Placebo construction | resolved | 2026-09-19 | H1 plus H2: seeded unrelated-task `gt_evidence` (different entity and variable; never self-source), with a deterministic length-matched variant | Controls both semantic irrelevance and the possible generic effect of adding a longer context block, before interpreting document-level utility. |
| I | U1 cost cap | resolved | 2026-09-19 | USD 15.00 emergency hard cap, with live warnings at USD 5.00, 10.00 and 15.00 | Teo approved a higher ceiling for staged U1 cells while requiring persistent per-call and per-cell cost tracking. |
| J | Whether a second LLM forecaster is included in U1 | resolved | 2026-09-19 | No. One LLM forecaster in U1. | Keep U1 focused on validating the context-utility instrument; assess a second forecaster family later. |
| K | Sync channel between the two machines | resolved | 2026-09-18 | GitHub remote, **public** (not private K1 as drafted) at https://github.com/teoPazera/dr-cik-evidence-utility | MacBook uses a different GitHub account with no access to a private repo under teoPazera; public avoids a collaborator-invite step and nothing sensitive is committed (data/ and external/ are git-ignored, .env is git-ignored and empty on this machine). |
| L | Network and providers permitted on the MacBook | resolved | 2026-09-18 | GitHub, Hugging Face, and company LiteLLM proxy | GitHub and Hugging Face access were verified during Mac setup; Teo selected the company LiteLLM proxy for paid model calls. |
