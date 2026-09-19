# Decision ledger

Tracks resolution of the open decisions listed in `plan_a.md`, section 6.
The coding agent does not resolve these (plan_a.md section 0, rule 3); this file
is updated only once Teo records a choice, with date and reason.

| Id | Decision | Status | Resolved | Choice | Reason |
|---|---|---|---|---|---|
| A | Scaling for the headline scaled CRPS | open | | | |
| B | Utility form | open | | | |
| C | LLM forecaster model and provider for U1 | resolved | 2026-09-18 | `gemini-3.1-flash-lite` through the company LiteLLM OpenAI-compatible proxy | Matches the direct-LLM baseline/backbone named by Dr-CiK while using the approved company access path. |
| D | Forecaster prompt template | resolved | 2026-09-18 | D1: port the CiK Direct Prompt template (`cik_benchmark/baselines/direct_prompt.py`, `make_prompt`) | Teo manually checked that it matches the paper and approved it for U1. |
| E | Samples per forecast and repeats per cell in U1 | resolved | 2026-09-18 | 25 samples and 2 repeats per cell | Confirms the Plan A default for the U1 instrument check. |
| F | U1 task set | resolved | 2026-09-19 | `task_42`, `task_49`, `task_52`, `task_200`, `task_204` | Focus U1 on daily commercial-volume tasks relevant to the downstream Zurich monthly GWP case. The set covers temporary reporting/promotion effects and persistent business-expansion level shifts; `task_42` is retained because it has already been used for U1 smoke-test calls. |
| G | Rendering of contexts | open | | | |
| H | Placebo construction | open | | | |
| I | U1 cost cap | resolved | 2026-09-18 | USD 5.00 hard cap | Covers the current estimated U1 run while bounding retries and unexpected usage. |
| J | Whether a second LLM forecaster is included in U1 | open | | | |
| K | Sync channel between the two machines | resolved | 2026-09-18 | GitHub remote, **public** (not private K1 as drafted) at https://github.com/teoPazera/dr-cik-evidence-utility | MacBook uses a different GitHub account with no access to a private repo under teoPazera; public avoids a collaborator-invite step and nothing sensitive is committed (data/ and external/ are git-ignored, .env is git-ignored and empty on this machine). |
| L | Network and providers permitted on the MacBook | resolved | 2026-09-18 | GitHub, Hugging Face, and company LiteLLM proxy | GitHub and Hugging Face access were verified during Mac setup; Teo selected the company LiteLLM proxy for paid model calls. |
