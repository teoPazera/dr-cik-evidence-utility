# Evidence-Utility Track: Provisional Outline for Stages U2 to U7

Status: **provisional.** This file is not an instruction file. No stage here is to be implemented from it. Each stage gets its own plan, written after U0 and U1 are complete and after the direction has been discussed with the supervisors.

Read `plan_a.md` first. It defines the setting, the sign convention for utility (positive means the context helped), the experiment architecture and the conditions C0 to C4. Every stage below reuses that architecture: it adds condition types or forecasters and writes cells to the same forecast store.

---

## 1. What U0 and U1 will decide for the stages below

| Result from U0 / U1 | What it changes |
|---|---|
| Size of repeat-to-repeat noise relative to utility(C1) | Samples per forecast and repeats per cell in U2 and U3, and therefore cost |
| Share of tasks at the winsorisation cap under each scaling | Headline scaling; whether some tasks are excluded or reported separately |
| Whether the placebo check needed a length-matched control | Whether every later comparison also needs a length-matched control |
| Whether C2 (raw supporting documents) hurt, as Dr-CiK reports | Whether per-document measurement uses single documents, leave-one-out from the supporting set, or both |
| Valid-sample rate and cost per request | Model choice for the large stages; whether a cheaper model is sufficient |
| Whether stored document order reveals role | Shuffling requirements for every retrieval-based stage |

---

## 2. Stages

### U2. Noise floor and per-document pilot
- **Question:** is the spread of utility across documents larger than the spread across repeated runs of the same cell?
- **New condition type:** single-document context (one document of the task's corpus as the only context).
- **Scope (indicative):** about 10 dev tasks, every document, a reduced sample count, three repeats.
- **Provisional gate:** between-document spread exceeds repeat-to-repeat spread by a margin fixed before the run.
- **If it fails:** raise samples or repeats once; if still failing, per-document utility is not measurable with this forecaster and the track moves to context-level utility only.

### U3. Per-document utility table
- **Question (RQ1):** how does forecast utility relate to document role, distractor subtype and task properties?
- **Scope (indicative):** 30 to 60 dev tasks, 25 samples.
- **Outputs:** utility by role and subtype; share of supporting documents with negative utility; share of distractors with positive utility; which evidence spans a document contains against its utility.
- This result is usable on its own even if later stages fail.

### U4. DR agent runs with logged signals
- **Question:** what is the utility of the context that real agents produce, and how do Dr-CiK's DR-quality metrics relate to it?
- **New components:** one simple retrieval agent and one agentic DR agent; definitions of evidence recall, supporting-document recall and distractor avoidance. The official definitions are not public, so the chosen definitions are documented and their sensitivity reported.
- **Logged per run:** cited document ids, extracted evidence, stated confidence, agreement of evidence across repeated runs, shift between the C0 forecast and the with-context forecast, entity and date consistency checks between documents and the series.
- **Open:** which agents; how many repeats per task; whether a naive "retrieve top-k and concatenate" baseline is included (a supervisor has asked for a naive-retrieval baseline).

### U5. Predicting utility before the outcome is known
- **Question (RQ2):** which signals available at forecast time predict whether utility is positive?
- **Method:** fit simple models on some tasks, evaluate on held-out tasks. Report AUROC for the sign of utility and calibration of predicted probabilities.
- **No new forecasting cost.** Uses cells from U3 and U4.
- **Open:** whether an LLM-based sufficiency rater is one of the signals (adds cost).

### U6. Gating context on predicted utility
- **Question (RQ3):** does passing context only when predicted utility exceeds a threshold improve the mean score?
- **Comparisons:** always use context, never use context, gated, and an oracle gate that knows the true sign of utility (upper bound).
- **Outputs:** risk-coverage curve (share of tasks where context is used against mean scaled CRPS); a do-no-harm check on tasks where the oracle says context should not be used.
- **Second forecaster family** enters here or in U3, to test whether the utility ordering of documents depends on the forecaster.

### U7. Hidden-test submission and write-up
- Run the final pipeline on the 80 hidden tasks with at least 100 samples per task, produce `forecasts.jsonl` and `deep_research.jsonl` in the format of `SUBMISSION.md`, and submit for verified scoring.
- The 100-sample requirement applies here only.

---

## 3. Indicative costs

Assumptions: token sizes are rough prior guesses, not measurements (nothing has been measured yet): about 520 tokens per document, about 900 tokens per serialised history, about 38 documents per task, mean horizon about 70 steps; 25 samples per forecast; several samples per request; a low-cost API model at roughly 0.30 / 1.20 USD per million input / output tokens. A frontier-tier model is roughly ten times higher. Replace with the U0.6 dry-run figures and real prices.

| Stage | Indicative cost |
|---|---|
| U2 | about 5 USD |
| U3 (60 tasks) | about 70 USD |
| U4 | 5 to 20 USD, depends strongly on the agent design |
| U5 | none, or small if an LLM rater is added |
| U6 | under 20 USD |
| U7 | about 10 USD |

Prompt caching of the shared history and batch endpoints would lower these further where the provider supports them.

---

## 4. Possible external-validity chapters (not staged yet)

- **Public real-document case study:** quarterly commercial insurance rate-change series as targets, with a dated corpus of broker and reinsurer reports, where the agent may only read documents dated on or before the forecast origin. Few turning points, so descriptive only.
- **Detection framing for the company:** the agent flags events without touching the forecast; flags are compared afterwards with the frozen model's normalised forecast errors. Requires only errors per origin and horizon, not the underlying figures.
- **Closed-book control:** the forecaster or agent runs without documents, to separate retrieved information from memorised information. Relevant for real-document studies; Dr-CiK entities are fictional.