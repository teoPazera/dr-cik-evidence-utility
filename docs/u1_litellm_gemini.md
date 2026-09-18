# U1 LiteLLM / Gemini implementation notes

Status: configuration and verified API behavior as of 2026-09-18. This is an implementation note, not permission to start the paid U1 run.

## Endpoint and credentials

Use the company LiteLLM proxy through its OpenAI-compatible `/chat/completions` endpoint.
Credentials stay only in `.env`:

```dotenv
LITELLM_PROXY_URL=
LITELLM_API_KEY=
```

Selected model: `gemini-3.1-flash-lite`.

## Verified behavior

Verified against the configured company proxy on 2026-09-18:

- A normal completion succeeds.
- `temperature=1.0` is accepted.
- `n=2` is rejected with `Multiple candidates is not enabled for this model`.
- Therefore U1 must request one trajectory per API request: nominally 1,000 requests for 1,000 trajectories, plus retries.
- Gemini provider-side prompt caching works when a continuous content block is marked with `cache_control: {"type": "ephemeral"}`.
- In the cache test, 2,877 of 2,895 prompt tokens were reported as cached.

Do not enable LiteLLM completion/response caching or semantic caching. Those may replay a stored forecast and would invalidate independent probabilistic samples. Only provider-side prompt/context caching is allowed.

## Request shape

The approved forecast template is the CiK Direct Prompt template from
`external/context-is-key-forecasting/cik_benchmark/baselines/direct_prompt.py::make_prompt`.
Keep the reusable prompt text byte-identical across the 25 calls for a cell and mark one continuous block for ephemeral provider caching. Gemini explicit caching cannot be combined with a separate OpenAI `system` message, so the CiK system text is prefixed to the cached user block; this is the only deliberate role-level deviation from CiK.

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")
client = OpenAI(
    base_url=os.environ["LITELLM_PROXY_URL"],
    api_key=os.environ["LITELLM_API_KEY"],
)

response = client.chat.completions.create(
    model="gemini-3.1-flash-lite",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a useful forecasting assistant.\n\n" + static_prompt_block,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": generation_instruction},
    ],
    temperature=1.0,
    max_tokens=max_output_tokens,
)
```

If the OpenAI SDK rejects or strips provider-specific message fields, send the same JSON directly to `${LITELLM_PROXY_URL}/chat/completions`. The raw HTTP form was verified to work.

## Sampling and cache use

For each task × condition × repeat cell:

1. Render and validate the prompt once.
2. Keep its cached block exactly unchanged.
3. Send 25 sequential one-choice requests at temperature 1.0.
4. Do not put a changing sample number, timestamp, UUID or nonce inside the cached block.
5. Parse and validate every response independently.
6. Retry invalid trajectories only up to the configured limit and hard cost cap.

The cache saves prompt processing; it must not replace model generation. Repeated outputs are retained as legitimate stochastic outcomes unless there is evidence of response-cache replay.

## Usage and cost logging

Persist per request:

- LiteLLM call ID;
- requested and returned model;
- prompt, cached, output and total token counts;
- response cost and its input/cache-read/output components;
- finish reason, latency, retry number and validation result.

Useful response fields and headers observed from the proxy:

```text
usage.prompt_tokens
usage.prompt_tokens_details.cached_tokens
usage.completion_tokens
x-litellm-call-id
x-litellm-response-cost
x-litellm-response-cost-input
x-litellm-response-cost-cache-read
x-litellm-response-cost-output
```

Prefer the proxy-reported response cost for the ledger. If unavailable, estimate using $0.25 per million input tokens and $1.50 per million output tokens. Before sending a request, abort if its conservative estimated cost could take cumulative U1 spend beyond the USD 5.00 hard cap.

## Execution sequence

- U1.1: implement the adapter and cost ledger with mocked HTTP tests only.
- U1.2: paid smoke test on `task_42`, C0 and C1, five trajectories each.
- Stop and review validity, prompt leakage, actual costs and cached-token reporting.
- U1.3: only after approval, run all 1,000 trajectories.
