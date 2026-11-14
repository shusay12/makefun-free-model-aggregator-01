# Free-Model Aggregator

One Python server (standard library only) that serves two public APIs — the
OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in
server-sent events — by routing every request to the best free model across
ten free-tier providers.

## Run

```
python3 main.py          # serve: registers with $MAKEFUN_ENV_URL, works, exits 0
python3 main.py --list   # print the catalog document and exit 0
```

The trial brief (`$MAKEFUN_PROMPT_FILE`) carries the provider directory
(ids, wires, auth schemes, base paths, keys), the sealed quality table
(rating, context length, tools/vision/JSON support per model) and the
workload manifest (each request's requirements). At startup the server
probes every provider's own model list, builds the catalog document from
what the providers report, and registers its port and that catalog with the
env (`POST $MAKEFUN_ENV_URL/register`, protocol version 1). It then polls
`GET $MAKEFUN_ENV_URL/work` and exits 0 when the workload is done.

## How a request is served

Requests arrive one at a time, in manifest order, and name model `auto`;
the requirements of the request now arriving are read from the manifest
(the body itself is the fallback). Candidates are the free models that meet
the requirements — tools, vision, JSON mode, minimum context — ranked by
the quality table's rating, ties broken by directory order then model id.
A request that asks for JSON mode is never routed to the Anthropic wire,
which cannot carry that flag to echo back.

The server speaks each provider's own wire:

- OpenAI: `POST <base>/v1/chat/completions`, `GET <base>/v1/models`
- Anthropic: `POST <base>/v1/messages`, `GET <base>/v1/models`
- Gemini: `POST <base>/v1beta/models/<model>:generateContent`,
  `GET <base>/v1beta/models`

with the provider's own auth header (`Authorization: Bearer`, `x-api-key`,
`x-goog-api-key`, or none). A same-wire request is passed through untouched
apart from the chosen model; a cross-wire request is translated field by
field — system prompt, user text, images, tools, stop sequences,
`max_tokens`, temperature and the JSON-mode flag all travel. The provider's
answer text is returned whole, its usage counts as that wire reports them,
and its stop reason mapped into the client's wire (`end_turn` → `stop`,
`max_tokens` → `length`, `tool_use` → `tool_calls`, `refusal` →
`content_filter`, and back).

## Failover

A refusal (402, 429 or 5xx) is never the client's answer: the request moves
to the next candidate. A provider that refuses three calls in a row is
treated as spent for the rest of the trial and skipped (later candidates
from it are tried only if nothing else is left), which keeps the call
count — the tie-break — low. Every choice is a function of the brief and
the providers' answers seen so far: no wall clock, no randomness, so a
replayed trial makes the same calls and leaves the same state.

## Catalog

`GET /v1/models` answers in the wire the caller's auth header names:
`Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic
one — in both cases exactly the free model ids the catalog registers.
