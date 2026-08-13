# CLI and Python Client

## Service Startup

Use port `8013` for the inference API. Port `8000` is reserved for the local
MkDocs documentation server in these instructions.

```bash
uv sync --extra api --extra telemetry
uv run cognityx-inference serve --host 127.0.0.1 --port 8013
```

Keep this terminal open. In other terminals, define a shell variable for
copy-paste-safe commands:

```bash
export COGNITYX_INFERENCE_URL=http://127.0.0.1:8013
```

The CLI does not read this variable automatically; the commands below pass it
through `--base-url`.

If commercial providers are needed, set `OPENAI_API_KEY`, `GROQ_API_KEY`, or
`XAI_API_KEY`
before starting the service.

The discovery job database is shared with the service process. For a local
single-user workflow, keep it in the repository root so the CLI and server see
the same state:

```bash
export COGNITYX_JOBS_DATABASE="$PWD/cognityx_jobs.sqlite3"
```

## Copy-Paste CLI Workflow

Run these commands from the repository root.

Terminal 1 — start the inference API:

```bash
export COGNITYX_JOBS_DATABASE="$PWD/cognityx_jobs.sqlite3"
uv run cognityx-inference serve --host 127.0.0.1 --port 8013
```

Terminal 2 — select the API, load the model, and wait for readiness:

```bash
export COGNITYX_INFERENCE_URL=http://127.0.0.1:8013

uv run cognityx-inference model load \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4 \
  --discovery-policy ask

uv run cognityx-inference model status \
  --base-url "$COGNITYX_INFERENCE_URL"
```

`loading` means initialization is still in progress. `ready` means the model
can accept inference. The load command itself waits until loading or any
required discovery has finished.

Then run streaming inference:

```bash
uv run cognityx-inference infer \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4 \
  --prompt "Explain KV cache precision." \
  --no-thinking \
  --max-output-tokens 256 \
  --discovery-policy ask
```

Thinking is disabled when neither flag is supplied. Add `--thinking` only when
the selected model and backend expose a real native control. Qwen3 uses its
tokenizer chat-template switch rather than an output-text rewrite. The normal
`infer`, interactive `chat`, and `research pair` commands all accept
`--thinking` and `--no-thinking`; a research pair defaults to no thinking and a
512-token output allowance.

For an explicitly experimental Training adapter, add
`--adapter-manifest storage://.../adapter-manifest.json` and
`--adapter-purpose evaluation`. For a frozen base-versus-adapter batch, use
`cognityx-inference research pair`. The complete contract and the RTX 5090
copy-paste smoke workflow are in
[Training adapters and paired research runs](adapter-handoff.md).

## CLI Overview

The main operational CLI is `cognityx-inference`.

### Run the manager and start a worker

Prepare `.cognityx/inference.toml` as described in
[Configuration](configuration.md), then keep the lightweight manager running:

```bash
export COGNITYX_INFERENCE_CONFIG="$PWD/.cognityx/inference.toml"
uv run cognityx-inference manager serve
```

In another terminal:

```bash
uv run cognityx-inference server status
uv run cognityx-inference server start --profile qwen3-8b-int4
uv run cognityx-inference server watch
uv run cognityx-inference server stop
```

Use `--manager-url http://127.0.0.1:8000` when the manager is not at its
default URL. `server watch` prints durable startup events and can reconnect
with `--after <sequence>`.

`server status --human` renders the same finite manager response as labelled
sections. `server watch --human` renders each event immediately while retaining
its sequence and replay cursor; the unchanged default remains one compact JSON
event per line.

The earlier `cognityx-inference serve` command remains the direct worker mode.
Use it when an external supervisor already owns the worker lifecycle.

### Test configured commercial providers

Provider smoke tests are explicit and make only small calls:

```bash
uv run cognityx-inference providers test --provider openai \
  --config .cognityx/inference.toml
uv run cognityx-inference providers test --provider groq \
  --config .cognityx/inference.toml
uv run cognityx-inference providers test --all \
  --config .cognityx/inference.toml
```

The diagnostic authenticates through the model-list endpoint, then validates
one tiny non-streaming and one streaming `OK` response. Output contains only
provider, configured model, status, latency, and a safe error category. A
missing credential is `skipped`.

Credentials may come from the configured JSON path; exporting provider keys is
not required:

```toml
secrets_file = "/secure/location/provider-secrets.json"
```

Successful diagnostic output includes measured prompt/completion/total usage
and the provider's allowlisted rate-limit headers when supplied:

- remaining requests;
- remaining tokens;
- request/token reset windows.

These are rate-window budgets, not account credit balances. Provider account
balance is reported as unavailable because the inference APIs do not return a
portable billing balance.

Automated tests never make these calls unless explicitly enabled:

```bash
export COGNITYX_LIVE_PROVIDER_TESTS=1
export COGNITYX_INFERENCE_CONFIG="$PWD/.cognityx/inference.toml"
uv run pytest -q tests/test_live_providers.py
```

Both the opt-in flag and each provider credential must be present; otherwise
that provider's live test is skipped.

Applications use the same client contract for configured providers:

```python
from cognityx_inference import InferenceClient

client = InferenceClient("http://127.0.0.1:8013")
reply = client.chat(
    model="gpt-4.1-mini",
    provider="openai",
    messages=[
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Reply with OK."},
    ],
    temperature=0,
    max_output_tokens=8,
)
print(reply["cognityx"]["usage"])
print(reply["cognityx"]["token_budget"])
print(reply["cognityx"]["extensions"]["rate_limits"])
```

Change `provider` and `model` to the configured Groq values without changing
application flow. Remote providers do not start a local worker.

### Load a Local Model

```bash
uv run cognityx-inference model load \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4 \
  --discovery-policy ask
```

Notes:

- default backend is `vllm`;
- default profile is `bf16`;
- no context length is supplied at load time;
- the service resolves the effective limit from model metadata plus the latest
  compatible certified hardware result;
- if no certification exists, discovery policy controls what happens next.

Discovery policies:

- `require_existing`
  Fail immediately if no compatible certified profile exists.
- `ask`
  Prompt in the CLI client before starting discovery.
- `auto`
  Start discovery automatically and wait for completion.

For unattended execution, replace `ask` with `auto`. To prohibit discovery,
use `require_existing`.

One-call load/discover/infer is also supported. The streaming `infer` command
performs a lifecycle preflight, reuses a compatible resident model when one is
already ready, and otherwise follows the selected discovery policy:

```bash
uv run cognityx-inference infer \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4 \
  --prompt "Reply with READY." \
  --discovery-policy auto
```

### Recommended Discovery Modes

There are two common ways to use hardware-boundary discovery:

1. Manual discovery first, then load or infer with `require_existing`.
2. Auto discovery from `model load` or `infer` when you want the client to
   discover a certified boundary on demand.

Both paths use the same discovery engine and the same job status and cancel
commands.

## Live Job Workflow

Discovery runs as a background job. While a job is active, you can keep one
terminal focused on the service and use another terminal to watch status,
stream events, or cancel the job.

Example layout:

```text
Terminal 1: uv run cognityx-inference serve --host 127.0.0.1 --port 8013
Terminal 2: discovery start / model load / infer
Terminal 3: discovery status / discovery watch / discovery cancel
```

If auto discovery is triggered from `model load` or `infer`, the CLI prints the
job ID immediately and keeps streaming progress. You can copy that ID into a
second shell and use the same `status`, `watch`, or `cancel` commands.

### Inspect Loaded Models

```bash
uv run cognityx-inference model status \
  --base-url "$COGNITYX_INFERENCE_URL"
```

### Unload One Model

```bash
uv run cognityx-inference model unload \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B
```

### Unload Everything

```bash
uv run cognityx-inference model unload-all \
  --base-url "$COGNITYX_INFERENCE_URL"
```

## Inference

### Simple Local Inference

```bash
uv run cognityx-inference infer \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4 \
  --prompt "Explain KV cache precision." \
  --max-output-tokens 256 \
  --discovery-policy ask
```

Inference streams generated text to the terminal by default. The Cognityx
client first ensures that the certified local model is loaded, then opens an
OpenAI-compatible SSE request using `require_loaded`.

To wait for completion and print the original full JSON response instead:

```bash
uv run cognityx-inference infer \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B \
  --prompt "Explain KV cache precision." \
  --no-stream
```

Optional CLI parameters currently exposed on the command line:

- `--required-context-length`
- `--max-output-tokens` (`--max-tokens` remains a migration alias)
- `--backend`
- `--profile`
- `--discovery-policy`

The normalized request contract supports more parameters than the current CLI
flags expose. For full control, use the Python client or direct HTTP payloads.

## Discovery Jobs

### Start Discovery Explicitly

This is the most direct way to run the boundary evaluator when you already know
the model, backend, and profile you want to test.

```bash
uv run cognityx-inference discovery start \
  --base-url "$COGNITYX_INFERENCE_URL" \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4
```

### Watch Live Discovery Events

Use this in a second terminal while discovery is running.

```bash
uv run cognityx-inference discovery watch \
  --base-url "$COGNITYX_INFERENCE_URL" \
  <job-id>
```

This streams server-sent events from the service. Each event is printed as one
JSON object and may include:

- `discovery_started`
- `trial_started`
- `trial_completed`
- `cancel_requested`
- `discovery_completed`
- `discovery_failed`
- `discovery_cancelled`

### List Your Discovery Jobs

```bash
uv run cognityx-inference discovery status \
  --base-url "$COGNITYX_INFERENCE_URL"
```

This lists the current caller's active jobs and their IDs. Use `--all` to
include history.

Typical use:

1. Run `discovery status` in a second terminal.
2. Copy the job ID from the active row.
3. Use `discovery watch <job-id>` for live SSE progress.
4. Use `discovery cancel <job-id>` if the trial is no longer useful.

The command is owner-scoped, so one user only sees their own jobs unless they
explicitly ask for history in the same principal scope.

### List Certified Profiles

Discovery runs and certified profiles are separate: a run is a job history;
a profile is a reusable hardware result. List profiles without manually
browsing storage paths:

```bash
uv run cognityx-inference certified-profiles list \
  --base-url "${COGNITYX_INFERENCE_URL:-http://127.0.0.1:8000}" \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4
```

Inspect one complete result, including `certified_trial` and its nested phase
telemetry:

```bash
uv run cognityx-inference certified-profiles show <profile-id> \
  --base-url "${COGNITYX_INFERENCE_URL:-http://127.0.0.1:8000}"
```

Control-plane commands use readable terminal output by default. List/status
commands render compact tables; profile show renders identity, runtime,
performance, overall resources and measured phases. For the exact API payload
for scripts or archival use, add `--format json`:

```bash
uv run cognityx-inference certified-profiles show <profile-id> \
  --format json

uv run cognityx-inference discovery watch <job-id> \
  --format json
```

The existing `--format table|detail|json` contract remains authoritative for
model, discovery, and certified-profile commands. Those established human
renderers are not replaced by the additive `--human` option used for
configuration, manager status, and provider gaps.

`discovery watch` otherwise shows one concise line per event with trial
progress, context, generation length, KV-cache precision, elapsed time,
throughput and TTFT.

The client uses `COGNITYX_INFERENCE_URL` when present, otherwise defaults to
`http://127.0.0.1:8000`. Passing an empty `--base-url` follows the same rule.

### Cancel a Discovery Job

```bash
uv run cognityx-inference discovery cancel \
  --base-url "$COGNITYX_INFERENCE_URL" \
  <job-id>
```

## Python Client

The Python client accepts the same OpenAI-style `messages` list used by the
HTTP API, so system and user prompts are just dictionaries in order.

### Explicit Load Followed by Repeated Inference

Use explicit loading when an application wants to prepare the model before
accepting requests:

```python
from cognityx_inference import CognityxInferenceClient

client = CognityxInferenceClient(
    "http://127.0.0.1:8013",
    discovery_policy="ask",
)

loaded = client.load_model(
    "Qwen/Qwen3-8B",
    backend="vllm",
    profile="int4",
    discovery_policy="ask",
)

print(loaded["state"])
print(client.model_status())

final_chunk = None
for chunk in client.stream_chat(
    model="Qwen/Qwen3-8B",
    client_type="openai",
    backend="vllm",
    profile="int4",
    messages=[
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Explain KV caching."},
    ],
    max_output_tokens=128,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.05,
    stop=["END"],
    seed=42,
    log_probabilities=True,
    discovery_policy="require_existing",
):
    choices = chunk.get("choices") or []
    if choices:
        text = (choices[0].get("delta") or {}).get("content")
        if text:
            print(text, end="", flush=True)
    if "usage" in chunk:
        final_chunk = chunk

print()
if final_chunk is not None:
    print("usage:", final_chunk["usage"])
    print("finish reason:", final_chunk["choices"][0]["finish_reason"])
    print("timings:", final_chunk["cognityx"]["timings"])
```

`stream_chat()` performs a lightweight load preflight. If the same compatible
model is already resident, the lifecycle manager reuses it; weights are not
loaded again.

The common inference parameters travel with the call:

- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `max_output_tokens` (`max_tokens` remains a Python migration alias)
- `stop`
- `seed`
- `log_probabilities`
- `top_log_probabilities`
- `required_context_length`

The `client_type` argument defaults to `openai`. Use it when you want the
client-facing interface to stay explicit even as more client types are added
later.

### Inspect Discovery Runs and Certified Profiles

```python
runs = client.list_discoveries(include_history=True)

profiles = client.list_certified_profiles(
    model="Qwen/Qwen3-8B",
    backend="vllm",
    profile="int4",
)
for profile in profiles:
    print(profile["profile_id"], profile["compatibility"]["kv_cache_precision"])

complete_profile = client.get_certified_profile(profiles[0]["profile_id"])
print(complete_profile["certified_trial"]["metrics"]["resource_summary"])
```

Constructing `CognityxInferenceClient()` with no URL, or an empty URL, uses
`COGNITYX_INFERENCE_URL` when set and otherwise uses `http://127.0.0.1:8000`.

### One-Call Automatic Load or Discovery

An application may omit a separate `load_model()` call:

```python
from cognityx_inference import CognityxInferenceClient

client = CognityxInferenceClient(
    "http://127.0.0.1:8013",
    discovery_policy="auto",
)

for chunk in client.stream_chat(
    model="Qwen/Qwen3-8B",
    client_type="openai",
    backend="vllm",
    profile="int4",
    messages=[
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Explain KV caching."},
    ],
    max_output_tokens=128,
    discovery_policy="auto",
):
    choices = chunk.get("choices") or []
    if choices:
        print(
            (choices[0].get("delta") or {}).get("content", ""),
            end="",
            flush=True,
        )
print()
```

This sequence checks certification, starts durable discovery when necessary,
loads the certified configuration, keeps it resident, and then opens the SSE
inference stream.

### Manager-owned local auto-start

Use this mode when the manager should start the named worker on demand:

```python
from cognityx_inference import InferenceClient, InferenceRequest

events = []
client = InferenceClient(
    backend="local",
    profile="qwen3-8b-int4",
    auto_start=True,
    manager_url="http://127.0.0.1:8000",
    startup_timeout_seconds=600,
    on_server_event=events.append,
)

reply = client.infer(
    InferenceRequest(
        model="Qwen/Qwen3-8B",
        messages=(
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Explain KV cache precision."},
        ),
        max_output_tokens=None,
    )
)
print(reply["choices"][0]["message"]["content"])
print(reply["cognityx"]["token_budget"])
```

The client checks manager status, submits one idempotent start, follows
reconnectable events until `READY`, and then sends the original request to the
worker using `require_loaded`. Concurrent callers for the same profile share
the same manager startup. Commercial requests never invoke this path.

If you want the raw non-streaming JSON response instead of SSE chunks, use
`chat(...)` with the same message list and parameters:

```python
reply = client.chat(
    model="Qwen/Qwen3-8B",
    client_type="openai",
    backend="vllm",
    profile="int4",
    messages=[
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Explain KV caching."},
    ],
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.05,
    max_output_tokens=128,
    stop=["END"],
    seed=42,
    discovery_policy="require_existing",
)
print(reply["choices"][0]["message"]["content"])
```

### Non-Streaming JSON and Unload

```python
reply = client.chat(
    model="Qwen/Qwen3-8B",
    backend="vllm",
    profile="int4",
    prompt="Reply with READY.",
    max_output_tokens=32,
    discovery_policy="require_existing",
)
print(reply["choices"][0]["message"]["content"])

print(client.model_status())
print(client.unload_model("Qwen/Qwen3-8B", "vllm"))
# Or release every local model:
# print(client.unload_all())
```

Useful client methods:

- `load_model(...)`
- `model_status()`
- `unload_model(...)`
- `unload_all()`
- `chat(...)`
- `stream_chat(...)`
- `infer(InferenceRequest(...))`
- `stream_infer(InferenceRequest(...))`
- `start_discovery(...)`
- `discovery_status(job_id)`
- `stream_discovery(job_id)`
- `cancel_discovery(job_id)`

The client can also subscribe to live discovery events while it waits for an
auto-discovery path to complete. That is the same mechanism the CLI uses for
the two-terminal live progress view.

## OpenAI-Compatible Clients

Pure OpenAI-compatible clients can call `/v1/chat/completions`, but they do not
automatically load local models. Those requests succeed only when the requested
local model is already loaded, or when the request targets an external provider.

Use the Cognityx client when you want one interface that can both:

- manage local model lifecycle; and
- submit inference requests across local and provider-backed backends.

## Troubleshooting

### Model status returns an HTML or non-JSON 404

Confirm that `--base-url` points to the inference API. In this guide:

- `http://127.0.0.1:8013` is the inference API;
- `http://127.0.0.1:8000` is MkDocs.

Check the listening processes:

```bash
ss -ltnp 'sport = :8013'
ss -ltnp 'sport = :8000'
```

Then retry:

```bash
uv run cognityx-inference model status \
  --base-url http://127.0.0.1:8013
```

### Status remains `loading`

Model loading is synchronous inside the load request but visible through the
status endpoint from another terminal. Large checkpoints may take several
minutes. Do not start a second server for the same GPU; monitor the existing
server and wait for `ready` or `failed`.

### No compatible hardware certification exists

- use `--discovery-policy ask` for an interactive confirmation;
- use `--discovery-policy auto` for unattended execution;
- use `discovery status`, `discovery watch <job-id>`, and
  `discovery cancel <job-id>` from another terminal.

### Model metadata cannot be found

Use the exact cached Hugging Face model identifier. For example, the existing
cache directory `models--Qwen--Qwen3-8B` corresponds to
`Qwen/Qwen3-8B`, not `Qwen/Qwen-8B`. Confirm that `HF_HOME` points at the
intended shared cache before starting the server.

### Manager reports `FAILED` or `DEGRADED`

Run:

```bash
uv run cognityx-inference server status
uv run cognityx-inference server watch
```

`FAILED` includes a categorical error such as `worker_startup_timeout` or
`worker_load_http_400`; secrets and raw provider responses are never returned.
`DEGRADED` means a recovered worker PID exists but its health endpoint is not
ready. Stop the worker, correct its profile/certification, and start it again.

## Provider operations

```bash
# Human-readable status; add --json for automation.
uv run cognityx-inference providers list
uv run cognityx-inference providers status

# Safe empty credential template and exact requirements.
uv run cognityx-inference providers setup
uv run cognityx-inference providers setup --create-template --yes

# Exact model identifiers, declared capability/profile, and smoke tests.
uv run cognityx-inference providers models --provider groq --refresh
uv run cognityx-inference providers capabilities \
  --provider groq --model llama-3.1-8b-instant
uv run cognityx-inference providers test --provider groq
uv run cognityx-inference providers test --all
```

`test --all` skips providers without credentials and reports why. It never
prints credential values. Add `--start-local` to start the configured local
worker for its smoke test, and `--stop-local` to release it afterward.

Python exposes the same control surface:

```python
from cognityx_inference import CognityxInferenceClient

client = CognityxInferenceClient("http://127.0.0.1:8000")
print(client.list_providers())
print(client.provider_status())
print(client.provider_models("openai", refresh=True))
print(client.provider_capabilities("openai", "gpt-4.1-mini"))
print(client.test_provider("openai", model="gpt-4.1-mini"))
```

For inference, use the existing message-list contract and choose a provider:

```python
reply = client.chat(
    provider="openai",
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Explain KV cache precision."},
    ],
    temperature=0.2,
    top_p=0.9,
    max_output_tokens=256,
)
```

Provider-specific request metadata belongs in extensions. Unsupported common
parameters are rejected explicitly instead of silently changing semantics.
