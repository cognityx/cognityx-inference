# HTTP API

## Overview

The service exposes two API layers:

- OpenAI-compatible inference at `/v1/chat/completions`
- Cognityx control endpoints for model lifecycle and discovery
- manager endpoints under `/management/v1` on the lightweight manager process

The API app is created by `cognityx_inference.api.create_app(...)`.

## OpenAI-Compatible Inference

### `POST /v1/chat/completions`

Standard OpenAI-style fields are accepted, including:

- `model`
- `messages`
- `temperature`
- `top_p`
- `max_tokens` (OpenAI compatibility alias)
- `max_completion_tokens`
- `stop`
- `seed`
- `logprobs`
- `stream`

Cognityx-specific routing and runtime fields are carried under `cognityx`:

```json
{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "Reply with OK."}
  ],
  "max_completion_tokens": 32,
  "cognityx": {
    "client_type": "openai",
    "provider": "local",
    "backend": "vllm",
    "profile": "int4",
    "load_policy": "require_loaded",
    "discovery_policy": "require_existing",
    "required_context_length": 8192,
    "top_k": 20,
    "min_p": 0.05,
    "reasoning": {},
    "timeouts": {
      "request": 300,
      "first_token": 60,
      "no_token_progress": 30
    },
    "extensions": {
      "runtime": {
        "kv_cache_precision": "fp8"
      }
    }
  }
}
```

Important behavior:

- `client_type` defaults to `openai`
- `provider` defaults to `local`
- `backend` defaults to `vllm`
- `profile` defaults to `bf16`
- local OpenAI-compatible requests use `require_loaded`; they do not trigger
  unexpected model loading
- unsupported fields are reported as unavailable, not invented

### Success Response

The response contains:

- standard OpenAI fields such as `id`, `choices`, and `usage`
- a full normalized Cognityx response under `cognityx`

The `cognityx` object includes normalized fields such as content, usage,
timings, finish reason, provider/backend/model identity, request metadata,
extensions, and local telemetry where available.

When a certified context capability applies, it also includes:

```json
{
  "token_budget": {
    "input_tokens": 120,
    "max_context_tokens": 8192,
    "reserved_tokens": 256,
    "requested_max_output_tokens": null,
    "effective_max_output_tokens": 2048,
    "source": "automatic",
    "counting": "tokenizer"
  }
}
```

The complete serialized request is counted, including system messages,
conversation history, tools, tool choice, and structured-response schema.
Manual `max_output_tokens` wins over the automatic default but is rejected,
not clamped, if it exceeds the certified context.

## Management API

Run the manager separately from the worker. Its endpoints are:

- `POST /management/v1/server/start` with `{"profile": "<name>"}`
- `POST /management/v1/server/stop`
- `GET /management/v1/server/status`
- `GET /management/v1/server/events?after=<sequence>`

Status values are `STOPPED`, `STARTING`, `READY`, `DEGRADED`, `STOPPING`, and
`FAILED`. Start and stop are idempotent. Startup returns the `server_id`,
profile/model/backend, state, worker URL, and durable `job_id`. The events
endpoint is SSE and terminates when the worker becomes ready, fails, or stops.
Reconnect with `after` to resume after the last observed sequence.

### Streaming

When `stream=true`, the service returns server-sent events in
OpenAI-compatible chunk form. Content deltas are sent as they are generated.
The terminal chunk includes the finish reason, usage, timings, and normalized
Cognityx metadata, followed by `data: [DONE]`.

For local models, a pure OpenAI-compatible client should call this endpoint
only after the model has reached `ready` through
`POST /v1/cognityx/models/load`. The Cognityx Python client performs this
lifecycle preflight automatically for `stream_chat()` and `stream_infer()`.

## Model Lifecycle Endpoints

### `POST /v1/cognityx/models/load`

Load a local model and keep it resident:

```json
{
  "model": "Qwen/Qwen3-8B",
  "backend": "vllm",
  "profile": "int4",
  "runtime": {
    "kv_cache_precision": "fp8"
  },
  "discovery_policy": "ask"
}
```

Behavior:

- resolves canonical local model identity
- checks model-declared context limit
- checks latest compatible certified hardware profile
- raises a discovery challenge when certification is missing
- returns both loaded-model status and context-resolution details

### `GET /v1/cognityx/models/status`

Return all currently tracked model statuses.

### `POST /v1/cognityx/tokens/count`

Count a message list using the active local backend's prompt formatting without
generating output. The interactive chat client uses this endpoint to reserve
certified context for both the next completion and history compression.

```json
{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "Count these tokens."}],
  "backend": "vllm",
  "profile": "int4"
}
```

The model must already be loaded. A backend that cannot count this prompt
returns `input_tokens: null`; callers must not estimate it as an exact value.

### `POST /v1/cognityx/models/unload`

Unload either:

- the active loaded instance for a model name; or
- a specific runtime identity when `runtime` is supplied

### `POST /v1/cognityx/models/unload-all`

Unload all local models managed by the service.

## Discovery Endpoints

### `GET /v1/cognityx/discoveries`

List the caller's active discovery jobs. Use `?all=true` to include terminal
jobs. Results are owner-scoped.

### `POST /v1/cognityx/discoveries`

Start explicit hardware-boundary discovery for one model/backend/profile
combination.

This is the same discovery engine used by auto-discovery from `model load` and
`infer`; the explicit endpoint just makes the job submission step visible.

### `GET /v1/cognityx/discoveries/{job_id}`

Return the current in-memory status snapshot for the discovery job, including
trial outcomes collected so far.

### `GET /v1/cognityx/discoveries/{job_id}/events`

Stream live server-sent events. Query parameter:

- `after`
  Resume after a previously seen event sequence number

### `POST /v1/cognityx/discoveries/{job_id}/cancel`

Request cancellation for a running discovery job.

The job remains owner-scoped. A caller can list only their own active jobs
unless they request `?all=true` within the same principal scope.

## Error Model

Common service errors:

- `404 model_metadata_unavailable`
  Local model metadata could not be resolved from cache and downloads are
  unavailable or disabled.
- `409 model_not_loaded`
  An OpenAI-compatible local inference request targeted a model that is not
  already resident.
- `422`
  Requested context exceeded a model/certified limit, or exact token counting
  was unavailable under the configured policy.
- `428 hardware_discovery_required`
  No compatible certified hardware result exists for the requested local load.

The `428` response includes structured fields such as model, backend, profile,
model context limit, and optional `discovery_request_id`.
