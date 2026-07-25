# Cognityx inference platform

## Normalized contracts

`cognityx_inference.contracts` defines the common request and response shape
used across:

- local self-hosted inference;
- commercial provider adapters;
- the Python client;
- the OpenAI-compatible HTTP layer.

The normalized contract covers common industry parameters and outputs such as:

- `temperature`, `top_p`, `top_k`, `min_p`;
- `max_tokens`, `stop`, `seed`;
- log probabilities and token details where supported;
- prompt, completion, and total token usage;
- finish reason;
- latency, time to first token, and tokens per second where available;
- reasoning mode when supported;
- backend-specific extension fields without breaking the common interface.

Unsupported fields remain explicitly unavailable rather than being estimated.

## Service routing

`InferenceService` is the central application service.

- provider requests are routed to commercial adapters such as OpenAI or xAI;
- local requests are routed through `ModelManager` and the selected local
  backend;
- artifacts are persisted through storage repositories instead of direct path
  construction.

For local inference, the service resolves canonical model identity, compatible
hardware certification, and effective context length before model acquisition.

## Storage conventions

`cognityx-inference` uses `cognityx-storage` for persisted artifacts such as:

- inference exchanges;
- boundary manifests;
- boundary trials;
- boundary summaries;
- certified inference profiles.

Logical keys are published under storage-managed namespaces such as
`inference/` and `evaluations/hardware-boundary/`. Only storage abstractions may
materialize physical locations.

## Local model lifecycle

Resident identity is based on:

- model;
- backend;
- load-time runtime settings.

Sampling parameters do not define residency. A model can move through states
such as loading, ready, busy, draining, unloading, or failed.

Important lifecycle rules:

- local models are loaded explicitly through the Cognityx lifecycle API or
  client;
- OpenAI-compatible local requests do not auto-load a model;
- unload must not interrupt active work unexpectedly;
- repeated requests can reuse a resident model without paying load time again.

The default local backend is `vllm`, and the default profile is `bf16`.

## Context and certification

Loading does not require a caller-supplied context length. Instead, the service
combines:

- the model-declared maximum context length; and
- the latest compatible certified hardware result for the active machine and
  runtime compatibility.

If the caller later requests a context length above either hard limit, the
request is rejected clearly.

If no certification exists, the system raises a structured discovery challenge.
Depending on the client path and discovery policy, that may:

- fail immediately;
- prompt the CLI user;
- or start discovery automatically and wait for completion.

## Boundary evaluation and discovery

Boundary discovery is the operational bridge between evaluation and inference.

- evaluation uses explicit finite candidate values from configuration;
- discovery tests those values for one requested model/backend/profile scope;
- successful evidence is converted into a certified profile;
- later model loads reuse that certification automatically.

Trials are append-only. Each executed trial is persisted before the immutable
summary so the system does not rely on overwrite semantics in storage.

Legacy `/benchmark` results can also publish conservative certified profiles
based on the actual exercised prompt-plus-completion span.

## Background jobs

Discovery is exposed as a background job with:

- start;
- status;
- live server-sent event streaming;
- cancellation.

This is the start of a broader Cognityx durable job backbone intended for
additional future workloads beyond inference discovery.

## Telemetry

For local execution, the platform captures as much machine telemetry as practical,
including:

- dedicated GPU memory;
- shared GPU memory where available;
- GPU utilization;
- GPU temperature;
- GPU power;
- CPU and RAM usage;
- process-level resource information;
- model loading time;
- prompt and generation timing;
- tokens per second and time to first token.

Commercial providers normally contribute provider-reported usage and latency,
but not their internal machine telemetry.
