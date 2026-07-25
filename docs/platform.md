# Cognityx inference platform

## Boundaries

`cognityx_inference.contracts` defines common request, response, usage, timing,
token-detail, capability, and extension fields. Unsupported provider fields are
serialized as `null`; adapters do not estimate them.

`InferenceService` routes commercial calls to provider adapters. Local calls
acquire a `ModelLease` from `ModelManager`, ensuring an unload request drains
active work before releasing GPU resources.

Storage repositories publish logical keys below `inference/` and
`evaluations/hardware-boundary/`. Only `StorageClient.materialize()` may turn a
model key into a runtime-local path.

## Local lifecycle

Resident identity includes model, backend, and load-time runtime settings.
Sampling and generation parameters do not change that identity. A model moves
through loading, ready, busy, draining, unloading, or failed states. New leases
are refused while draining.

The default local backend is vLLM and the default profile is BF16. Loading does
not require a context length. The backend reads the model-declared maximum and
combines it with the latest compatible certified hardware result. A requested
context above either hard limit is rejected with a distinct error.

Pure OpenAI-compatible requests use `require_loaded`: they never start an
unexpected local model load. The Cognityx client additionally supports
`load_policy=auto` and discovery policies `ask`, `auto`, and
`require_existing`. HTTP clients receive status 428 with a structured discovery
challenge when certification is missing.

## Boundary evaluation

The planner enumerates only caller-supplied candidates. Trials are ordered by
load identity. A timeout or configured slow-generation result creates a
boundary on the active monotonic axis. Higher candidates are skipped only where
all other configuration values match, following the contextual pruning
convention used by `cognityx-training`.

Each executed trial is persisted independently before the immutable summary.
This avoids requiring mutable overwrite behavior from `cognityx-storage`.

Successful legacy `/benchmark` runs also publish a conservative certified
profile. They certify the prompt-plus-completion token span actually exercised,
not merely the larger engine allocation. Automatic discovery increases the
configured context through finite candidates and stops at the first failed,
slow, timed-out, or out-of-memory boundary.

## Telemetry

Local monitoring records process CPU/RAM, visible host CPU/RAM, NVIDIA
dedicated memory, utilization, temperature, and power. An optional Windows JSON
bridge can supply Windows-host and shared-GPU-memory fields. Every record
identifies its source and host scope; unavailable data remains `null`.
