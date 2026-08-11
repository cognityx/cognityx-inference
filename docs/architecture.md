# Architecture blueprint

## Repository shape

This repository now contains two related but distinct layers:

- `cognityx_inference`
  The reusable inference platform used by other Cognityx services.
- `llm_benchmark`
  The preserved interactive benchmark and diagnostics capability.

## Platform execution flow

```mermaid
flowchart LR
    Client[CLI / Python client / HTTP client] --> API[FastAPI endpoints]
    API --> Service[InferenceService]
    Service --> Providers[Commercial provider adapters]
    Service --> Manager[ModelManager]
    Manager --> LocalBackends[vLLM or Transformers backend]
    Service --> Storage[Artifact repositories]
    Storage --> Adapter[Verified Training adapter]
    Adapter --> LocalBackends
    Service --> Discovery[BoundaryDiscoveryCoordinator]
    Discovery --> Storage
    Discovery --> Manager
```

`InferenceService` is the orchestration center. It decides whether a request
goes to a provider adapter or a locally managed model, and it centralizes
artifact persistence plus context/certification resolution.

For research adapters, Storage first resolves the public Training manifest and
verifies every immutable file. The service compares that manifest with the
actual resident base identity before the backend sees it. vLLM then applies one
adapter to a request without treating it as another full resident model. The
[adapter handoff guide](adapter-handoff.md) describes paired publication and
the boundary with DataForge, Training, MLflow, and Evaluator.

## Local lifecycle flow

```mermaid
sequenceDiagram
    actor Caller
    participant Client as Cognityx client
    participant API as API
    participant Service as InferenceService
    participant Profiles as Certified profiles
    participant Models as ModelManager
    participant Backend as Local backend

    Caller->>Client: load_model(model, backend, profile)
    Client->>API: POST /v1/cognityx/models/load
    API->>Service: load_model(...)
    Service->>Profiles: find compatible certification
    alt certification exists
        Service->>Models: load(model, backend, runtime)
        Models->>Backend: initialize resident model
        Backend-->>Caller: loaded and ready
    else no certification
        Service-->>API: hardware discovery required
    end
```

The key idea is that load-time residency is defined by model, backend, and
runtime identity. Sampling parameters are request-scoped and do not define the
resident model identity.

## Manager and worker processes

The lightweight `InferenceManager` is a separate control process. It remains
available while the heavyweight worker is stopped and manages one local worker
in this release.

```mermaid
flowchart LR
    Client[Cognityx client or CLI] --> ManagerAPI[Management API]
    ManagerAPI --> InferenceManager
    InferenceManager --> Jobs[cognityx-jobs events]
    InferenceManager --> State[cognityx-storage state]
    InferenceManager --> Worker[Local inference worker]
    Client --> Worker
```

The manager persists only secret-free state: server/job identity, named
profile, model/backend, state, PID/process group, worker URL, timestamps, and a
categorical failure. It never returns the worker command, environment, or
credentials. Startup is single-flight for a profile, so concurrent callers
share one job. Stop first requests graceful process-group termination and then
uses a bounded force fallback.

The worker still uses the existing `InferenceService`, certified-profile
repository, `ModelManager`, leases, and vLLM/Transformers backends. The manager
does not duplicate those responsibilities.

## Discovery flow

```mermaid
sequenceDiagram
    actor Caller
    participant API as API
    participant Service as InferenceService
    participant Discovery as BoundaryDiscoveryCoordinator
    participant Models as ModelManager
    participant Storage as cognityx-storage

    Caller->>API: start discovery or auto-trigger load
    API->>Service: prepare runtime / start discovery
    Service->>Discovery: start(model, backend, profile, runtime)
    Discovery->>Storage: save manifest
    loop configured candidates
        Discovery->>Models: acquire model for candidate runtime
        Models-->>Discovery: backend lease
        Discovery->>Discovery: run trial and measure timing
        Discovery->>Storage: save trial artifact
        Discovery-->>Caller: SSE trial events
    end
    Discovery->>Storage: save summary + certified profile
    Discovery->>Models: load winning runtime
```

This is the operational bridge between boundary evaluation and ordinary
inference use.

## Control surfaces

The current primary surfaces are:

- `cognityx-inference serve`
- `cognityx-inference model ...`
- `cognityx-inference infer ...`
- `cognityx-inference discovery ...`
- `POST /v1/chat/completions`
- `POST /v1/cognityx/models/load`
- `GET /v1/cognityx/discoveries/{job_id}/events`

## Legacy benchmark flow

The benchmark workflow remains intact and continues to use its own orchestration
and reporting path:

```mermaid
flowchart LR
    main --> app
    app --> LocalLLM[llm.LocalLLM]
    app --> VLLMLLM[vllm_engine.VLLMLLM]
    app --> reporting
    LocalLLM --> model_diagnostics
    VLLMLLM --> model_diagnostics
```

That path still covers interactive prompts, `/benchmark`, long-context corpora,
diagnostics, comparison, and saved benchmark JSON.

## Provider architecture

```mermaid
flowchart LR
    Client[Cognityx client / OpenAI client] --> API[OpenAI-compatible API]
    API --> Service[InferenceService]
    Service --> Registry[ProviderRegistry]
    Service --> Router[Purpose RoutingPolicy]
    Registry --> Local[Local model lifecycle]
    Registry --> Compat[OpenAI-compatible adapter]
    Registry --> Gemini[Gemini native adapter]
    Registry --> Anthropic[Anthropic native adapter]
    Registry --> GitHub[GitHub Models adapter]
```

Provider definitions and model profiles are configuration data. The registry
owns provider state, exact model discovery with a TTL cache, capabilities, and
safe status. Adapters translate transport; they do not decide routing. Parameter
policies reject known unsupported combinations before a commercial request.
The service retains the common request/response contract and represents
unavailable provider information as unavailable rather than estimated.

The local provider remains special only for lifecycle and machine telemetry.
Commercial providers have no load/unload operation and expose only
provider-reported usage and latency. This preserves the existing local model
manager, leases, boundary certification, and `cognityx-storage` paths.
