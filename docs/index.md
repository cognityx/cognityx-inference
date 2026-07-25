# Cognityx Inference

`cognityx-inference` is the Cognityx inference platform and hardware-boundary
evaluation system. It keeps the original `llm-benchmark` capability in the same
repository, but the primary surface is now a reusable inference service with:

- a normalized inference contract for local and provider-backed models;
- an OpenAI-compatible `/v1/chat/completions` API;
- a Cognityx client and CLI for load, infer, unload, discovery, watch, and cancel;
- certified hardware-boundary profiles saved through `cognityx-storage`;
- detailed local telemetry and persisted trial artifacts;
- a durable background-job direction for discovery and future ingestion,
  dataset-preparation, and evaluation workloads.

## What This Repository Contains

- `cognityx_inference`
  The platform package: contracts, lifecycle management, providers, API,
  discovery, storage integration, and evaluation primitives.
- `llm_benchmark`
  The original interactive benchmark and diagnostics workflow, preserved as a
  compatibility capability inside the same repository.
- `examples/boundary/config.toml`
  The finite, user-supplied Cartesian search configuration used for hardware
  boundary discovery and evaluation planning.

## Recommended Reading Order

- [Platform](platform.md)
  Concepts, lifecycle rules, storage conventions, telemetry, and the current
  discovery model.
- [CLI and Python Client](cli.md)
  Commands and client calls for model lifecycle, inference, discovery, live
  job viewing, and cancellation.
- [HTTP API](http-api.md)
  OpenAI-compatible requests plus Cognityx control endpoints.
- [Boundary Evaluation](boundary-evaluation.md)
  Search axes, pruning behavior, artifacts, and certification outputs.
- [Configuration](configuration.md)
  Boundary TOML settings, accepted values, and practical defaults.
- [Architecture](architecture.md)
  High-level execution flow and module relationships.
- [Usage](usage.md)
  Legacy interactive `llm-benchmark` usage.
- [API Reference](api/index.md)
  Source-generated reference for major modules.

## Current Outcome

The repository already supports:

- serving an inference API with local `vllm` and `transformers` backends;
- loading and keeping local models resident with certification-aware runtime
  resolution;
- inference through the CLI, Python client, or OpenAI-compatible HTTP calls;
- discovery start, status, event streaming, and cancellation endpoints;
- a practical CLI workflow for manual discovery, auto discovery, live status
  viewing, and cancellation;
- finite hardware-boundary candidate configuration with contextual monotonic
  pruning;
- persistence of inference artifacts, discovery manifests, trials, summaries,
  and certified profiles through `cognityx-storage`;
- compatibility preservation for the original interactive benchmark path.

The discovery job flow is available now for practical use, while broader
execution unification between the explicit boundary-evaluation planner and every
future durable worker type is still an area for continued evolution.

## Start Here

Use the copy-paste workflow in [CLI and Python Client](cli.md). It deliberately
uses separate local ports:

- inference API: `http://127.0.0.1:8013`;
- MkDocs: `http://127.0.0.1:8000`.

The guide covers explicit model loading, one-call automatic discovery and load,
streaming inference, complete JSON responses, status inspection, cancellation,
and unload from both the CLI and Python.
