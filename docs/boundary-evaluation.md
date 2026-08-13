# Boundary Evaluation

## Purpose

The original `llm-benchmark` functionality is retained inside
`cognityx-inference` as the hardware-capacity and configuration-boundary
evaluation capability.

This capability is used for two related outcomes:

- planning and running finite, user-supplied Cartesian searches across selected
  inference axes;
- producing certified local runtime profiles that the inference platform can
  reuse during model loading.

## Search Model

The search space is intentionally finite and explicit. Candidates come from a
TOML file rather than from open-ended automatic sweeps.

The current example configuration defines these ordered axes:

- `model`
- `backend`
- `quantization`
- `context_length`
- `kv_cache_precision`
- `batch_size`
- `generation_length`

The configured monotonic axes are:

- `context_length`
- `generation_length`

## Pruning Rule

The pruning rule follows the same contextual monotonic convention used in
`cognityx-training`:

- if a candidate is slow, times out, runs out of memory, or otherwise fails on
  a monotonic axis;
- then equal-or-higher values on that axis are skipped only for the same values
  on the other relevant axes.

This prevents one bad configuration from invalidating unrelated parts of the
search space.

## Discovery and Certification

Discovery jobs currently use the configured finite candidate sets to test one
requested model/backend/profile combination.

For each trial, the system records:

- complete effective configuration, including quantization, KV-cache precision,
  engine context length, GPU-memory utilization and other runtime extensions;
- representative system/user prompts, reasoning mode and generation parameters;
- available prompt, completion, reasoning and answer token breakdowns;
- model loading, prompt-processing, TTFT, generation and total duration;
- tokens per second;
- process and host CPU/RAM average and peak values;
- dedicated/shared GPU memory, GPU utilization, temperature and power average
  and peak values when the local counters provide them;
- failure or out-of-memory details when relevant.

Each trial is persisted independently before the final summary. The final
summary may also publish a `CertifiedInferenceProfile`, which is then used by
future model loads. The profile contains the selected winner's full evidence;
in particular, an `fp8` winner is certified and loaded as `fp8`, rather than
being silently represented as `auto`.

`certified_trial` is an immutable copy of the winning trial as saved by the
job. It retains the complete nested telemetry record, including every phase's
`gpu_usage` averages/peaks, dedicated/shared memory, utilization, temperature,
power draw and power limit. The shorter profile telemetry fields are retained
as convenient summaries, not as a replacement for this evidence.

Trials are grouped by engine load identity. For example, all generation-length
requests for the same model, quantization, KV-cache precision and context
length reuse one resident model. Impossible combinations where the requested
completion budget is not smaller than the engine context are recorded without
performing a model load.

Windows host and shared-GPU values are optional. Configure the compatible
Windows performance-counter bridge in the telemetry section; unavailable
fields remain `null` and identify their measurement source.

## Artifact Persistence

Boundary artifacts are stored through `cognityx-storage` and include:

- discovery manifest;
- per-trial artifact records;
- immutable summary artifact;
- certified profile artifact when a usable result exists.

Successful legacy `/benchmark` runs may also publish a conservative certified
profile based on the actual prompt-plus-completion span exercised.

## Planning CLI

The boundary planning CLI shows the finite search plan:

```bash
uv run cognityx-boundary --config examples/boundary/config.toml --plan
uv run cognityx-boundary --config examples/boundary/config.toml --plan --human
```

This is useful for reviewing the exact Cartesian combinations before starting a
live discovery or fuller evaluation workflow.
