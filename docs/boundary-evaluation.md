# Boundary Evaluation

## Purpose

The original `llm-benchmark` repository is retained inside
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

- configuration values;
- runtime duration;
- model loading time;
- generation duration;
- tokens per second;
- time to first token when available;
- failure or out-of-memory details when relevant.

Each trial is persisted independently before the final summary. The final
summary may also publish a `CertifiedInferenceProfile`, which is then used by
future model loads.

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
```

This is useful for reviewing the exact Cartesian combinations before starting a
live discovery or fuller evaluation workflow.
