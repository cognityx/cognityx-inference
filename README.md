# Cognityx Inference

`cognityx-inference` is the Cognityx model-inference platform and
hardware/configuration boundary evaluator. It evolves the original
`llm-benchmark` repository without discarding its persistent Transformers and
vLLM engines, diagnostics, interactive commands, or saved-result structure.

The platform adds normalized inference contracts, an OpenAI-compatible API and
client, explicit local-model lifecycle leases, local/OpenAI/xAI routing,
logical-key persistence through `cognityx-storage`, finite Cartesian boundary
plans with contextual monotonic pruning, and local machine telemetry.

The legacy `llm-benchmark` command and `llm_benchmark` imports remain available
as compatibility interfaces.

## Platform quick start

```bash
uv sync --extra api --extra telemetry
uv run cognityx-inference serve --host 127.0.0.1 --port 8013
```

Commercial adapters are enabled only when their configured environment
credential is present. OpenAI, Groq, and the existing xAI adapter are
supported; credentials never belong in normal configuration.

Run the lightweight manager and let the Cognityx client start one named local
worker on demand:

```bash
mkdir -p .cognityx
cp configs/inference.toml.example .cognityx/inference.toml
export COGNITYX_INFERENCE_CONFIG="$PWD/.cognityx/inference.toml"
uv run cognityx-inference manager serve

# In another terminal:
uv run cognityx-inference server start --profile qwen3-8b-int4
uv run cognityx-inference server watch
uv run cognityx-inference server status
```

The manager persists secret-free state and durable startup events. The worker
continues to use the existing inference service, lifecycle leases, certified
profiles, and local backends.

Inspect a finite boundary plan without loading a model:

```bash
uv run cognityx-boundary --config examples/boundary/config.toml --plan
```

Every candidate is supplied explicitly. For monotonic axes, an unacceptable
time or generation-speed boundary prunes equal-or-higher values only for the
same values on all other axes. Compatible load-time identities are grouped so
request-time combinations reuse a resident model.

Load a model using its latest compatible certified hardware profile:

```bash
uv run cognityx-inference model load \
  --base-url http://127.0.0.1:8013 \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4 \
  --discovery-policy ask
```

No context allocation is required. The platform reads the model-declared limit
and the latest matching hardware certification. If no certification exists,
`ask` offers to run an increasing context-boundary discovery, persist every
trial through `cognityx-storage`, keep the final successful configuration
loaded, and then continue.

```bash
uv run cognityx-inference infer \
  --base-url http://127.0.0.1:8013 \
  --model Qwen/Qwen3-8B \
  --profile int4 \
  --prompt "Explain KV caching."

uv run cognityx-inference model status \
  --base-url http://127.0.0.1:8013

uv run cognityx-inference model unload \
  --base-url http://127.0.0.1:8013 \
  --model Qwen/Qwen3-8B
```

`infer` streams generated text by default. Add `--no-stream` to wait and print
the complete JSON response.

The Python client supports the same `ask`, `auto`, and `require_existing`
discovery policies. Pure OpenAI-compatible requests do not auto-load models;
they return HTTP 409 until the requested model has been loaded through the
Cognityx lifecycle API.

```python
from cognityx_inference import CognityxInferenceClient

client = CognityxInferenceClient(
    "http://127.0.0.1:8013",
    discovery_policy="ask",
)
client.load_model(
    "Qwen/Qwen3-8B",
    backend="vllm",
    profile="int4",
)

for chunk in client.stream_chat(
    model="Qwen/Qwen3-8B",
    backend="vllm",
    profile="int4",
    prompt="Explain KV caching.",
):
    choices = chunk.get("choices") or []
    if choices:
        print(
            (choices[0].get("delta") or {}).get("content", ""),
            end="",
            flush=True,
        )
```

See [CLI and Python Client](docs/cli.md) for explicit loading, one-call
auto-discovery, status, cancellation, streaming, JSON responses, and unload.

## Legacy benchmark

Run a persistent local Transformers model with streaming output and saved benchmark
diagnostics.

## Documentation

Serve the MkDocs Material site locally:

```bash
uv run mkdocs serve --dev-addr 127.0.0.1:8000
```

The API and documentation servers must use different ports. All inference
examples use `8013`; MkDocs uses `8000`.

Build the documentation with warnings treated as errors:

```bash
uv run mkdocs build --strict
```

Documentation focus:

- `docs/cli.md` covers `cognityx-inference` CLI commands and the Python client.
- `docs/http-api.md` covers OpenAI-compatible inference plus Cognityx control endpoints.
- `docs/jobs.md` covers discovery status streaming and cancellation.
- `docs/boundary-evaluation.md` and `docs/configuration.md` cover the finite
  search configuration, pruning behavior, and certification flow.
- `docs/usage.md` remains the legacy `llm-benchmark` interactive guide.

Before loading, a Hugging Face Hub dry run reports cached and missing checkpoint
bytes without downloading weights. Missing downloads require confirmation (No by
default). Scripts should combine `--non-interactive` with either `--allow-download`
or `--no-download`.

The loading menu offers strict GPU-only placement, BitsAndBytes INT4 nested
quantization, controlled GPU-first CPU offload, or a separate verified GPTQ 3-bit
or 2-bit checkpoint. CPU-offloaded results are classified separately from
GPU-only capacity results. The GPTQ choices never quantize an ordinary checkpoint
and may require `uv add gptqmodel`.

```bash
uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-8B --profile bf16
```

Generation defaults are read from `config.toml`. A different file can be selected
with `--config PATH`; interactive `/set` changes still apply to the running process.

```toml
[generation]
enable_thinking = true
max_new_tokens = 1024
temperature = 0.6
top_p = 0.95
top_k = 20
do_sample = true
repetition_penalty = 1.0
```

Load profiles distinguish checkpoint-native quantization from runtime quantization:

- `native` uses an embedded checkpoint `quantization_config` and fails if none exists.
- `auto` uses embedded quantization when present and otherwise defaults to BF16.
- `int4` and `int8` apply bitsandbytes only to unquantized checkpoints; requests for
  an already quantized checkpoint adapt to its native format with a warning.
- `int4-double` remains an INT4 NF4 profile and enables nested metadata
  quantization; it is not described as true 3.6-bit quantization.
- `gptq3` and `gptq2` require a separate checkpoint whose embedded configuration
  verifies the requested GPTQ bit width.

Retry an already cached large checkpoint with:

```bash
uv run python src/llm_benchmark/main.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --profile int4
```

## Long-context benchmarks

Prepare WikiText-103 Raw with the tokenizer belonging to the model being tested:

```bash
uv run python src/llm_benchmark/main.py \
  --model Qwen/Qwen3-32B \
  --prepare-context wikitext \
  --corpus wikitext-103-raw \
  --chunk-tokens 1024 \
  --chunk-overlap 128
```

Prepared JSONL chunks are stored under `data/contexts/wikitext/`. Run the normal
interactive application with a requested context budget, then enter `/benchmark`:

```bash
uv run python src/llm_benchmark/main.py \
  --model Qwen/Qwen3-32B \
  --profile int4 \
  --context wikitext \
  --context-tokens 32000
```

The application reserves `max_new_tokens`, measures the exact chat-templated input,
and reports whenever the final context chunk was shortened. It refuses prompts
that cannot fit instead of enabling tokenizer-side silent truncation.

While the model remains loaded, change or disable the context used by subsequent
`/benchmark` runs:

```text
/context wikitext 64000
/context
/context off
```

During streamed generation, press `Ctrl+C` once to cancel only the current request.
The decoder stops at the next safe generation step and returns to the `You>` prompt;
the model remains loaded, so `/context` and `/set` can be adjusted immediately.

The performance report includes a GPU-memory breakdown. Model and prompt tensor
storage are measured where PyTorch exposes them; KV-cache size and transient
prefill/decode memory are explicitly labelled estimates. Prefill is not reported as
a separate persistent cache because it uses the KV cache plus temporary activations
and workspaces. Quantized/fused runtime state may appear in the unclassified
allocated-memory remainder.

For native checkpoints, diagnostics separately report the checkpoint format and the
actual runtime path. MXFP4 checks use the installed Transformers requirements for GPU
capability, Triton, and `kernels`, then confirm any BF16 dequantization fallback from
the loaded model's quantizer state.

Saved schema-version 2 results retain the legacy `metrics` object for backward
compatibility. For new readers, `performance` is canonical for timing and throughput,
`model_runtime.model_size` is canonical for logical parameter count and weight-storage
estimates, and `quality_indicators` is canonical for structural completeness.

Timing definitions:

- `prompt_tokenization_seconds` includes chat-template rendering and CPU tokenization,
  but excludes moving the encoded tensors to the model device.
- `prefill_seconds` is the closest observable approximation: generation start until
  the first raw generated token reaches the streamer. It includes the first-token
  forward pass and is not a separately instrumented GPU kernel measurement.
- `time_to_first_streamed_output_seconds` ends when the first non-empty decoded text
  chunk is consumed; tokenizer buffering can make it later than the raw first token.
- `decode_seconds` is generation completion minus the raw first-token timestamp.
  Decode throughput excludes that first token.
- `end_to_end_seconds` spans prompt preparation, generation, decoding, parsing, and
  runtime measurement inside `LocalLLM.generate()`.

Logical parameters and physical storage are deliberately separate. Nominal raw weight
storage applies the selected profile's bit width to the logical parameter count; it is
not total VRAM. Quantization metadata, scales, packing, non-quantized layers, allocator
overhead, KV cache, activations, and runtime buffers add memory.

API-equivalent costs are disabled by default and use only prices entered by the user:

```text
/cost-model "User comparison" 2.50 10.00
/cost-status
/cost-off
```

They are hypothetical hosted-token-price comparisons, not measured local costs or
savings. Use `/compare` to compare saved benchmark runs; `/compare --help` is not
implemented, but `/help` lists supported filters and export formats.

For context-backed runs, reports store only a 24-word `prompt` preview instead of
duplicating the full corpus prompt. The original character count, exact prompt-token
count, SHA-256 hash, and compact `chunks_used_count` remain available for
traceability. Per-chunk details and full text remain in the prepared context JSONL
instead of being copied into every benchmark report.

## Optional vLLM engine

The default remains `--engine transformers`. vLLM is isolated from the working
Transformers environment:

```bash
uv venv .venv-vllm --python 3.12 --seed
uv pip install --python .venv-vllm/bin/python vllm bitsandbytes --torch-backend=auto
# Keep the CUDA compiler front end, assembler, headers, and CCCL coherent.
uv pip install --python .venv-vllm/bin/python \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85
```

The application automatically re-executes `--engine vllm` under `.venv-vllm`. On
this WSL RTX 5090 host it selects vLLM's legacy model runner and disables the
FlashInfer sampler because UVA is unavailable. The attention backend remains under
vLLM's control; FP8 KV-cache startup may JIT-compile FlashInfer attention kernels.

Start with the proven 10K corpus budget and normal KV cache:

```bash
uv run python src/llm_benchmark/main.py \
  --engine vllm \
  --model Qwen/Qwen3-32B \
  --profile int4 \
  --context wikitext \
  --context-tokens 10000 \
  --vllm-max-model-len 29616 \
  --kv-cache-dtype auto
```

Then repeat with FP8 KV cache:

```bash
uv run python src/llm_benchmark/main.py \
  --engine vllm \
  --model Qwen/Qwen3-32B \
  --profile int4 \
  --context wikitext \
  --context-tokens 10000 \
  --vllm-max-model-len 40960 \
  --kv-cache-dtype fp8 \
  --vllm-prefix-caching
```

`config.toml` supplies the same 8,192-token output limit, and `/context wikitext N`
can increase the corpus budget without reloading as long as the exact templated
prompt plus output reservation stays within `--vllm-max-model-len`.

vLLM prefix caching is enabled by default. Use `--no-vllm-prefix-caching` for a
cold-prefix comparison. Each vLLM result records queried prompt tokens as
`vllm:prefix_cache_queries` and reused cached tokens as `vllm:prefix_cache_hits`.
Like vLLM's counters, these values are cumulative for the lifetime of the loaded
engine and reset when the engine is restarted.

The vLLM adapter drives its persistent offline engine in delta-output mode, printing
decoded text progressively and retaining the complete output for reporting. Pressing
Ctrl+C aborts only the active vLLM request and keeps the engine loaded. Interactive
`/model` switching remains Transformers-only. FP8 KV cache without calibrated scales
is marked in diagnostics; it can increase cache capacity but may reduce accuracy.
