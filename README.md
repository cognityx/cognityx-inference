# LLM Benchmark

Run a persistent local Transformers model with streaming output and saved benchmark
diagnostics.

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
