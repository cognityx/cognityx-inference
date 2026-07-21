# LLM Benchmark

Run a persistent local Transformers model with streaming output and saved benchmark
diagnostics.

```bash
uv run python src/llm_benchmark/main.py --model Qwen/Qwen3-8B --profile bf16
```

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
