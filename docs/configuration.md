# Configuration

## Boundary Evaluation Config

The main boundary-evaluation configuration lives in:

`examples/boundary/config.toml`

It is designed to mirror the way Cognityx training receives explicit candidate
values per axis.

## Example Structure

```toml
[search]
ordered_axes = [
  "model",
  "backend",
  "quantization",
  "context_length",
  "kv_cache_precision",
  "batch_size",
  "generation_length",
]
monotonic_axes = ["context_length", "generation_length"]

[search.axes]
model = ["Qwen/Qwen3-8B"]
backend = ["vllm"]
quantization = ["int4"]
context_length = [2048, 4096, 8192, 16384, 32768, 40960]
kv_cache_precision = ["auto", "fp8"]
batch_size = [1]
generation_length = [1024, 2048, 4096]

[limits]
request_timeout_seconds = 300
first_token_timeout_seconds = 60
no_token_progress_timeout_seconds = 30
minimum_tokens_per_second = 2
slow_window_seconds = 10

[workload]
system_prompt = "You are a concise technical assistant."
user_prompt = "Briefly explain why model context capacity matters."
enable_thinking = false
temperature = 0.2
# top_p = 0.95
# top_k = 40
# min_p = 0.05
# seed = 42

[telemetry]
interval_seconds = 0.25
# windows_bridge_path = "/mnt/d/AI/cognityx/windows-performance.json"
windows_bridge_max_age_seconds = 5
```

## Axis Meanings

- `model`
  Model identifiers to test.
- `backend`
  Inference backend implementations. Current local options include `vllm` and
  `transformers`.
- `quantization`
  Runtime profile such as `bf16`, `fp16`, `int8`, or `int4`, subject to backend
  support.
- `context_length`
  Requested working context candidates.
- `kv_cache_precision`
  Cache precision choices such as `auto` and `fp8`.
- `batch_size`
  Included for throughput or batched inference evaluation. Keep `[1]` for
  single-request latency and capacity checks.
- `generation_length`
  Completion-length candidates to evaluate.

## Operational Limits

- `request_timeout_seconds`
  Whole-trial timeout budget.
- `first_token_timeout_seconds`
  Upper bound for waiting on first token progress in production-style
  workloads.
- `no_token_progress_timeout_seconds`
  Timeout once generation has stalled.
- `minimum_tokens_per_second`
  Threshold below which a run is considered impractically slow.
- `slow_window_seconds`
  Window used by the wider boundary model for slow-generation classification.

## Other Runtime Configuration

The load and inference APIs also accept backend-specific runtime fields under
`runtime` or `cognityx.extensions.runtime`, for example:

- `kv_cache_precision`
- `gpu_memory_utilization`
- `tensor_parallelism`
- `allow_download`

These fields do not replace the common normalized interface. They are carried as
backend-specific extensions where needed.

## Discovery Workload and Telemetry

`[workload]` defines the fixed representative request used for each discovery
trial. It records the system/user prompts, requested reasoning mode and common
generation settings with the resulting certification. This makes the capacity
limit traceable to the actual prompt and completion budget that produced it.

`[telemetry]` controls phase-aware sampling for model loading and inference.
NVIDIA counters provide dedicated memory, utilization, temperature and power;
an optional Windows bridge adds Windows-host CPU/RAM and shared GPU memory.
The final trial/profile stores average and peak measurements. Unsupported or
unavailable fields are `null`, never estimated.
