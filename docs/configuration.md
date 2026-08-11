# Configuration

## Manager, local server, and provider configuration

Copy the checked-in secret-free example:

```bash
mkdir -p .cognityx
cp configs/inference.toml.example .cognityx/inference.toml
export COGNITYX_INFERENCE_CONFIG="$PWD/.cognityx/inference.toml"
```

The file contains:

- `[manager]` for the lightweight management API;
- `[server_profiles.<name>]` for one named local worker configuration;
- `[providers.openai]` and `[providers.groq]` for endpoint metadata;
- provider model context capabilities used for safe token budgeting.
- optional `[tracking]` settings that index completed Storage research
  publications in MLflow.

Tracking is disabled unless selected explicitly:

```toml
[tracking]
backend = "none" # or "mlflow"
experiment_name = "cognityx-research"
failure_policy = "warn" # or "error"
# tracking_uri = "http://127.0.0.1:5000"
# parent_run_id = "optional-default-parent-run"
```

Storage remains authoritative. The default `warn` policy preserves a valid
Storage publication if MLflow is unavailable.

### Local secrets file

Normal configuration contains only a path and JSON entry names:

```toml
secrets_file = "/secure/location/provider-secrets.json"

[providers.openai]
adapter = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
api_key_secret = "openai_api_key"
```

The referenced file is a local JSON object:

```json
{
  "openai_api_key": "",
  "groq_api_key": ""
}
```

Credential precedence is:

1. the provider's environment variable;
2. its named entry in `secrets_file`.

The file is read lazily when a provider authenticates. Credential values are
never copied into the parsed configuration, API responses, diagnostics,
exceptions, logs, or saved inference artifacts. Set
`COGNITYX_SECRETS_FILE` to override the configured path for one process.

On Windows-mounted WSL filesystems, Unix permission bits may appear as `777`
even when Windows ACLs control access. Restrict the file using Windows ACLs and
do not place it inside a Git repository.

A local profile may select a specific saved certification:

```toml
[server_profiles.qwen3-8b-int4]
model = "Qwen/Qwen3-8B"
backend = "vllm"
load_profile = "int4"
certified_profile_id = "replace-with-a-saved-profile-id"
host = "127.0.0.1"
port = 8100

[server_profiles.qwen3-8b-int4.runtime]
kv_cache_precision = "fp8"
```

The selected profile must match the current model, backend, runtime, and
hardware fingerprint. Otherwise startup fails rather than silently using
incompatible evidence.

Provider configuration stores only the environment-variable name:

```toml
[providers.openai]
adapter = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
api_key_secret = "openai_api_key"
smoke_test_model = "configured-model-name"
```

Set actual credentials only in the process environment:

```bash
export OPENAI_API_KEY="..."
export GROQ_API_KEY="..."
```

`.env`, `.cognityx/inference.toml`, and local secret JSON/YAML paths are
ignored. The application does not
automatically read `.env`; use your shell or process supervisor to inject it.
Missing credentials are reported by variable name without exposing any value.

### Certified token capability

Each newly discovered local profile, and each configured remote model, may
provide:

```toml
[providers.example.models.example-model.context]
max_context_tokens = 131072
max_output_tokens_limit = 8192
reserved_tokens = 256
tokenizer = "tokenizer-or-model-identifier"
allow_estimated_counting = false
```

Remote models require a configured capability. Estimated counting is used only
when `allow_estimated_counting = true`, and response metadata labels it
`conservative_estimate`.

## Boundary Evaluation Config

The main boundary-evaluation configuration lives in:

`examples/boundary/config.toml`

It is designed to mirror the way Cognityx training receives explicit candidate
values per axis.

The long-running Inference service does not read this source-checkout example
when it starts. It uses the safe finite values built into `DiscoveryConfig`, so
an installed wheel starts the same way even when the repository's `examples/`
directory is not present. Pass the example explicitly to the boundary command
when you want to run that particular search recipe.

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

## Commercial provider configuration

Start from `configs/inference.toml.example`. Provider sections select an adapter,
credential names, default/smoke-test model, and optional model profiles:

```toml
secrets_file = "/secure/location/provider-secrets.json"

[providers.groq]
enabled = true
adapter = "openai_compatible"
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"
api_key_secret = "GROQ_API_KEY"
default_model = "llama-3.1-8b-instant"
default_profile = "groq-default"
```

Environment variables override the JSON secrets file. Lower-case legacy aliases
such as `groq_api_key` are accepted for migration, but uppercase names are the
documented format. Missing values produce `not_configured`, not a startup
failure or network call.

Model profiles can declare exact context and parameter policy as `unverified`,
`discovered`, `tested`, `certified`, or `deprecated`. Discovery confirms
provider availability; it does not certify a context limit. See
[Inference providers](providers/index.md).

Purpose routing is ordered and configuration-driven:

```toml
[[routing.primary_validation.preferred]]
provider = "groq"
profile = "groq-default"

[routing.sensitive_data]
allowed_providers = ["local"]
```

An explicit caller override is honored. Sensitive routes never fall back to an
unapproved provider.
