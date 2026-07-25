# Usage

This page documents the preserved legacy `llm-benchmark` workflow. For the
newer inference service, client, and API usage, see [CLI and Python
Client](cli.md) and [HTTP API](http-api.md).

## Install

```bash
uv sync --dev
```

Generation defaults are read from `config.toml`. Use `--config PATH` to select
a different file.

## Transformers engine

```bash
uv run python src/llm_benchmark/main.py \
  --engine transformers \
  --model Qwen/Qwen3-8B \
  --profile bf16
```

## vLLM engine

vLLM is intentionally installed in `.venv-vllm`; see the repository README for
the CUDA-aligned environment setup.

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

Use `--no-vllm-prefix-caching` for a cold-prefix comparison.

## Prepare a context

```bash
uv run python src/llm_benchmark/main.py \
  --model Qwen/Qwen3-32B \
  --prepare-context wikitext \
  --corpus wikitext-103-raw \
  --chunk-tokens 1024 \
  --chunk-overlap 128
```

Prepared data is stored under `data/contexts/<name>/`. Prompt assembly includes
full chunks until the requested budget is approached and trims only the final
chunk when required.

## Interactive commands

| Command | Purpose |
| --- | --- |
| `/benchmark` | Run the fixed travelling-salesman prompt through the normal pipeline. |
| `/context NAME TOKENS` | Change the active long-context budget without reloading. |
| `/context off` | Disable context injection. |
| `/set PARAMETER VALUE` | Change an active generation setting. |
| `/settings` | Display model, profile, context, and settings. |
| `/status` | Display runtime and latest-query diagnostics. |
| `/model MODEL [PROFILE]` | Switch models on the Transformers path. |
| `/compare ...` | Compare saved results. |
| `/help` | Display command help. |
| `/quit` | Exit and release the loaded model. |

Press `Ctrl+C` during generation to cancel the request while preserving the
loaded model. Pressing it outside generation follows normal terminal
interruption behavior.

## Documentation

```bash
uv run mkdocs serve --dev-addr 127.0.0.1:8000
uv run mkdocs build --strict
```

The inference API examples use port `8013`, so MkDocs on `8000` does not
intercept API requests.
