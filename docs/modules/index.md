# Module responsibilities

The package favors small functional modules over framework-level abstractions.

| Module | Responsibility | Inputs and outputs | Important collaborators |
| --- | --- | --- | --- |
| `main` | Minimal executable bootstrap. | Process arguments in; exit status out. | `app.main` |
| `app` | CLI parsing, interactive commands, orchestration, terminal ordering. | User commands and prompts in; terminal output and saved results out. | All workflow modules and both engines |
| `configuration` | Validate TOML generation defaults. | TOML path in; `GenerationSettings` out. | `llm` |
| `llm` | Persistent Transformers lifecycle, streaming, cancellation, parsing, metrics. | Model/profile/settings and prompts in; result dictionaries out. | Transformers, diagnostics, quantization, reporting |
| `vllm_engine` | Optional persistent vLLM compatibility adapter and delta streaming. | Same application-facing request shape as `LocalLLM`; compatible result dictionaries out. | vLLM, `llm` metric helpers |
| `contexts` | Context provider registry, corpus ingestion, chunk preparation, prompt budgeting. | Corpus or prepared context in; JSONL chunks or bounded prompt out. | Datasets, tokenizer, `app` |
| `download_preflight` | Dry-run checkpoint and disk inspection. | Model ID/path in; `DownloadPreflight` out. | Hugging Face Hub |
| `loading_decision` | Capacity estimates and explicit placement choices. | Hardware/profile/choice in; `ModelLoadOptions` and diagnostics out. | PyTorch, `model_size` |
| `model_selection` | Select a compatible Transformers generation loader. | Model config in; `ModelLoaderSelection` out. | Transformers AutoModel mappings |
| `quantization` | Resolve requested versus native quantization and runtime requirements. | Profile, config, hardware in; `QuantizationResolution` out. | Transformers, BitsAndBytes, package metadata |
| `model_loading` | Classify and report download, authentication, architecture, and placement failures. | Exception and attempted configuration in; structured error/report path out. | `reporting`, Hugging Face errors |
| `model_diagnostics` | Extract device, dtype, attention, cache, architecture, and quantization facts. | Loaded model/config in; JSON-ready runtime section out. | `model_architecture`, `model_size` |
| `model_architecture` | Dense/MoE classification and active-parameter estimates. | Generic model config in; family metadata out. | `model_diagnostics` |
| `model_size` | Logical parameter and physical-storage estimates. | Config/model/profile in; size metadata out. | Quantized model parameter metadata |
| `reporting` | CUDA/system inspection and collision-safe benchmark persistence. | Result mapping in; JSON file path out. | PyTorch, Transformers, filesystem |
| `comparison` | Normalize old/new result schemas and render comparisons. | Saved JSON and filters in; table/Markdown/CSV out. | `reporting` output tree |
| `cost_estimation` | Optional user-priced hosted API comparison. | Token counts and prices in; disclaimer-bearing estimate out. | `app` |
| `bkp_main` | Legacy prototype retained for reference, not invoked by the supported entry point. | Historical standalone workflow. | Superseded by `main`, `app`, and `llm` |

## Data boundaries

- `config.toml` contains generation defaults.
- `data/contexts/<name>/chunks.jsonl` contains prepared corpus chunks.
- `outputs/benchmarks/<model>/<profile>/` contains immutable run JSON files.
- `.venv-vllm` isolates vLLM and its CUDA build toolchain from the Transformers environment.
